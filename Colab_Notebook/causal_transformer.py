"""
Causal Transformer and Domain Normalization
============================================
Gold Standard Implementation for Cow Lameness Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DomainNorm(nn.Module):
    """
    Domain Normalization Layer.
    
    Purpose:
    - Handle domain shift between different farms/cameras
    - Normalize features before fusion
    - Improve generalization across conditions
    
    Based on konusma.md v29 requirements.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.layer_norm = nn.LayerNorm(dim, eps=eps)
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply domain normalization.
        
        Args:
            x: Input tensor of shape (..., dim)
            
        Returns:
            Normalized tensor of same shape
        """
        # Apply layer norm
        x = self.layer_norm(x)
        # Apply learnable scale and shift
        return x * self.scale + self.shift


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal sequences.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Tensor of shape (B, T, D)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CausalTransformerEncoder(nn.Module):
    """
    Causal Transformer Encoder.
    
    Key Feature:
    - Uses causal (upper triangular) mask to prevent information leakage
    - Each position can only attend to previous positions
    - Enables online/streaming prediction
    
    Based on konusma.md v30 requirements:
    "Model, gelecekteki frame'lerden bilgi sızdırmaz"
    """
    
    def __init__(self, 
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Cache for causal mask
        self._causal_mask = None
        
    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate causal attention mask.
        
        The mask prevents attention to future positions.
        mask[i,j] = True means position i cannot attend to position j.
        
        Args:
            seq_len: Sequence length
            device: Device for tensor
            
        Returns:
            Boolean mask of shape (seq_len, seq_len)
        """
        # Upper triangular matrix (excluding diagonal)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()
    
    def forward(self, x: torch.Tensor, use_causal: bool = True) -> torch.Tensor:
        """
        Forward pass with optional causal masking.
        
        Args:
            x: Input tensor of shape (B, T, D)
            use_causal: Whether to apply causal mask
            
        Returns:
            Encoded tensor of shape (B, T, D)
        """
        B, T, D = x.shape
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Generate causal mask if needed
        if use_causal:
            if self._causal_mask is None or self._causal_mask.size(0) != T:
                self._causal_mask = self._generate_causal_mask(T, x.device)
            mask = self._causal_mask
        else:
            mask = None
        
        # Apply transformer encoder
        output = self.transformer(x, mask=mask)
        
        return output


class MILAttention(nn.Module):
    """
    Multiple Instance Learning Attention Pooling.
    
    Aggregates multiple instances (windows) into a single bag representation
    using learned attention weights.
    
    Based on konusma.md requirements:
    "α_i = softmax(wᵀ tanh(W h_i))"
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Compute attention-weighted bag representation.
        
        Args:
            x: Instance features of shape (B, N, D)
               where N is number of instances (windows)
               
        Returns:
            bag: Aggregated bag representation (B, D)
            attention_weights: Attention weights (B, N)
        """
        # Compute attention scores
        attn_logits = self.attention(x).squeeze(-1)  # (B, N)
        
        # Apply softmax
        attn_weights = F.softmax(attn_logits, dim=1)  # (B, N)
        
        # Weighted sum
        bag = (x * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, D)
        
        return bag, attn_weights


class TemporalOrderNet(nn.Module):
    """
    Self-Supervised Temporal Order Verification Network.
    
    Pretext Task:
    - Given a sequence of frames, predict if the order is correct or reversed
    - Learns temporal motion patterns without labels
    
    Based on konusma.md v29 SSL requirements:
    "Temporal Order Verification - etiket yokken öğrenme"
    """
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Temporal pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Binary classifier: correct order vs reversed
        self.classifier = nn.Linear(hidden_dim, 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict temporal order of sequence.
        
        Args:
            x: Sequence features of shape (T, D)
            
        Returns:
            Logits for binary classification (1, 2)
        """
        # Encode each frame
        encoded = self.encoder(x)  # (T, hidden_dim)
        
        # Pool over time
        pooled = encoded.mean(dim=0)  # (hidden_dim,)
        
        # Classify
        return self.classifier(pooled).unsqueeze(0)


def pretrain_temporal_order(videos: list, 
                            videomae_encoder, 
                            epochs: int = 5,
                            device: str = "cuda") -> TemporalOrderNet:
    """
    Self-supervised pretraining using temporal order verification.
    
    Args:
        videos: List of video paths
        videomae_encoder: VideoMAE encoder to extract features
        epochs: Number of training epochs
        device: Device for training
        
    Returns:
        Pretrained TemporalOrderNet
    """
    import cv2
    import numpy as np
    
    tov = TemporalOrderNet(dim=768).to(device)
    optimizer = torch.optim.Adam(tov.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    print("Starting SSL Pretraining (Temporal Order Verification)...")
    
    for epoch in range(epochs):
        total_loss = 0
        
        for video_path in videos:
            # Load frames
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.resize(frame, (224, 224)))
            cap.release()
            
            if len(frames) < 16:
                continue
            
            # Sample a window
            idx = np.random.randint(0, len(frames) - 16)
            seq = np.array(frames[idx:idx + 16])
            
            # Randomly reverse with 50% probability
            is_reversed = np.random.rand() > 0.5
            if is_reversed:
                seq = seq[::-1].copy()
            
            # Extract features
            x = torch.tensor(seq).permute(0, 3, 1, 2).unsqueeze(0).float() / 255.0
            x = x.to(device)
            
            with torch.no_grad():
                feat = videomae_encoder.videomae(x.permute(0, 2, 1, 3, 4)).last_hidden_state.squeeze(0)
            
            # Predict order
            y = torch.tensor([1 if is_reversed else 0]).to(device)
            out = tov(feat)
            
            loss = loss_fn(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"  SSL Epoch {epoch + 1}/{epochs}: Loss = {total_loss / len(videos):.4f}")
    
    print("✅ SSL Pretraining complete!")
    return tov


if __name__ == "__main__":
    print("Testing Causal Transformer and MIL modules...")
    
    # Test DomainNorm
    dn = DomainNorm(256)
    x = torch.randn(2, 10, 256)
    out = dn(x)
    print(f"DomainNorm: {x.shape} -> {out.shape}")
    
    # Test CausalTransformerEncoder
    cte = CausalTransformerEncoder(d_model=256, nhead=8, num_layers=4)
    x = torch.randn(2, 10, 256)
    out = cte(x)
    print(f"CausalTransformer: {x.shape} -> {out.shape}")
    
    # Test MILAttention
    mil = MILAttention(256)
    x = torch.randn(2, 10, 256)
    bag, attn = mil(x)
    print(f"MILAttention: {x.shape} -> bag: {bag.shape}, attn: {attn.shape}")
    
    # Test TemporalOrderNet
    tov = TemporalOrderNet(768, 256)
    x = torch.randn(16, 768)
    out = tov(x)
    print(f"TemporalOrderNet: {x.shape} -> {out.shape}")
    
    print("✅ All tests passed!")

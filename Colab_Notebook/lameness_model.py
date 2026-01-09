"""
Lameness Severity Model
=======================
Gold Standard Implementation for Cow Lameness Detection
Supports both Binary Classification and Severity Regression (0-3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiModalFusion(nn.Module):
    """
    Multi-modal feature fusion layer.
    
    Combines:
    - Pose features (biomechanical)
    - Optical flow features (motion)
    - VideoMAE features (spatiotemporal)
    
    Based on konusma.md fusion strategy:
    "Pose → MLP → 128D, Flow → MLP → 64D, VideoMAE → Linear → 128D"
    """
    
    def __init__(self, 
                 pose_dim: int = 16,
                 flow_dim: int = 3,
                 video_dim: int = 128,
                 use_pose: bool = True,
                 use_flow: bool = True,
                 use_videomae: bool = True):
        super().__init__()
        
        self.use_pose = use_pose
        self.use_flow = use_flow
        self.use_videomae = use_videomae
        
        # Modal-specific encoders
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1)
        ) if use_pose else None
        
        self.flow_encoder = nn.Sequential(
            nn.Linear(flow_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.1)
        ) if use_flow else None
        
        self.video_encoder = nn.Sequential(
            nn.Linear(video_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1)
        ) if use_videomae else None
        
        # Calculate total dimension
        self.total_dim = 0
        if use_pose:
            self.total_dim += 128
        if use_flow:
            self.total_dim += 64
        if use_videomae:
            self.total_dim += 128
            
    def forward(self, 
                pose: Optional[torch.Tensor] = None,
                flow: Optional[torch.Tensor] = None,
                video: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Fuse multi-modal features.
        
        Args:
            pose: Pose features (B, N, W, pose_dim) or (B, N, pose_dim)
            flow: Flow features (B, N, flow_dim)
            video: Video features (B, N, video_dim)
            
        Returns:
            Fused features (B, N, total_dim)
        """
        features = []
        
        if self.use_pose and pose is not None:
            # Handle window dimension if present
            if pose.dim() == 4:
                B, N, W, D = pose.shape
                pose = pose.mean(dim=2)  # Aggregate over window
            features.append(self.pose_encoder(pose))
            
        if self.use_flow and flow is not None:
            features.append(self.flow_encoder(flow))
            
        if self.use_videomae and video is not None:
            features.append(self.video_encoder(video))
        
        if len(features) == 0:
            raise ValueError("At least one modality must be enabled!")
            
        return torch.cat(features, dim=-1)


class LamenessSeverityModel(nn.Module):
    """
    Complete Lameness Detection Model with Severity Regression.
    
    Architecture:
    1. Multi-modal fusion (Pose + Flow + VideoMAE)
    2. Domain normalization
    3. Causal transformer encoder
    4. MIL attention pooling
    5. Severity regression head (0-3 score)
    
    Based on konusma.md v30 requirements:
    - Causal temporal modeling
    - Severity regression (ordinal/continuous)
    - Multiple Instance Learning
    """
    
    def __init__(self,
                 config: dict,
                 mode: str = "regression"):  # "regression" or "classification"
        super().__init__()
        
        self.config = config
        self.mode = mode
        
        # Multi-modal fusion
        self.fusion = MultiModalFusion(
            pose_dim=config.get("POSE_DIM", 16),
            flow_dim=config.get("FLOW_DIM", 3),
            video_dim=config.get("VIDEO_DIM", 128),
            use_pose=config.get("USE_POSE", True),
            use_flow=config.get("USE_FLOW", True),
            use_videomae=config.get("USE_VIDEOMAE", False)
        )
        
        total_dim = self.fusion.total_dim
        
        # Domain normalization
        self.domain_norm = nn.LayerNorm(total_dim)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(total_dim)
        
        # Causal transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=total_dim,
            nhead=config.get("NUM_HEADS", 8),
            dim_feedforward=total_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.get("NUM_LAYERS", 4)
        )
        
        # MIL attention
        self.attention = nn.Sequential(
            nn.Linear(total_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Output head
        if mode == "regression":
            # Severity regression (0-3 continuous)
            self.head = nn.Sequential(
                nn.Linear(total_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
                nn.Sigmoid()  # Output in [0, 1], scale to [0, 3]
            )
        else:
            # Binary classification
            self.head = nn.Sequential(
                nn.Linear(total_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        
        # Causal mask cache
        self._causal_mask = None
        
    def _create_positional_encoding(self, d_model: int, max_len: int = 500):
        """Create sinusoidal positional encoding."""
        import math
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        if self._causal_mask is None or self._causal_mask.size(0) != seq_len:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
            self._causal_mask = mask.bool()
        return self._causal_mask
        
    def forward(self,
                pose: Optional[torch.Tensor] = None,
                flow: Optional[torch.Tensor] = None,
                video: Optional[torch.Tensor] = None,
                use_causal: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            pose: Pose features
            flow: Flow features
            video: VideoMAE features
            use_causal: Whether to apply causal masking
            
        Returns:
            prediction: Lameness score or probability
            attention_weights: Window-level attention weights
        """
        # Fuse modalities
        x = self.fusion(pose, flow, video)  # (B, N, D)
        B, N, D = x.shape
        
        # Apply domain normalization
        x = self.domain_norm(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :N, :]
        
        # Apply causal transformer
        if use_causal:
            mask = self._get_causal_mask(N, x.device)
        else:
            mask = None
        x = self.transformer(x, mask=mask)
        
        # MIL attention pooling
        attn_logits = self.attention(x).squeeze(-1)  # (B, N)
        attn_weights = F.softmax(attn_logits, dim=1)  # (B, N)
        bag = (x * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, D)
        
        # Get prediction
        pred = self.head(bag).squeeze(-1)  # (B,)
        
        # Scale to [0, 3] for regression mode
        if self.mode == "regression":
            pred = pred * 3.0
        
        return pred, attn_weights


class TrainingManager:
    """
    Training manager with LR groups, checkpointing, and resume support.
    
    Features:
    - Layer-wise learning rates (lower for VideoMAE)
    - Gradient clipping
    - Checkpoint saving and loading
    - Resume from checkpoint
    """
    
    def __init__(self,
                 model: nn.Module,
                 videomae_encoder: Optional[nn.Module] = None,
                 config: dict = None,
                 device: str = "cuda"):
        
        self.model = model.to(device)
        self.videomae = videomae_encoder.to(device) if videomae_encoder else None
        self.config = config or {}
        self.device = device
        
        # Setup optimizer with LR groups
        self.optimizer = self._create_optimizer()
        
        # Loss function
        if config.get("MODE") == "regression":
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.BCELoss()
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def _create_optimizer(self):
        """Create optimizer with layer-wise learning rates."""
        param_groups = []
        
        # Main model parameters
        main_lr = self.config.get("LR", 1e-4)
        param_groups.append({
            "params": self.model.parameters(),
            "lr": main_lr
        })
        
        # VideoMAE parameters (lower LR)
        if self.videomae is not None:
            videomae_lr = self.config.get("LR_VIDEOMAE", 1e-5)
            param_groups.append({
                "params": self.videomae.parameters(),
                "lr": videomae_lr
            })
        
        return torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.get("WEIGHT_DECAY", 1e-4)
        )
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "config": self.config
        }
        
        if self.videomae is not None:
            checkpoint["videomae_state_dict"] = self.videomae.state_dict()
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = path.replace(".pt", "_best.pt")
            torch.save(checkpoint, best_path)
            
    def load_checkpoint(self, path: str):
        """Load training checkpoint and resume."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        
        if self.videomae is not None and "videomae_state_dict" in checkpoint:
            self.videomae.load_state_dict(checkpoint["videomae_state_dict"])
            
        print(f"Resumed from epoch {self.epoch}, best val loss: {self.best_val_loss:.4f}")
        
    def train_step(self, batch):
        """Single training step."""
        self.model.train()
        if self.videomae:
            self.videomae.train()
        
        pose, flow, video, label = batch
        pose = pose.to(self.device) if pose is not None else None
        flow = flow.to(self.device) if flow is not None else None
        video = video.to(self.device) if video is not None else None
        label = label.to(self.device)
        
        self.optimizer.zero_grad()
        
        pred, attn = self.model(pose, flow, video)
        loss = self.criterion(pred, label)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        if self.videomae:
            torch.nn.utils.clip_grad_norm_(self.videomae.parameters(), 1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def eval_step(self, batch):
        """Single evaluation step."""
        self.model.eval()
        if self.videomae:
            self.videomae.eval()
        
        with torch.no_grad():
            pose, flow, video, label = batch
            pose = pose.to(self.device) if pose is not None else None
            flow = flow.to(self.device) if flow is not None else None
            video = video.to(self.device) if video is not None else None
            label = label.to(self.device)
            
            pred, attn = self.model(pose, flow, video)
            loss = self.criterion(pred, label)
        
        return loss.item(), pred.cpu(), attn.cpu()


if __name__ == "__main__":
    print("Testing Lameness Severity Model...")
    
    config = {
        "POSE_DIM": 16,
        "FLOW_DIM": 3,
        "VIDEO_DIM": 128,
        "USE_POSE": True,
        "USE_FLOW": True,
        "USE_VIDEOMAE": True,
        "NUM_HEADS": 8,
        "NUM_LAYERS": 4,
        "LR": 1e-4,
        "LR_VIDEOMAE": 1e-5,
        "MODE": "regression"
    }
    
    model = LamenessSeverityModel(config, mode="regression")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    B, N, W = 2, 10, 60
    pose = torch.randn(B, N, W, 16)
    flow = torch.randn(B, N, 3)
    video = torch.randn(B, N, 128)
    
    pred, attn = model(pose, flow, video)
    print(f"Prediction shape: {pred.shape}")  # (B,)
    print(f"Attention shape: {attn.shape}")   # (B, N)
    print(f"Prediction range: [{pred.min():.2f}, {pred.max():.2f}]")
    
    print("✅ Lameness Severity Model test passed!")

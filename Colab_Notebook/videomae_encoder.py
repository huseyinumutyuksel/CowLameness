"""
VideoMAE Encoder with Partial Fine-Tuning
==========================================
Gold Standard Implementation for Cow Lameness Detection
"""

import torch
import torch.nn as nn
from transformers import VideoMAEModel, VideoMAEImageProcessor
import cv2
import numpy as np


class VideoMAEEncoder(nn.Module):
    """
    VideoMAE encoder with partial fine-tuning strategy.
    
    Architecture:
    - Blocks 0-8: FROZEN (preserve general motion representation)
    - Blocks 9-11: TRAINABLE (adapt to lameness-specific patterns)
    - Projection layer: Maps 768D to 128D
    
    This approach:
    - Prevents overfitting on small datasets
    - Preserves temporal inductive bias
    - Allows task-specific adaptation
    """
    
    def __init__(self, model_name: str = "MCG-NJU/videomae-base", 
                 output_dim: int = 128,
                 trainable_blocks: list = [9, 10, 11]):
        super().__init__()
        
        self.model_name = model_name
        self.output_dim = output_dim
        self.trainable_blocks = trainable_blocks
        
        # Load pretrained VideoMAE
        self.videomae = VideoMAEModel.from_pretrained(
            model_name,
            output_hidden_states=True
        )
        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
        
        # Apply partial fine-tuning strategy
        self._apply_freeze_strategy()
        
        # Projection layer (768D -> output_dim)
        self.projection = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def _apply_freeze_strategy(self):
        """
        Freeze early layers, keep later layers trainable.
        
        Strategy based on konusma.md requirements:
        - Early blocks: General motion features (frozen)
        - Late blocks: Task-specific adaptation (trainable)
        """
        # Freeze all parameters first
        for param in self.videomae.parameters():
            param.requires_grad = False
        
        # Unfreeze specified blocks
        for name, param in self.videomae.named_parameters():
            for block_idx in self.trainable_blocks:
                if f"encoder.layer.{block_idx}" in name:
                    param.requires_grad = True
                    break
        
        # Count trainable vs frozen parameters
        trainable = sum(p.numel() for p in self.videomae.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.videomae.parameters())
        print(f"VideoMAE: {trainable:,}/{total:,} parameters trainable ({100*trainable/total:.1f}%)")
    
    def preprocess_frames(self, frames: np.ndarray) -> torch.Tensor:
        """
        Preprocess video frames for VideoMAE input.
        
        Args:
            frames: numpy array of shape (T, H, W, C) in BGR format
            
        Returns:
            Tensor of shape (1, T, C, H, W) normalized
        """
        # Convert BGR to RGB
        frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        
        # Resize to 224x224
        frames_resized = [cv2.resize(f, (224, 224)) for f in frames_rgb]
        
        # Stack and normalize
        frames_array = np.stack(frames_resized)
        frames_tensor = torch.tensor(frames_array).permute(0, 3, 1, 2).float() / 255.0
        
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        frames_tensor = (frames_tensor - mean) / std
        
        return frames_tensor.unsqueeze(0)  # Add batch dimension
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Video tensor of shape (B, T, C, H, W)
            
        Returns:
            Video features of shape (B, output_dim)
        """
        B, T, C, H, W = x.shape
        
        # VideoMAE expects (B, C, T, H, W) format
        x = x.permute(0, 2, 1, 3, 4)
        
        # Get VideoMAE output
        outputs = self.videomae(pixel_values=x)
        
        # Get last hidden state: (B, num_patches, 768)
        hidden_states = outputs.last_hidden_state
        
        # Temporal attention: Use CLS token or mean pooling
        # Here we use mean pooling over all tokens for better temporal coverage
        video_features = hidden_states.mean(dim=1)  # (B, 768)
        
        # Project to output dimension
        return self.projection(video_features)  # (B, output_dim)
    
    def extract_temporal_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract per-window temporal features (for MIL).
        
        Args:
            x: Video tensor of shape (B, T, C, H, W)
            
        Returns:
            Temporal features of shape (B, T//patch_size, output_dim)
        """
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        
        outputs = self.videomae(pixel_values=x)
        hidden_states = outputs.last_hidden_state  # (B, num_patches, 768)
        
        # Project each patch
        temporal_feats = self.projection(hidden_states)  # (B, num_patches, output_dim)
        
        return temporal_feats


def extract_videomae_features(video_path: str, 
                               encoder: VideoMAEEncoder,
                               window_size: int = 16,
                               stride: int = 8,
                               device: str = "cuda") -> np.ndarray:
    """
    Extract VideoMAE features from a video file using sliding windows.
    
    Args:
        video_path: Path to video file
        encoder: VideoMAEEncoder instance
        window_size: Number of frames per window
        stride: Step between windows
        device: Device to run inference on
        
    Returns:
        Features array of shape (N_windows, output_dim)
    """
    # Load video frames
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    if len(frames) < window_size:
        return None
    
    frames = np.array(frames)
    
    # Extract features for each window
    features = []
    encoder.eval()
    encoder.to(device)
    
    with torch.no_grad():
        for start in range(0, len(frames) - window_size + 1, stride):
            window = frames[start:start + window_size]
            
            # Preprocess
            x = encoder.preprocess_frames(window).to(device)
            
            # Extract features
            feat = encoder(x)
            features.append(feat.cpu().numpy().squeeze())
    
    return np.array(features) if features else None


if __name__ == "__main__":
    # Test the encoder
    print("Testing VideoMAE Encoder...")
    encoder = VideoMAEEncoder()
    print(f"Output dimension: {encoder.output_dim}")
    
    # Test with dummy input
    dummy_input = torch.randn(1, 16, 3, 224, 224)
    output = encoder(dummy_input.to("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Output shape: {output.shape}")
    print("✅ VideoMAE Encoder test passed!")

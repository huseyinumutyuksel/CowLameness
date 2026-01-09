"""
Complete Gold Standard Pipeline
================================
Integrates all modules for cow lameness detection:
- VideoMAE with partial fine-tuning
- Causal Transformer with MIL
- Severity Regression (0-3)
- Domain Normalization
- Self-Supervised Pretraining
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
from typing import Optional, List, Dict, Tuple
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

# Import custom modules
from videomae_encoder import VideoMAEEncoder, extract_videomae_features
from causal_transformer import DomainNorm, CausalTransformerEncoder, MILAttention
from lameness_model import LamenessSeverityModel, MultiModalFusion, TrainingManager
from gait_features import GaitFeatureExtractor


# =============================================================================
# CONFIGURATION
# =============================================================================

CFG = {
    # Paths (from user's v20)
    "VIDEO_DIR": "/content/drive/MyDrive/Inek Topallik Tespiti Parcalanmis Inek Videolari/cow_single_videos",
    "POSE_DIR": "/content/drive/MyDrive/DeepLabCut/outputs",
    "FLOW_DIR": "/content/optical_flow",
    "VIDEOMAE_DIR": "/content/videomae_features",
    "MODEL_DIR": "/content/models",
    "RESULT_DIR": "/content/results",
    
    # Video processing
    "FPS": 30,
    "WINDOW_FRAMES": 60,
    "STRIDE_FRAMES": 15,
    
    # Model dimensions
    "POSE_DIM": 16,
    "FLOW_DIM": 3,
    "VIDEO_DIM": 128,
    "HIDDEN_DIM": 256,
    "NUM_HEADS": 8,
    "NUM_LAYERS": 4,
    
    # Training
    "BATCH_SIZE": 1,
    "EPOCHS": 30,
    "LR": 1e-4,
    "LR_VIDEOMAE": 1e-5,
    "WEIGHT_DECAY": 1e-4,
    
    # Ablation flags
    "USE_POSE": True,
    "USE_FLOW": True,
    "USE_VIDEOMAE": True,  # NOW ENABLED
    
    # Mode
    "MODE": "regression",  # "regression" or "classification"
    "USE_CAUSAL": True,    # Enable causal attention
    "USE_SSL": False,      # Self-supervised pretraining
    
    # Reproducibility
    "SEED": 42
}


# =============================================================================
# DATASET
# =============================================================================

class CowLamenessDatasetV2(Dataset):
    """
    Enhanced dataset with VideoMAE support and severity labels.
    """
    
    def __init__(self, 
                 video_list: List[str],
                 labels: List[float],  # Now supports continuous 0-3 severity
                 config: dict,
                 pose_dir: str,
                 videomae_encoder: Optional[VideoMAEEncoder] = None):
        
        self.video_list = video_list
        self.labels = labels
        self.config = config
        self.pose_dir = pose_dir
        self.videomae_encoder = videomae_encoder
        self.gait_extractor = GaitFeatureExtractor(fps=config["FPS"])
        
    def __len__(self):
        return len(self.video_list)
    
    def __getitem__(self, idx: int) -> Tuple:
        video_path = self.video_list[idx]
        label = self.labels[idx]
        video_name = Path(video_path).stem
        
        pose_feat = None
        flow_feat = None
        video_feat = None
        
        # Determine subfolder based on label
        sub_folder = "Saglikli" if label < 0.5 else "Topal"
        
        # Extract pose features
        if self.config["USE_POSE"]:
            pose_pattern = f"{self.pose_dir}/{sub_folder}/{video_name}*DLC*.csv"
            pose_files = glob(pose_pattern)
            
            if pose_files:
                try:
                    pose_df = pd.read_csv(pose_files[0], header=[1, 2])
                    pose_feat = self.gait_extractor.extract_features(pose_df)
                    if pose_feat is not None:
                        pose_feat = self._sliding_windows(
                            pose_feat, 
                            self.config["WINDOW_FRAMES"],
                            self.config["STRIDE_FRAMES"]
                        )
                except Exception:
                    pass
        
        # Extract flow features
        if self.config["USE_FLOW"]:
            flow_feat = self._extract_flow(video_path)
        
        # Extract VideoMAE features
        if self.config["USE_VIDEOMAE"] and self.videomae_encoder is not None:
            video_feat = extract_videomae_features(
                video_path, 
                self.videomae_encoder,
                window_size=16,
                stride=8
            )
        
        # Get number of windows
        n_windows = self._get_n_windows(pose_feat, flow_feat, video_feat)
        
        # Handle None cases
        if pose_feat is None:
            pose_feat = np.zeros((n_windows, self.config["WINDOW_FRAMES"], 16))
        if flow_feat is None:
            flow_feat = np.zeros((n_windows, 3))
        if video_feat is None:
            video_feat = np.zeros((n_windows, 128))
        
        # Align dimensions
        min_len = min(len(pose_feat), len(flow_feat), len(video_feat))
        pose_feat = pose_feat[:min_len]
        flow_feat = flow_feat[:min_len]
        video_feat = video_feat[:min_len]
        
        return (
            torch.tensor(pose_feat, dtype=torch.float32),
            torch.tensor(flow_feat, dtype=torch.float32),
            torch.tensor(video_feat, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32)
        )
    
    def _sliding_windows(self, features: np.ndarray, 
                         window_size: int, stride: int) -> np.ndarray:
        if features is None or len(features) < window_size:
            return None
        windows = []
        for start in range(0, len(features) - window_size + 1, stride):
            windows.append(features[start:start + window_size])
        return np.stack(windows) if windows else None
    
    def _extract_flow(self, video_path: str) -> np.ndarray:
        import cv2
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        cap.release()
        
        if len(frames) < 16:
            return None
        
        features = []
        for i in range(0, len(frames) - 16, 8):
            window = frames[i:i + 16]
            mags = []
            prev = window[0]
            for f in window[1:]:
                flow = cv2.calcOpticalFlowFarneback(
                    prev, f, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                mags.append([mag.mean(), mag.var(), np.std(ang)])
                prev = f
            features.append(np.mean(mags, axis=0))
        
        return np.array(features) if features else None
    
    def _get_n_windows(self, *args) -> int:
        for feat in args:
            if feat is not None:
                return len(feat)
        return 1


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, loader, optimizer, criterion, device, config):
    """Single training epoch."""
    model.train()
    total_loss = 0
    
    for pose, flow, video, label in loader:
        pose = pose.to(device)
        flow = flow.to(device)
        video = video.to(device) if config["USE_VIDEOMAE"] else None
        label = label.to(device)
        
        optimizer.zero_grad()
        pred, attn = model(pose, flow, video, use_causal=config["USE_CAUSAL"])
        loss = criterion(pred, label)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def eval_epoch(model, loader, criterion, device, config):
    """Single evaluation epoch."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_attns = []
    
    with torch.no_grad():
        for pose, flow, video, label in loader:
            pose = pose.to(device)
            flow = flow.to(device)
            video = video.to(device) if config["USE_VIDEOMAE"] else None
            label = label.to(device)
            
            pred, attn = model(pose, flow, video, use_causal=config["USE_CAUSAL"])
            loss = criterion(pred, label)
            
            total_loss += loss.item()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_attns.append(attn.cpu().numpy())
    
    return total_loss / len(loader), np.array(all_preds), np.array(all_labels), all_attns


def run_training(config: dict, device: str = "cuda"):
    """Full training pipeline."""
    
    # Create directories
    for d in [config["FLOW_DIR"], config["VIDEOMAE_DIR"], 
              config["MODEL_DIR"], config["RESULT_DIR"]]:
        os.makedirs(d, exist_ok=True)
    
    # Load video lists
    healthy_videos = glob(f"{config['VIDEO_DIR']}/Saglikli/*.mp4")
    lame_videos = glob(f"{config['VIDEO_DIR']}/Topal/*.mp4")
    
    print(f"Found {len(healthy_videos)} healthy, {len(lame_videos)} lame videos")
    
    # Create labels (0 for healthy, 3 for lame in regression mode)
    if config["MODE"] == "regression":
        labels = [0.0] * len(healthy_videos) + [3.0] * len(lame_videos)
    else:
        labels = [0.0] * len(healthy_videos) + [1.0] * len(lame_videos)
    
    all_videos = healthy_videos + lame_videos
    
    # Split
    train_videos, test_videos, train_labels, test_labels = train_test_split(
        all_videos, labels, test_size=0.2, stratify=[1 if l > 0.5 else 0 for l in labels],
        random_state=config["SEED"]
    )
    
    # Initialize VideoMAE encoder if needed
    videomae_encoder = None
    if config["USE_VIDEOMAE"]:
        videomae_encoder = VideoMAEEncoder().to(device)
    
    # Create datasets
    train_dataset = CowLamenessDatasetV2(
        train_videos, train_labels, config, config["POSE_DIR"], videomae_encoder
    )
    test_dataset = CowLamenessDatasetV2(
        test_videos, test_labels, config, config["POSE_DIR"], videomae_encoder
    )
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Create model
    model = LamenessSeverityModel(config, mode=config["MODE"]).to(device)
    
    # Optimizer with LR groups
    param_groups = [{"params": model.parameters(), "lr": config["LR"]}]
    if videomae_encoder is not None:
        param_groups.append({"params": videomae_encoder.parameters(), "lr": config["LR_VIDEOMAE"]})
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config["WEIGHT_DECAY"])
    
    # Loss function
    criterion = nn.MSELoss() if config["MODE"] == "regression" else nn.BCELoss()
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    for epoch in range(config["EPOCHS"]):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, config)
        val_loss, preds, labels_arr, attns = eval_epoch(model, test_loader, criterion, device, config)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model': model.state_dict(),
                'videomae': videomae_encoder.state_dict() if videomae_encoder else None,
                'epoch': epoch,
                'val_loss': val_loss,
                'config': config
            }, f"{config['MODEL_DIR']}/best_model.pt")
        
        if (epoch + 1) % 5 == 0:
            if config["MODE"] == "regression":
                mae = mean_absolute_error(labels_arr, preds)
                print(f"Epoch {epoch+1}/{config['EPOCHS']} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | MAE: {mae:.2f}")
            else:
                acc = ((preds > 0.5) == labels_arr).mean()
                print(f"Epoch {epoch+1}/{config['EPOCHS']} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {acc:.2%}")
    
    print(f"\n✅ Training complete. Best val loss: {best_val_loss:.4f}")
    
    return model, videomae_encoder, train_losses, val_losses


if __name__ == "__main__":
    import random
    
    # Set seed
    random.seed(CFG["SEED"])
    np.random.seed(CFG["SEED"])
    torch.manual_seed(CFG["SEED"])
    torch.cuda.manual_seed_all(CFG["SEED"])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Run training
    model, videomae, train_losses, val_losses = run_training(CFG, device)

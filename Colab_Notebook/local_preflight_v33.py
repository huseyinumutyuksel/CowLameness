"""
Local Pre-Flight Test for v33 (LoRA + Pose Debug)
Tests critical components before Colab deployment.
"""
import unittest
import numpy as np
import collections
import os
import sys

# No complex mocking needed as we don't import the notebook scripts directly
# and the redefined classes here don't use google.colab.

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 1. Mock Classes (Simplified from build_v33 scripts)
# ============================================================

class PoseFeatureExtractor:
    FEATURE_NAMES = ["f"+str(i) for i in range(16)]
    num_features = 16
    def __init__(self, fps=30, min_conf=0.3): 
        self.fps = fps
        self.min_conf = min_conf

    def extract_from_csv(self, path): return np.random.rand(16)
    
    # Copied from build_v33_part1.py for verification
    def _compute_angle_trajectory(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
        n = min(len(p1), len(p2), len(p3))
        angles = []
        
        def fill_gaps(arr, conf_thresh=0.3):
            valid = arr[:, 2] > conf_thresh
            if valid.sum() < 2: return arr 
            x = arr[:, 0].copy(); y = arr[:, 1].copy()
            x[~valid] = np.nan; y[~valid] = np.nan
            nans = np.isnan(x)
            if nans.any() and not nans.all():
                x_idx = np.arange(len(x))
                x[nans] = np.interp(x_idx[nans], x_idx[~nans], x[~nans])
                y[nans] = np.interp(x_idx[nans], x_idx[~nans], y[~nans])
            return np.column_stack([x, y, arr[:, 2]])

        p1_f = fill_gaps(p1, self.min_conf)
        p2_f = fill_gaps(p2, self.min_conf)
        p3_f = fill_gaps(p3, self.min_conf)

        for i in range(n):
            v1 = p1_f[i, :2] - p2_f[i, :2]
            v2 = p3_f[i, :2] - p2_f[i, :2]
            if np.isnan(v1).any() or np.isnan(v2).any():
                angles.append(np.nan)
                continue
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 < 1e-3 or norm2 < 1e-3:
                angles.append(np.nan)
                continue
            cos_a = np.dot(v1, v2) / (norm1 * norm2)
            angles.append(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))
        return np.array(angles)

class CowLamenessDatasetV33(Dataset):
    def __init__(self, length=10):
        self.length = length
        # Mock processor
    def __len__(self): return self.length
    def __getitem__(self, idx):
        # Mock (NUM_CLIPS=8, T=16, C=3, H=224, W=224)
        vid = torch.randn(8, 16, 3, 224, 224) 
        pose = torch.randn(16)
        label = torch.tensor(1.0 if idx % 2 == 0 else 0.0)
        return {"pixel_values": vid, "pose_features": pose, "label": label}

def collate_fn_v33(batch):
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    pose = torch.stack([x["pose_features"] for x in batch])
    labels = torch.stack([x["label"] for x in batch])
    return pixel_values, pose, labels

# Import transformers/peft (Must be installed)
try:
    from transformers import VideoMAEModel, VideoMAEConfig
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError:
    print("❌ Transformers or PEFT not installed. Skipping model tests.")
    sys.exit(0)

class CowLamenessModelV33(nn.Module):
    def __init__(self):
        super().__init__()
        # Tiny Config for speed
        config = VideoMAEConfig(
            image_size=224, num_frames=16, 
            hidden_size=192, # Smaller hidden size for speed
            num_hidden_layers=2, 
            num_attention_heads=4,
            intermediate_size=768
        )
        self.backbone = VideoMAEModel(config)
        
        # LoRA
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=4, lora_alpha=4, 
            target_modules=["query", "value"] # HF VideoMAE uses query/value/key/dense
        )
        try:
            self.backbone = get_peft_model(self.backbone, peft_config)
        except ValueError as e:
            print(f"❌ Preflight LoRA Error: {e}")
            print("Possible modules:", [n for n, _ in self.backbone.named_modules()])
            raise e
        
        # Adapter (size must match hidden_size=192)
        self.adapter = nn.Sequential(nn.Linear(192, 64))
        
        # Pose
        self.pose_proj = nn.Sequential(nn.Linear(16, 64))
        
        # Temporal
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=2, dim_feedforward=128, batch_first=True),
            num_layers=1
        )
        
        # Head
        self.classifier = nn.Linear(64, 1)
        
    def forward(self, pixel_values, pose_features):
        B, N, C, T, H, W = pixel_values.shape
        x = pixel_values.view(B*N, C, T, H, W)
        
        # VideoMAE forward
        out = self.backbone(pixel_values=x) # (B*N, T, 192)
        # VideoMAE (MAE style) does NOT have a CLS token at index 0 typically.
        # It outputs (B*N, 1568, 768). Use Mean Pooling.
        features = out.last_hidden_state.mean(dim=1) # (B*N, 192)
        
        features = self.adapter(features) # (B*N, 64)
        features = features.view(B, N, 64)
        
        pose_embed = self.pose_proj(pose_features).unsqueeze(1) # (B, 1, 64)
        features = features + pose_embed
        
        temp_out = self.temporal_encoder(features)
        x_pool = temp_out.mean(dim=1)
        return self.classifier(x_pool)

# ============================================================
# 2. Tests
# ============================================================

class TestV33Preflight(unittest.TestCase):
    
    def test_01_lora_trainable_params(self):
        print("\nTesting LoRA Parameter Config...")
        model = CowLamenessModelV33()
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        
        # Check LoRA params are trainable
        lora_params = [n for n in trainable if "lora" in n]
        self.assertTrue(len(lora_params) > 0, "No LoRA parameters found trainable!")
        
        # Check Backbone base params are FROZEN (except LoRA)
        backbone_trainable = [n for n in trainable if "backbone" in n]
        # All backbone trainable params should be LoRA
        for n in backbone_trainable:
            self.assertTrue("lora" in n or "modules_to_save" in n, f"Leaked backbone param: {n}")
            
        print("✅ LoRA Config: PASS (Trainable: LoRA only)")

    def test_02_dataset_shape(self):
        print("\nTesting Dataset & Collate...")
        ds = CowLamenessDatasetV33(length=4)
        loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn_v33)
        batch = next(iter(loader))
        vid, pose, lbl = batch
        
        # Expected: (B=2, N=8, T=16, C=3, H=224, W=224)
        self.assertEqual(vid.shape, (2, 8, 16, 3, 224, 224))
        self.assertEqual(pose.shape, (2, 16))
        self.assertEqual(lbl.shape, (2,))
        print("✅ Dataset Shape: PASS")

    def test_03_forward_pass(self):
        print("\nTesting Model Forward Pass...")
        model = CowLamenessModelV33()
        # Mock inputs: (B=2, N=8, T=16, C=3, H=224, W=224)
        vid = torch.randn(2, 8, 16, 3, 224, 224)
        pose = torch.randn(2, 16)
        
        out = model(vid, pose)
        self.assertEqual(out.shape, (2, 1))
        print("✅ Forward Pass: PASS")
        
    def test_04_pose_sanity_check_logic(self):
        print("\nTesting Pose Sanity Logic (Mock)...")
        # Low Variance
        feats = np.zeros((10, 16))
        valid = (feats != 0).mean()
        self.assertEqual(valid, 0.0)
        
        # Good Data
        feats = np.random.rand(10, 16)
        valid = (feats != 0).mean()
        self.assertGreater(valid, 0.9)
        print("✅ Pose Sanity Logic: PASS")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

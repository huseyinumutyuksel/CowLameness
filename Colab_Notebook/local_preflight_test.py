"""
Local Pre-Flight Test — v32 Notebook Bileşen Doğrulaması
=========================================================
CPU ortamında tüm model, veri, eğitim bileşenlerini test eder.
Colab'a yüklemeden önce tüm hataları yakalar.

Çalıştırma:
  python local_preflight_test.py
"""
import sys, os, time, traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
from typing import Optional, List, Tuple
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

DEVICE = "cpu"
PASS_COUNT = 0
FAIL_COUNT = 0

def test_header(name):
    print(f"\n{'='*60}")
    print(f"  TEST: {name}")
    print(f"{'='*60}")

def test_pass(msg):
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"  ✅ PASS: {msg}")

def test_fail(msg, err=None):
    global FAIL_COUNT
    FAIL_COUNT += 1
    print(f"  ❌ FAIL: {msg}")
    if err:
        traceback.print_exc()

# ═══════════════════════════════════════════════════════════════
# CFG — Minimal config (same structure as notebook)
# ═══════════════════════════════════════════════════════════════
CFG = {
    "SEED": 42,
    "IMG_SIZE": 224,
    "CLIP_LENGTH": 16,
    "NUM_CLIPS": 8,
    "BATCH_SIZE": 2,
    "EPOCHS": 2,
    "PATIENCE": 7,
    "LR_VIDEOMAE": 1e-5,
    "LR_HEAD": 1e-4,
    "WEIGHT_DECAY": 0.01,
    "GRAD_CLIP": 1.0,
    "DROPOUT": 0.3,
    "CV_FOLDS": 2,  # 2 folds for speed
    "VIDEOMAE_MODEL": "MCG-NJU/videomae-base",
    "VIDEOMAE_DIM": 768,
    "PROJECTION_DIM": 256,
    "HIDDEN_DIM": 256,
    "NUM_HEADS": 8,
    "NUM_LAYERS": 4,
    "POSE_FEAT_DIM": 16,
    "TRAINABLE_BLOCKS": [10, 11],
    "DLC_DIR": os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "DeepLabCutOutputs", "outputs"),
    "RESULTS_DIR": os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "results_v32_test"),
}

os.makedirs(CFG["RESULTS_DIR"], exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# TEST 1: VideoMAE Model Loading + Frozen Encoder
# ═══════════════════════════════════════════════════════════════
test_header("1. VideoMAE Model Loading + Frozen Encoder")

try:
    from transformers import VideoMAEModel

    class VideoMAEFrozenEncoder(nn.Module):
        def __init__(self, videomae_model, split_at: int = 10):
            super().__init__()
            self.model = videomae_model
            self.split_at = split_at
            for p in self.parameters():
                p.requires_grad = False

        @torch.no_grad()
        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            outputs = self.model(pixel_values, output_hidden_states=True)
            intermediate = outputs.hidden_states[self.split_at]
            return intermediate.mean(dim=1)

    print("  Loading VideoMAE model (this may take a minute)...")
    _full_videomae = VideoMAEModel.from_pretrained(CFG["VIDEOMAE_MODEL"])
    test_pass("VideoMAE model loaded")

    split_at = min(CFG["TRAINABLE_BLOCKS"])
    frozen_encoder = VideoMAEFrozenEncoder(_full_videomae, split_at=split_at)
    test_pass(f"Frozen encoder created (split_at={split_at})")

    # Verify all params frozen
    n_trainable = sum(p.numel() for p in frozen_encoder.parameters() if p.requires_grad)
    assert n_trainable == 0, f"Frozen encoder has {n_trainable} trainable params!"
    test_pass(f"All params frozen ({n_trainable} trainable)")

    # Forward pass test with dummy clip: (B, T, C, H, W)
    dummy_clip = torch.randn(1, 16, 3, 224, 224)
    with torch.no_grad():
        out = frozen_encoder(dummy_clip)
    assert out.shape == (1, 768), f"Expected (1, 768), got {out.shape}"
    test_pass(f"Forward pass: input (1,16,3,224,224) → output {out.shape}")

    del frozen_encoder
    test_pass("Frozen encoder cleanup OK")

except Exception as e:
    test_fail("VideoMAE Frozen Encoder", e)


# ═══════════════════════════════════════════════════════════════
# TEST 2: Domain Adapter (FFN-based)
# ═══════════════════════════════════════════════════════════════
test_header("2. Domain Adapter (FFN-based)")

try:
    class VideoMAEDomainAdapter(nn.Module):
        def __init__(self, input_dim: int = 768, hidden_dim: int = 3072,
                     projection_dim: int = 256, dropout: float = 0.1):
            super().__init__()
            self.ffn1 = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, input_dim),
                nn.Dropout(dropout),
            )
            self.ffn2 = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, input_dim),
                nn.Dropout(dropout),
            )
            self.projection = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, projection_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.ffn1(x)
            x = x + self.ffn2(x)
            return self.projection(x)

    domain_adapter = VideoMAEDomainAdapter(
        input_dim=CFG["VIDEOMAE_DIM"],
        projection_dim=CFG["PROJECTION_DIM"],
    )

    n_params = sum(p.numel() for p in domain_adapter.parameters())
    n_trainable = sum(p.numel() for p in domain_adapter.parameters() if p.requires_grad)
    test_pass(f"Created domain adapter: {n_params:,} params, {n_trainable:,} trainable")

    # Forward pass: (B, 768) → (B, 256)
    dummy_in = torch.randn(4, 768)
    out = domain_adapter(dummy_in)
    assert out.shape == (4, 256), f"Expected (4, 256), got {out.shape}"
    test_pass(f"Forward pass: (4, 768) → {out.shape}")

    # Batch forward: (B*N, 768) simulating clip batch
    dummy_batch = torch.randn(16, 768)
    out_batch = domain_adapter(dummy_batch)
    assert out_batch.shape == (16, 256), f"Expected (16, 256), got {out_batch.shape}"
    test_pass(f"Batch forward: (16, 768) → {out_batch.shape}")

    # projection[1] access (used by CowLamenessModelV32)
    vis_dim = domain_adapter.projection[1].out_features
    assert vis_dim == 256, f"Expected 256, got {vis_dim}"
    test_pass(f"projection[1].out_features = {vis_dim}")

except Exception as e:
    test_fail("Domain Adapter", e)


# ═══════════════════════════════════════════════════════════════
# TEST 3: CowLamenessModelV32 (Full Model)
# ═══════════════════════════════════════════════════════════════
test_header("3. CowLamenessModelV32 (Full Model)")

try:
    class CowLamenessModelV32(nn.Module):
        def __init__(self, adapter, pose_dim, hidden_dim, num_heads,
                     num_layers, dropout, max_clips=32):
            super().__init__()
            self.adapter = adapter
            visual_dim = adapter.projection[1].out_features
            self.input_dim = visual_dim + pose_dim
            self.hidden_dim = hidden_dim

            self.input_proj = nn.Sequential(
                nn.Linear(self.input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.register_buffer('pos_encoding',
                                 self._create_pos_encoding(hidden_dim, max_clips))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads,
                dim_feedforward=hidden_dim * 4, dropout=dropout,
                activation='gelu', batch_first=True, norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers)
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )

        def _create_pos_encoding(self, d_model, max_len):
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2).float() *
                            (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            return pe.unsqueeze(0)

        def _get_causal_mask(self, seq_len, device):
            return torch.triu(torch.ones(seq_len, seq_len, device=device),
                              diagonal=1).bool()

        def forward(self, clip_intermediate, clip_pose,
                    padding_mask=None, use_causal=True):
            B, N, D = clip_intermediate.shape
            flat = clip_intermediate.reshape(B * N, D)
            clip_visual = self.adapter(flat)
            clip_visual = clip_visual.reshape(B, N, -1)
            x = torch.cat([clip_visual, clip_pose], dim=-1)
            x = self.input_proj(x)
            x = x + self.pos_encoding[:, :N, :]
            causal_mask = self._get_causal_mask(N, x.device) if use_causal else None
            x = self.transformer(x, mask=causal_mask,
                                src_key_padding_mask=padding_mask)
            if padding_mask is not None:
                valid_mask = ~padding_mask
                x_masked = x * valid_mask.unsqueeze(-1).float()
                pooled = x_masked.sum(dim=1) / valid_mask.sum(
                    dim=1, keepdim=True).float().clamp(min=1)
            else:
                pooled = x.mean(dim=1)
            with torch.no_grad():
                attn_weights = torch.norm(x, dim=-1)
                if padding_mask is not None:
                    attn_weights = attn_weights.masked_fill(padding_mask, 0.0)
                attn_weights = F.softmax(attn_weights, dim=-1)
            logits = self.classifier(pooled)
            return logits, attn_weights

    model = CowLamenessModelV32(
        adapter=domain_adapter,
        pose_dim=CFG["POSE_FEAT_DIM"],
        hidden_dim=CFG["HIDDEN_DIM"],
        num_heads=CFG["NUM_HEADS"],
        num_layers=CFG["NUM_LAYERS"],
        dropout=CFG["DROPOUT"],
        max_clips=CFG["NUM_CLIPS"] * 2,
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    test_pass(f"Model created: {total:,} params, {trainable:,} trainable")

    # Shape test WITHOUT padding mask
    dummy_int = torch.randn(2, 8, 768)
    dummy_pose = torch.randn(2, 8, 16)
    with torch.no_grad():
        out, attn = model(dummy_int, dummy_pose)
    assert out.shape == (2, 1), f"Expected (2,1) got {out.shape}"
    assert attn.shape == (2, 8), f"Expected (2,8) got {attn.shape}"
    test_pass(f"Forward (no mask): output={out.shape}, attn={attn.shape}")

    # Shape test WITH padding mask
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[0, 5:] = True  # pad last 3 clips of sample 0
    mask[1, 6:] = True  # pad last 2 clips of sample 1
    with torch.no_grad():
        out2, attn2 = model(dummy_int, dummy_pose, padding_mask=mask)
    assert out2.shape == (2, 1)
    test_pass(f"Forward (with mask): output={out2.shape}")

    # Gradient flow test
    model.train()
    dummy_labels = torch.tensor([0.0, 1.0])
    logits, _ = model(dummy_int, dummy_pose, padding_mask=mask)
    loss = nn.BCEWithLogitsLoss()(logits.squeeze(-1), dummy_labels)
    loss.backward()
    test_pass(f"Gradient flow OK (loss={loss.item():.4f})")

    # Verify adapter gradients flow
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.adapter.parameters())
    assert has_grad, "No gradients in adapter!"
    test_pass("Adapter gradient flow confirmed")

    del model
    test_pass("Model cleanup OK")

except Exception as e:
    test_fail("CowLamenessModelV32", e)


# ═══════════════════════════════════════════════════════════════
# TEST 4: Pose Feature Extraction (real DLC CSV)
# ═══════════════════════════════════════════════════════════════
test_header("4. Pose Feature Extraction (real DLC CSV)")

try:
    # KEYPOINT_GROUPS — updated with real DLC SuperAnimal names
    KEYPOINT_GROUPS = {
        "front_left_hoof": ["front_left_paw"],
        "front_right_hoof": ["front_right_paw"],
        "hind_left_hoof": ["back_left_paw"],
        "hind_right_hoof": ["back_right_paw"],
        "withers": ["neck_base"],
        "tailhead": ["tail_base"],
        "spine": ["back_middle"],
        "left_hip": ["back_left_thai"],
        "right_hip": ["back_right_thai"],
        "front_left_knee": ["front_left_knee"],
        "front_right_knee": ["front_right_knee"],
        "hind_left_knee": ["back_left_knee"],
        "hind_right_knee": ["back_right_knee"],
        "nose": ["nose"],  # Corrected from 'nose_top' based on debug output
    }

    def get_kp(df, group_name, coord, conf_thresh=0.3):
        if group_name not in KEYPOINT_GROUPS:
            return np.full(len(df), np.nan)
        for kp_name in KEYPOINT_GROUPS[group_name]:
            col = f"{kp_name}_{coord}"
            conf_col = f"{kp_name}_likelihood"
            if col in df.columns and conf_col in df.columns:
                vals = df[col].values.astype(float)
                conf = df[conf_col].values.astype(float)
                vals[conf < conf_thresh] = np.nan
                return vals
        return np.full(len(df), np.nan)

    def extract_pose_features_from_csv(csv_path):
        try:
            df = pd.read_csv(csv_path, header=[0, 1, 2])
            new_cols = []
            for c in df.columns:
                if isinstance(c, tuple):
                    # Use only bodypart (1) and coord (2), ignore scorer (0)
                    if len(c) >= 3:
                        part = str(c[1])
                        coord = str(c[2])
                        new_cols.append(f"{part}_{coord}")
                    else:
                         new_cols.append("_".join([str(x) for x in c]))
                else:
                    new_cols.append(str(c))
            df.columns = new_cols

            features = {}
            # 1-4: Hoof heights
            for side in ["front_left", "front_right", "hind_left", "hind_right"]:
                y_vals = get_kp(df, f"{side}_hoof", "y")
                features[f"{side}_hoof_height_std"] = np.nanstd(y_vals) if np.any(~np.isnan(y_vals)) else 0.0

            # 5: Head bob
            nose_y = get_kp(df, "nose", "y")
            features["head_bob_magnitude"] = np.nanstd(nose_y) if np.any(~np.isnan(nose_y)) else 0.0

            # 6: Spine angle variation
            withers_x = get_kp(df, "withers", "x")
            withers_y = get_kp(df, "withers", "y")
            spine_x = get_kp(df, "spine", "x")
            spine_y = get_kp(df, "spine", "y")
            tailhead_x = get_kp(df, "tailhead", "x")
            tailhead_y = get_kp(df, "tailhead", "y")

            v1 = np.stack([spine_x - withers_x, spine_y - withers_y], axis=1)
            v2 = np.stack([tailhead_x - spine_x, tailhead_y - spine_y], axis=1)
            cos_vals = np.sum(v1 * v2, axis=1) / (
                np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-8)
            angles = np.arccos(np.clip(cos_vals, -1, 1))
            features["spine_angle_std"] = np.nanstd(angles) if np.any(~np.isnan(angles)) else 0.0

            # 7-8: Stride length asymmetry
            fl_x = get_kp(df, "front_left_hoof", "x")
            fr_x = get_kp(df, "front_right_hoof", "x")
            hl_x = get_kp(df, "hind_left_hoof", "x")
            hr_x = get_kp(df, "hind_right_hoof", "x")
            features["front_stride_asymmetry"] = abs(float(np.nanstd(fl_x) - np.nanstd(fr_x))) if np.any(~np.isnan(fl_x)) else 0.0
            features["hind_stride_asymmetry"] = abs(float(np.nanstd(hl_x) - np.nanstd(hr_x))) if np.any(~np.isnan(hl_x)) else 0.0

            # 9: Step frequency
            fl_y = get_kp(df, "front_left_hoof", "y")
            if np.any(~np.isnan(fl_y)):
                clean = fl_y[~np.isnan(fl_y)]
                mean_val = np.mean(clean)
                crossings = np.sum(np.diff(np.sign(clean - mean_val)) != 0)
                features["step_frequency"] = crossings / (len(clean) / 30.0 + 1e-8)
            else:
                features["step_frequency"] = 0.0

            # 10: Weight shifting
            fl_y = get_kp(df, "front_left_hoof", "y")
            fr_y = get_kp(df, "front_right_hoof", "y")
            if np.any(~np.isnan(fl_y)) and np.any(~np.isnan(fr_y)):
                diff = fl_y - fr_y
                features["weight_shift_std"] = np.nanstd(diff)
            else:
                features["weight_shift_std"] = 0.0

            # 11: Hip asymmetry
            lh_y = get_kp(df, "left_hip", "y")
            rh_y = get_kp(df, "right_hip", "y")
            if np.any(~np.isnan(lh_y)) and np.any(~np.isnan(rh_y)):
                features["hip_height_asymmetry"] = np.nanmean(np.abs(lh_y - rh_y))
            else:
                features["hip_height_asymmetry"] = 0.0

            # 12: Knee angle asymmetry (hind legs — anatomically correct)
            for side in ["left", "right"]:
                hip_x = get_kp(df, f"{side}_hip", "x")
                hip_y = get_kp(df, f"{side}_hip", "y")
                knee_x = get_kp(df, f"hind_{side}_knee", "x")
                knee_y = get_kp(df, f"hind_{side}_knee", "y")
                hoof_x = get_kp(df, f"hind_{side}_hoof", "x")
                hoof_y = get_kp(df, f"hind_{side}_hoof", "y")

                v1 = np.stack([hip_x - knee_x, hip_y - knee_y], axis=1)
                v2 = np.stack([hoof_x - knee_x, hoof_y - knee_y], axis=1)
                cos_v = np.sum(v1 * v2, axis=1) / (
                    np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-8)
                angles = np.arccos(np.clip(cos_v, -1, 1))
                features[f"hind_{side}_knee_angle_mean"] = np.nanmean(angles) if np.any(~np.isnan(angles)) else 0.0

            features["knee_angle_asymmetry"] = abs(
                features["hind_left_knee_angle_mean"] - features["hind_right_knee_angle_mean"])

            # 13-14: Stance/swing duration proxy
            fl_y = get_kp(df, "front_left_hoof", "y")
            if np.any(~np.isnan(fl_y)):
                clean = fl_y[~np.isnan(fl_y)]
                median_y = np.median(clean)
                stance = np.sum(clean > median_y)
                swing = np.sum(clean <= median_y)
                total = stance + swing
                features["stance_ratio"] = stance / total if total > 0 else 0.5
                features["swing_ratio"] = swing / total if total > 0 else 0.5
            else:
                features["stance_ratio"] = 0.5
                features["swing_ratio"] = 0.5

            # Return ordered vector (16 features)
            feature_order = [
                "front_left_hoof_height_std", "front_right_hoof_height_std",
                "hind_left_hoof_height_std", "hind_right_hoof_height_std",
                "head_bob_magnitude", "spine_angle_std",
                "front_stride_asymmetry", "hind_stride_asymmetry",
                "step_frequency", "weight_shift_std",
                "hip_height_asymmetry", "knee_angle_asymmetry",
                "hind_left_knee_angle_mean", "hind_right_knee_angle_mean",
                "stance_ratio", "swing_ratio",
            ]
            return np.array([features.get(f, 0.0) for f in feature_order], dtype=np.float32)

        except Exception:
            return np.zeros(16, dtype=np.float32)

    # Find a real DLC CSV to test
    dlc_dir = CFG["DLC_DIR"]
    csv_files = []
    for root, dirs, files in os.walk(dlc_dir):
        for f in files:
            if f.endswith('.csv'):
                csv_files.append(os.path.join(root, f))
    
    if len(csv_files) == 0:
        test_fail(f"No DLC CSV files found in {dlc_dir}")
    else:
        test_pass(f"Found {len(csv_files)} DLC CSV files")

        # Test with first CSV
        csv_path = csv_files[0]
        features = extract_pose_features_from_csv(csv_path)
        assert features.shape == (16,), f"Expected (16,), got {features.shape}"
        n_nonzero = np.count_nonzero(features)
        test_pass(f"Extracted {n_nonzero}/16 features from {os.path.basename(csv_path)}")

        if n_nonzero < 10:
            print(f"  ⚠️  WARNING: Only {n_nonzero}/16 features non-zero!")

        # Test across multiple CSVs
        n_test = min(10, len(csv_files))
        all_feats = []
        for csv_file in csv_files[:n_test]:
            f = extract_pose_features_from_csv(csv_file)
            all_feats.append(f)
        all_feats = np.array(all_feats)
        
        # Check feature distribution
        feature_names = [
            "fl_hoof_std", "fr_hoof_std", "hl_hoof_std", "hr_hoof_std",
            "head_bob", "spine_angle", "front_stride_asym", "hind_stride_asym",
            "step_freq", "weight_shift", "hip_asym", "knee_asym",
            "hl_knee_mean", "hr_knee_mean", "stance_ratio", "swing_ratio",
        ]
        zero_features = []
        for i, name in enumerate(feature_names):
            col = all_feats[:, i]
            if np.all(col == 0):
                zero_features.append(name)
        
        if zero_features:
            test_fail(f"Permanently zero features: {zero_features}")
        else:
            test_pass(f"All 16 features active across {n_test} CSVs")
        
        print(f"\n  Feature summary ({n_test} CSVs):")
        for i, name in enumerate(feature_names):
            col = all_feats[:, i]
            print(f"    {name:20s}: mean={col.mean():.4f}  std={col.std():.4f}  "
                  f"nonzero={np.count_nonzero(col)}/{n_test}")

except Exception as e:
    test_fail("Pose Feature Extraction", e)


# ═══════════════════════════════════════════════════════════════
# TEST 5: Clip Extraction (from video file)
# ═══════════════════════════════════════════════════════════════
test_header("5. Clip Extraction + Encoding Pipeline")

try:
    def extract_clips_from_video(video_path, cfg):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (cfg["IMG_SIZE"], cfg["IMG_SIZE"]))
                frames.append(frame)
            cap.release()

            if len(frames) < cfg["CLIP_LENGTH"]:
                return None

            clips = []
            stride = max(1, (len(frames) - cfg["CLIP_LENGTH"]) // cfg["NUM_CLIPS"])
            for start in range(0, len(frames) - cfg["CLIP_LENGTH"] + 1, stride):
                clip = np.array(frames[start:start + cfg["CLIP_LENGTH"]])
                clips.append(clip)
                if len(clips) >= cfg["NUM_CLIPS"] * 2:
                    break
            return clips if clips else None
        except Exception:
            return None

    @torch.no_grad()
    def encode_clips_intermediate(encoder, clips, device):
        encoder.eval()
        embeddings = []
        for clip in clips:
            clip_f = clip.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            clip_f = (clip_f - mean) / std
            # (T, H, W, C) → (1, T, C, H, W) — HuggingFace VideoMAE format
            tensor = torch.from_numpy(clip_f).permute(0, 3, 1, 2).unsqueeze(0).float()
            tensor = tensor.to(device)
            emb = encoder(tensor)
            embeddings.append(emb.cpu().numpy().squeeze())
        return np.array(embeddings)

    # Find a test video
    video_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Videolar")
    test_video = None
    for root, dirs, files in os.walk(video_dir):
        for f in files:
            if f.endswith(('.mp4', '.avi', '.mov')):
                test_video = os.path.join(root, f)
                break
        if test_video:
            break

    if test_video is None:
        print("  ⚠️  No video files found — testing with synthetic clips")
        # Synthetic test
        fake_clips = [np.random.randint(0, 255, (16, 224, 224, 3), dtype=np.uint8) for _ in range(3)]
        test_pass("Synthetic clips created")
    else:
        clips = extract_clips_from_video(test_video, CFG)
        if clips is not None:
            test_pass(f"Extracted {len(clips)} clips from {os.path.basename(test_video)}")
            assert clips[0].shape == (16, 224, 224, 3), f"Unexpected shape: {clips[0].shape}"
            test_pass(f"Clip shape: {clips[0].shape}")
            fake_clips = clips[:2]  # Use real clips but limit
        else:
            test_fail(f"Could not extract clips from {test_video}")
            fake_clips = [np.random.randint(0, 255, (16, 224, 224, 3), dtype=np.uint8)]

    # Test tensor conversion
    clip_f = fake_clips[0].astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    clip_f = (clip_f - mean) / std
    tensor = torch.from_numpy(clip_f).permute(0, 3, 1, 2).unsqueeze(0).float()
    assert tensor.shape == (1, 16, 3, 224, 224), f"Wrong shape: {tensor.shape}"
    test_pass(f"Tensor conversion: {tensor.shape} (B, T, C, H, W)")

    # Test frozen encoder with real/synthetic clip
    frozen_encoder = VideoMAEFrozenEncoder(_full_videomae, split_at=split_at)
    print("  Encoding 1 clip through frozen encoder (CPU, may be slow)...")
    with torch.no_grad():
        emb = frozen_encoder(tensor)
    assert emb.shape == (1, 768), f"Expected (1, 768), got {emb.shape}"
    test_pass(f"Frozen encoder produced: {emb.shape}")

    del frozen_encoder
    test_pass("Encoder cleanup OK")

except Exception as e:
    test_fail("Clip Extraction + Encoding", e)


# ═══════════════════════════════════════════════════════════════
# TEST 6: Dataset + DataLoader
# ═══════════════════════════════════════════════════════════════
test_header("6. Dataset + DataLoader")

try:
    class CowLamenessDatasetV32(Dataset):
        def __init__(self, intermediate_features_list, pose_features,
                     labels, cfg):
            self.intermediate_features = intermediate_features_list
            self.pose_features = pose_features
            self.labels = labels
            self.cfg = cfg

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            vis = self.intermediate_features[idx]
            pose = np.nan_to_num(self.pose_features[idx], nan=0.0)
            label = self.labels[idx]
            n_clips = len(vis)
            target_n = self.cfg["NUM_CLIPS"]
            mask = np.zeros(target_n, dtype=bool)

            if n_clips > 0:
                pose_rep = np.tile(pose, (n_clips, 1))
            else:
                vis = np.zeros((0, self.cfg["VIDEOMAE_DIM"]), dtype=np.float32)
                pose_rep = np.zeros((0, self.cfg["POSE_FEAT_DIM"]), dtype=np.float32)

            if n_clips >= target_n:
                indices = np.linspace(0, n_clips - 1, target_n, dtype=int)
                vis = vis[indices]
                pose_rep = pose_rep[indices]
            elif n_clips > 0:
                pad_v = np.zeros((target_n - n_clips, self.cfg["VIDEOMAE_DIM"]),
                                 dtype=np.float32)
                pad_p = np.zeros((target_n - n_clips, self.cfg["POSE_FEAT_DIM"]),
                                 dtype=np.float32)
                vis = np.concatenate([vis, pad_v], axis=0)
                pose_rep = np.concatenate([pose_rep, pad_p], axis=0)
                mask[n_clips:] = True
            else:
                vis = np.zeros((target_n, self.cfg["VIDEOMAE_DIM"]), dtype=np.float32)
                pose_rep = np.zeros((target_n, self.cfg["POSE_FEAT_DIM"]),
                                    dtype=np.float32)
                mask[:] = True

            return vis, pose_rep, label, mask

    def collate_fn(batch):
        visuals, poses, labels, masks = zip(*batch)
        return (
            torch.tensor(np.array(visuals), dtype=torch.float32),
            torch.tensor(np.array(poses), dtype=torch.float32),
            torch.tensor(np.array(labels), dtype=torch.long),
            torch.tensor(np.array(masks), dtype=torch.bool),
        )

    # Create synthetic data
    n_samples = 20
    fake_intermediate = [np.random.randn(np.random.randint(3, 12), 768).astype(np.float32)
                         for _ in range(n_samples)]
    fake_pose = np.random.randn(n_samples, 16).astype(np.float32)
    fake_labels = np.array([0] * 10 + [1] * 10)

    ds = CowLamenessDatasetV32(fake_intermediate, fake_pose, fake_labels, CFG)
    assert len(ds) == n_samples
    test_pass(f"Dataset created: {len(ds)} samples")

    vis, pose, label, mask = ds[0]
    assert vis.shape == (8, 768), f"Expected (8, 768), got {vis.shape}"
    assert pose.shape == (8, 16), f"Expected (8, 16), got {pose.shape}"
    test_pass(f"__getitem__: vis={vis.shape}, pose={pose.shape}, mask={mask.shape}")

    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=0)
    batch = next(iter(dl))
    v_b, p_b, l_b, m_b = batch
    assert v_b.shape == (4, 8, 768)
    assert p_b.shape == (4, 8, 16)
    assert l_b.shape == (4,)
    assert m_b.shape == (4, 8)
    test_pass(f"DataLoader batch: vis={v_b.shape}, pose={p_b.shape}")

except Exception as e:
    test_fail("Dataset + DataLoader", e)


# ═══════════════════════════════════════════════════════════════
# TEST 7: Training Loop (1 epoch, synthetic data)
# ═══════════════════════════════════════════════════════════════
test_header("7. Training Loop (1 epoch)")

try:
    import copy

    # Fresh adapter + model
    test_adapter = VideoMAEDomainAdapter(
        input_dim=CFG["VIDEOMAE_DIM"],
        projection_dim=CFG["PROJECTION_DIM"],
    )
    test_model = CowLamenessModelV32(
        adapter=test_adapter,
        pose_dim=CFG["POSE_FEAT_DIM"],
        hidden_dim=CFG["HIDDEN_DIM"],
        num_heads=CFG["NUM_HEADS"],
        num_layers=CFG["NUM_LAYERS"],
        dropout=CFG["DROPOUT"],
    )

    # 2 LR groups
    adapter_params = list(test_model.adapter.parameters())
    temporal_params = [p for n, p in test_model.named_parameters()
                      if not n.startswith("adapter.")]
    optimizer = torch.optim.AdamW([
        {"params": adapter_params, "lr": CFG["LR_VIDEOMAE"]},
        {"params": temporal_params, "lr": CFG["LR_HEAD"]},
    ], weight_decay=CFG["WEIGHT_DECAY"])
    test_pass("Optimizer created (2 LR groups)")

    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)
    test_pass("Criterion + scheduler created")

    # Train 1 epoch
    test_model.train()
    total_loss = 0
    n_batches = 0
    for v_b, p_b, l_b, m_b in dl:
        optimizer.zero_grad()
        logits, _ = test_model(v_b, p_b, padding_mask=m_b)
        loss = criterion(logits.squeeze(-1), l_b.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(test_model.parameters(), CFG["GRAD_CLIP"])
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    test_pass(f"Training epoch: {n_batches} batches, avg_loss={avg_loss:.4f}")

    scheduler.step(avg_loss)
    test_pass("Scheduler step OK")

    # Eval mode
    test_model.eval()
    with torch.no_grad():
        all_probs = []
        all_labels_list = []
        for v_b, p_b, l_b, m_b in dl:
            logits, attn = test_model(v_b, p_b, padding_mask=m_b, use_causal=True)
            probs = torch.sigmoid(logits.squeeze(-1))
            all_probs.extend(probs.numpy())
            all_labels_list.extend(l_b.numpy())

    probs_arr = np.array(all_probs)
    labels_arr = np.array(all_labels_list)
    preds = (probs_arr >= 0.5).astype(int)
    acc = accuracy_score(labels_arr, preds)
    f1 = f1_score(labels_arr, preds, zero_division=0)
    test_pass(f"Evaluation: acc={acc:.3f}, f1={f1:.3f}")

    # Deepcopy test (for fold model creation)
    fold_adapter = copy.deepcopy(test_adapter)
    fold_model = CowLamenessModelV32(
        adapter=fold_adapter,
        pose_dim=CFG["POSE_FEAT_DIM"],
        hidden_dim=CFG["HIDDEN_DIM"],
        num_heads=CFG["NUM_HEADS"],
        num_layers=CFG["NUM_LAYERS"],
        dropout=CFG["DROPOUT"],
    )
    with torch.no_grad():
        out, _ = fold_model(v_b, p_b, padding_mask=m_b)
    assert out.shape == (v_b.shape[0], 1)
    test_pass("Fold model deepcopy + forward OK")

    del test_model, fold_model, fold_adapter
    test_pass("Training cleanup OK")

except Exception as e:
    test_fail("Training Loop", e)


# ═══════════════════════════════════════════════════════════════
# TEST 8: StratifiedGroupKFold (CV splitter)
# ═══════════════════════════════════════════════════════════════
test_header("8. Cross-Validation Split")

try:
    n_samples = 40
    labels = np.array([0] * 20 + [1] * 20)
    animal_ids = np.array([f"cow_{i//4}" for i in range(n_samples)])
    paths = np.array([f"video_{i}.mp4" for i in range(n_samples)])

    cv = StratifiedGroupKFold(n_splits=CFG["CV_FOLDS"], shuffle=True,
                               random_state=CFG["SEED"])

    for fold, (train_idx, val_idx) in enumerate(cv.split(paths, labels, animal_ids)):
        train_animals = set(animal_ids[train_idx])
        val_animals = set(animal_ids[val_idx])
        leakage = train_animals & val_animals
        assert len(leakage) == 0, f"Fold {fold}: animal leakage! {leakage}"
        test_pass(f"Fold {fold+1}: train={len(train_idx)} ({len(train_animals)} animals), "
                  f"val={len(val_idx)} ({len(val_animals)} animals), no leakage")

except Exception as e:
    test_fail("Cross-Validation Split", e)


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
print(f"  ✅ PASSED: {PASS_COUNT}")
print(f"  ❌ FAILED: {FAIL_COUNT}")
print(f"{'='*60}")

if FAIL_COUNT == 0:
    print("\n  🎉 ALL TESTS PASSED — Safe to build Colab notebook!")
else:
    print(f"\n  ⚠️  {FAIL_COUNT} test(s) failed — fix before deploying to Colab!")

# Cleanup
del _full_videomae
print("\n  🗑️ Cleanup complete")

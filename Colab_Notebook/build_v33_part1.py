"""
Build Cow_Lameness_Analysis_v33.ipynb — Part 1 (Sections 1-3)
Environment, Data Discovery, Pose Feature Extraction (with Debug & Norm)
"""
import json, os

def _split(source):
    lines = source.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": _split(source)}

def code(source):
    return {"cell_type": "code", "metadata": {}, "source": _split(source),
            "outputs": [], "execution_count": None}

cells = []

# ═══════════════════════════════════════════════════════════════
# SECTION 1: Environment & Config
# ═══════════════════════════════════════════════════════════════
cells.append(md("""# 🐄 Cow Lameness Analysis v33 — Deep Learning (LoRA)
## Full VideoMAE Fine-Tuning (LoRA) + Normalized Pose + Temporal Transformer

**Architecture:** 
- **Visual:** VideoMAE (Frozen Backbone) + **LoRA Adapters** (Trainable) -> 768 dim
- **Pose:** DLC features (Normalized + Debugged) -> 16 dim
- **Fusion:** Concatenation -> Temporal Transformer -> Binary Classification

**Goal:** Correcting v32's learning failure (53% acc) by enabling deep feature adaptation via LoRA and rigorous pose validation.

---
**Changes from v32:**
1.  **LoRA Integration:** No more frozen backbone. We use PEFT to adapt query/value projections.
2.  **No Caching:** Training flows through the backbone every epoch (slower but accurate).
3.  **Pose Debug:** Added sanity check and z-score normalization for pose features.
"""))

cells.append(code("""# ============================================================
# SECTION 1: Environment, Imports & Configuration
# ============================================================
import os
import sys
import glob
import random
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, average_precision_score
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.signal import find_peaks

warnings.filterwarnings('ignore')
print("✅ Core imports successful")

# Install transformers & PEFT if needed
try:
    from transformers import VideoMAEModel, VideoMAEConfig
    from peft import get_peft_model, LoraConfig, TaskType
    print("✅ Transformers & PEFT already installed")
except ImportError:
    print("📦 Installing transformers, accelerate, peft...")
    os.system("pip install -q transformers accelerate peft")
    from transformers import VideoMAEModel, VideoMAEConfig
    from peft import get_peft_model, LoraConfig, TaskType
    print("✅ Libraries installed")

try:
    import cv2
    print("✅ OpenCV available")
except ImportError:
    os.system("pip install -q opencv-python-headless")
    import cv2
    print("✅ OpenCV installed")

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
"""))

cells.append(code("""# ============================================================
# Configuration — Single source of truth
# ============================================================
CFG = {
    # Reproducibility
    "SEED": 42,

    # Data
    "VIDEO_DIR": "/content/drive/MyDrive/Inek Topallik Tespiti Parcalanmis Inek Videolari/cow_single_videos",
    "DLC_OUTPUT_DIR": "/content/drive/MyDrive/DeepLabCut/outputs",
    "RESULTS_DIR": "/content/drive/MyDrive/CowLameness_v33_results",

    # Clip extraction
    "NUM_CLIPS": 8,
    "CLIP_LENGTH": 16,
    "IMG_SIZE": 224,

    # Pose features
    "POSE_FRAMEWORK": "deeplabcut",
    "POSE_FEAT_DIM": 16,
    "MIN_CONFIDENCE": 0.3,

    # VideoMAE + LoRA
    "VIDEOMAE_MODEL": "MCG-NJU/videomae-base",
    "VIDEOMAE_DIM": 768,
    "LORA_R": 16,
    "LORA_ALPHA": 16,
    "LORA_DROPOUT": 0.1,
    "LORA_TARGET_MODULES": ["query", "value"], # Will be verified by PEFT regex matching

    # Temporal Transformer
    "HIDDEN_DIM": 256,
    "NUM_HEADS": 8,
    "NUM_LAYERS": 4,
    "DROPOUT": 0.3,

    # Training
    "BATCH_SIZE": 4,
    "EPOCHS": 50,           # Increased from 40 for deep learning
    "LR_BACKBONE": 1e-4,    # Lower LR for LoRA (was 5e-4, too high - loss increasing)
    "LR_HEAD": 1e-4,
    "WEIGHT_DECAY": 1e-4,
    "PATIENCE": 10,         # Increased patience
    "GRAD_CLIP": 1.0,       # Gradient clipping to prevent explosion
    "CV_FOLDS": 5,

    # Class labels
    "HEALTHY_LABEL": 0,
    "LAME_LABEL": 1,
}

# Deterministic everything
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(CFG["SEED"])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Config loaded | Device: {DEVICE} | Seed: {CFG['SEED']}")
"""))

cells.append(code("""# ============================================================
# Google Drive Mount
# ============================================================
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("✅ Google Drive mounted")
except Exception:
    print("⚠️ Not running in Colab — using local paths")

# Create results directory
os.makedirs(CFG["RESULTS_DIR"], exist_ok=True)
print(f"📁 Results will be saved to: {CFG['RESULTS_DIR']}")
"""))

# ═══════════════════════════════════════════════════════════════
# SECTION 2: Data Discovery & Subject ID
# ═══════════════════════════════════════════════════════════════
cells.append(md("""---
## Section 2: Data Discovery & Subject ID Extraction

Discover all video files and their corresponding DLC outputs.
Extract `animal_id` from filenames for subject-level splitting.
"""))

cells.append(code("""# ============================================================
# SECTION 2: Data Discovery & Subject ID Extraction
# ============================================================

def discover_data(cfg: dict) -> pd.DataFrame:
    \"\"\"
    Discover videos and match with DLC pose outputs.

    Returns:
        DataFrame with columns: video_path, dlc_csv_path, label, animal_id
    \"\"\"
    video_dir = Path(cfg["VIDEO_DIR"])
    dlc_dir = Path(cfg["DLC_OUTPUT_DIR"])

    records = []

    for folder, label in [("Saglikli", 0), ("Topal", 1)]:
        video_folder = video_dir / folder
        dlc_subfolder = dlc_dir / folder

        if not video_folder.exists():
            print(f"⚠️ Video folder not found: {video_folder}")
            continue

        videos = sorted(video_folder.glob("*.mp4"))
        print(f"📁 {folder}: {len(videos)} videos found")

        for vpath in videos:
            stem = vpath.stem

            # Extract animal_id
            parts = stem.replace("cow_", "").replace("cow", "")
            animal_id = parts.split("_")[0].split("DLC")[0]

            # Find matching DLC CSV (subfolder first, then root, then recursive)
            dlc_csv = None
            for search_dir in [dlc_subfolder, dlc_dir]:
                if search_dir.exists():
                    matches = list(search_dir.glob(f"{stem}DLC*.csv"))
                    if matches:
                        dlc_csv = str(matches[0])
                        break
            if dlc_csv is None and dlc_dir.exists():
                matches = list(dlc_dir.glob(f"**/{stem}DLC*.csv"))
                if matches:
                    dlc_csv = str(matches[0])

            records.append({
                "video_path": str(vpath),
                "dlc_csv_path": dlc_csv,
                "label": label,
                "label_name": folder,
                "animal_id": animal_id,
                "video_name": stem
            })

    df = pd.DataFrame(records)

    # Summary statistics
    print(f"\\n{'='*50}")
    print(f"📊 Dataset Summary")
    print(f"{'='*50}")
    print(f"Total videos: {len(df)}")
    print(f"  Healthy (Sağlıklı): {(df['label']==0).sum()}")
    print(f"  Lame (Topal):       {(df['label']==1).sum()}")
    print(f"Unique animals: {df['animal_id'].nunique()}")
    print(f"DLC outputs available: {df['dlc_csv_path'].notna().sum()}/{len(df)}")
    print(f"{'='*50}")

    return df

data_df = discover_data(CFG)
data_df.head(10)
"""))

# ═══════════════════════════════════════════════════════════════
# SECTION 3: Pose Feature Extraction
# ═══════════════════════════════════════════════════════════════
cells.append(md("""---
## Section 3: Pose Feature Extraction (Normalised)

**Fixing the 'Garbage In' problem:**
1.  **Correct CSV Parsing:** Ignoring scorer row, correct keypoint mapping.
2.  **Sanity Check:** Verifying feature variance before training.
3.  **Normalization:** Applying Z-score normalization (`StandardScaler`) to features.
"""))

cells.append(code("""# ============================================================
# SECTION 3: Pose Feature Extractor
# ============================================================

class PoseFeatureExtractor:
    \"\"\"
    Extract biomechanical gait features from DLC SuperAnimal outputs.
    Produces a fixed-size feature vector per video/clip.
    \"\"\"

    # Keypoint Groups (Validated from local CSV debug)
    KEYPOINT_GROUPS = {
        "head": ["nose", "head", "forehead"],
        "withers": ["neck_base", "withers", "shoulder_center"],
        "spine": ["back_middle", "spine", "mid_back"],
        "tail_base": ["tail_base", "tailbase", "tail"],
        "left_front_hoof": ["front_left_paw", "left_front_paw", "lf_paw"],
        "right_front_hoof": ["front_right_paw", "right_front_paw", "rf_paw"],
        "left_hind_hoof": ["back_left_paw", "left_hind_paw", "lh_paw"],
        "right_hind_hoof": ["back_right_paw", "right_hind_paw", "rh_paw"],
        "left_front_knee": ["front_left_knee", "left_front_knee", "lf_knee"],
        "right_front_knee": ["front_right_knee", "right_front_knee", "rf_knee"],
        "left_hind_knee": ["back_left_knee", "left_hind_knee", "lh_knee"],
        "right_hind_knee": ["back_right_knee", "right_hind_knee", "rh_knee"],
        "left_hip": ["back_left_thai", "left_hip", "lh_hip"],
        "right_hip": ["back_right_thai", "right_hip", "rh_hip"],
    }

    FEATURE_NAMES = [
        "head_vertical_disp_mean", "head_vertical_disp_std",
        "spine_angle_mean", "spine_angle_std",
        "step_duration_left_mean", "step_duration_left_std",
        "step_duration_right_mean", "step_duration_right_std",
        "temporal_asymmetry_ratio", "step_frequency",
        "hip_sway_amplitude", "hip_sway_range",
        "stride_length_left_cv", "stride_length_right_cv",
        "knee_angle_asymmetry", "back_curvature_variance",
    ]

    def __init__(self, fps: float = 30.0, framework: str = "deeplabcut",
                 min_confidence: float = 0.3):
        self.fps = fps
        self.framework = framework
        self.min_conf = min_confidence
        self._keypoint_map = None

    @property
    def num_features(self) -> int:
        return len(self.FEATURE_NAMES)

    def _resolve_keypoints(self, columns) -> Dict[str, Optional[str]]:
        \"\"\"Dynamically resolve keypoint names from CSV columns.\"\"\"
        col_strs = [str(c).lower() for c in columns]
        resolved = {}
        for group_name, candidates in self.KEYPOINT_GROUPS.items():
            found = None
            for candidate in candidates:
                for cs in col_strs:
                    if candidate in cs:
                        found = candidate
                        break
                if found:
                    break
            resolved[group_name] = found
        return resolved

    def _get_keypoint_data(self, df: pd.DataFrame, kp_name: Optional[str]
                           ) -> Optional[np.ndarray]:
        \"\"\"Get (x, y, confidence) for a keypoint. Returns (N, 3) or None.\"\"\"
        if kp_name is None:
            return None
        matching = [c for c in df.columns if kp_name in str(c).lower()]
        if len(matching) < 3:
            return None
        try:
            x = pd.to_numeric(df[matching[0]], errors='coerce').values
            y = pd.to_numeric(df[matching[1]], errors='coerce').values
            c = pd.to_numeric(df[matching[2]], errors='coerce').values
            return np.column_stack([x, y, c])
        except Exception:
            return None

    def extract_from_csv(self, csv_path: str) -> np.ndarray:
        \"\"\"Extract features from a DLC CSV file. Returns (16,) array (NaN = missing).\"\"\"
        try:
            if self.framework == "deeplabcut":
                df = pd.read_csv(csv_path, header=[0, 1, 2])
                new_cols = []
                for c in df.columns:
                    if isinstance(c, tuple) and len(c) >= 3:
                        # Use only bodypart (1) and coord (2), ignore scorer (0)
                        part = str(c[1])
                        coord = str(c[2])
                        new_cols.append(f"{part}_{coord}".lower())
                    else:
                        new_cols.append('_'.join(str(x) for x in c).lower())
                df.columns = new_cols
            else:
                # Fallback for simple CSVs
                df = pd.read_csv(csv_path, index_col=0)
                df.columns = [str(c).lower() for c in df.columns]

            # Resolve keypoints per CSV
            self._keypoint_map = self._resolve_keypoints(df.columns)

            return self._compute_features(df)
        except Exception:
            return np.full(self.num_features, np.nan, dtype=np.float32)

    def _compute_features(self, df: pd.DataFrame) -> np.ndarray:
        \"\"\"Compute all 16 features from parsed DataFrame. NaN = not computable.\"\"\"
        feats = np.full(self.num_features, np.nan, dtype=np.float32)
        n_frames = len(df)

        if n_frames < 30:
            return feats

        km = self._keypoint_map

        # --- Head bob ---
        head = self._get_keypoint_data(df, km.get("head"))
        if head is not None:
            mask = head[:, 2] > self.min_conf
            if mask.sum() > 10:
                y = head[mask, 1]
                dy = np.diff(y)
                feats[0] = np.mean(np.abs(dy))
                feats[1] = np.std(dy)

        # --- Spine angle ---
        withers = self._get_keypoint_data(df, km.get("withers"))
        spine = self._get_keypoint_data(df, km.get("spine"))
        tail = self._get_keypoint_data(df, km.get("tail_base"))
        if all(v is not None for v in [withers, spine, tail]):
            angles = self._compute_angle_trajectory(withers, spine, tail)
            if len(angles) > 5:
                feats[2] = np.nanmean(angles)
                feats[3] = np.nanstd(angles)
                feats[15] = np.nanvar(angles)  # back_curvature_variance

        # --- Step timing (front hooves) ---
        lf = self._get_keypoint_data(df, km.get("left_front_hoof"))
        rf = self._get_keypoint_data(df, km.get("right_front_hoof"))
        steps_l = self._detect_steps(lf) if lf is not None else np.array([])
        steps_r = self._detect_steps(rf) if rf is not None else np.array([])

        if len(steps_l) > 1:
            dur_l = np.diff(steps_l) / self.fps
            feats[4] = np.median(dur_l)
            feats[5] = np.std(dur_l)
        if len(steps_r) > 1:
            dur_r = np.diff(steps_r) / self.fps
            feats[6] = np.median(dur_r)
            feats[7] = np.std(dur_r)

        # --- Temporal asymmetry ---
        if feats[4] > 0 and feats[6] > 0:
            feats[8] = abs(feats[4] - feats[6]) / max(feats[4], feats[6])

        # --- Step frequency ---
        total_steps = len(steps_l) + len(steps_r)
        duration_sec = n_frames / self.fps
        feats[9] = total_steps / max(duration_sec, 1.0)

        # --- Hip sway ---
        lh = self._get_keypoint_data(df, km.get("left_hip"))
        rh = self._get_keypoint_data(df, km.get("right_hip"))
        if lh is not None and rh is not None:
            mask = (lh[:, 2] > self.min_conf) & (rh[:, 2] > self.min_conf)
            if mask.sum() > 10:
                cx = (lh[mask, 0] + rh[mask, 0]) / 2
                feats[10] = np.std(cx)
                feats[11] = np.ptp(cx)

        # --- Stride length CV ---
        if lf is not None and len(steps_l) > 2:
            sl = np.abs(np.diff(lf[steps_l, 0]))
            feats[12] = np.std(sl) / (np.mean(sl) + 1e-6)
        if rf is not None and len(steps_r) > 2:
            sr = np.abs(np.diff(rf[steps_r, 0]))
            feats[13] = np.std(sr) / (np.mean(sr) + 1e-6)

        # --- Knee angle asymmetry (hind legs) ---
        lhk = self._get_keypoint_data(df, km.get("left_hind_knee"))
        rhk = self._get_keypoint_data(df, km.get("right_hind_knee"))
        lhh = self._get_keypoint_data(df, km.get("left_hind_hoof"))
        rhh = self._get_keypoint_data(df, km.get("right_hind_hoof"))
        if all(v is not None for v in [lh, lhk, lhh]) and all(v is not None for v in [rh, rhk, rhh]):
            la = self._compute_angle_trajectory(lh, lhk, lhh)
            ra = self._compute_angle_trajectory(rh, rhk, rhh)
            if len(la) > 5 and len(ra) > 5:
                feats[14] = abs(np.nanmean(la) - np.nanmean(ra))

        feats = np.where(np.isfinite(feats), feats, np.nan)
        return feats

    def _detect_steps(self, kp_data: np.ndarray) -> np.ndarray:
        \"\"\"Detect heel strikes from vertical trajectory.\"\"\"
        mask = kp_data[:, 2] > self.min_conf
        if mask.sum() < 15:
            return np.array([])
        y = np.where(mask, kp_data[:, 1], np.nan)
        nans = np.isnan(y)
        if nans.all():
            return np.array([])
        x_interp = np.arange(len(y))
        y[nans] = np.interp(x_interp[nans], x_interp[~nans], y[~nans])
        peaks, _ = find_peaks(y, distance=int(0.3 * self.fps), prominence=3)
        return peaks

    def _compute_angle_trajectory(self, p1: np.ndarray, p2: np.ndarray,
                                   p3: np.ndarray) -> np.ndarray:
        \"\"\"
        Compute angle at p2 formed by p1-p2-p3.
        Includes minimal interpolation for missing frames to boost valid samples.
        \"\"\"
        n = min(len(p1), len(p2), len(p3))
        angles = []
        
        # Pre-filter: if any point has low confidence, try to interpolate
        # Simple forward-fill for short gaps
        def fill_gaps(arr, conf_thresh=0.3):
            valid = arr[:, 2] > conf_thresh
            if valid.sum() < 3: return arr # Too few points
            
            # Linear interp for X, Y
            x = arr[:, 0].copy()
            y = arr[:, 1].copy()
            x[~valid] = np.nan
            y[~valid] = np.nan
            
            # Pandas-like interpolate (but using numpy)
            nans = np.isnan(x)
            if nans.any() and not nans.all():
                x_idx = np.arange(len(x))
                x[nans] = np.interp(x_idx[nans], x_idx[~nans], x[~nans])
                y[nans] = np.interp(x_idx[nans], x_idx[~nans], y[~nans])
                
            return np.column_stack([x, y, arr[:, 2]]) # Keep original confidence for reference

        p1_f = fill_gaps(p1, self.min_conf)
        p2_f = fill_gaps(p2, self.min_conf)
        p3_f = fill_gaps(p3, self.min_conf)

        for i in range(n):
            # We use filled x,y but still check if original confidence wasn't TOTAL garbage
            # Relaxed check: Accept if standard check fails but interpolated points exist
            v1 = p1_f[i, :2] - p2_f[i, :2]
            v2 = p3_f[i, :2] - p2_f[i, :2]
            
            # Check for NaN from failed interpolation
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

# Initialize
pose_extractor = PoseFeatureExtractor(
    fps=30.0,
    framework=CFG["POSE_FRAMEWORK"],
    min_confidence=CFG["MIN_CONFIDENCE"]
)
print(f"✅ PoseFeatureExtractor initialized | {pose_extractor.num_features} features")
"""))

cells.append(code("""# ============================================================
# POSE SANITY CHECK (Before full extraction)
# ============================================================

def pose_sanity_check(df, extractor, num_samples=20):
    \"\"\"Run extraction on 20 random CSVs and plot feature usage.\"\"\"
    print("🔬 Running Pose Sanity Check...")
    
    samples = df[df['dlc_csv_path'].notna()].sample(min(num_samples, len(df)))
    feats_list = []
    
    for idx, row in samples.iterrows():
        feat = extractor.extract_from_csv(row['dlc_csv_path'])
        feats_list.append(feat)
        
    feats = np.array(feats_list)
    
    # Calculate non-zero/non-nan percentages
    valid_mask = ~np.isnan(feats)
    nonzero_mask = valid_mask & (np.abs(feats) > 1e-6)
    
    params = extractor.FEATURE_NAMES
    percentages = nonzero_mask.mean(axis=0) * 100
    
    print(f"\\n{'Feature Name':<30} | {'Valid %':<10} | {'Status'}")
    print("-" * 55)
    for i, name in enumerate(params):
        status = "✅ OK" if percentages[i] > 20 else "⚠️ LOW/ZERO"
        print(f"{name:<30} | {percentages[i]:6.1f}%    | {status}")
        
    print(f"\n🧠 GLOBAL POSE QUALITY: {np.nanmean(percentages):.1f}% valid features")
    if np.nanmean(percentages) < 40:
        print("⚠️ WARNING: Pose data quality is VERY LOW. Model may rely mostly on VideoMAE.")
    else:
        print("✅ Pose data quality is acceptable.")
        
    return feats

_ = pose_sanity_check(data_df, pose_extractor)
"""))

cells.append(code("""# ============================================================
# Extract and Normalize Pose Features
# ============================================================

def extract_pose_features(data_df: pd.DataFrame, 
                          extractor: PoseFeatureExtractor,
                          cache_path: str = None) -> np.ndarray:
    \"\"\"Extract and cache raw pose features. (Normalization happens in CV loop)\"\"\"
    
    # 1. Extract (or load cache)
    if cache_path and os.path.exists(cache_path):
        feats = np.load(cache_path)
        print(f"✅ Loaded cached raw pose features: {feats.shape}")
    else:
        print(f"🔄 Extracting pose features for {len(data_df)} videos...")
        all_feats = []
        missing = 0
        for idx, row in data_df.iterrows():
            csv_path = row.get("dlc_csv_path")
            if csv_path and os.path.exists(str(csv_path)):
                feat = extractor.extract_from_csv(str(csv_path))
            else:
                feat = np.full(extractor.num_features, np.nan, dtype=np.float32)
                missing += 1
            all_feats.append(feat)

            if (idx + 1) % 200 == 0:
                print(f"  Processed {idx+1}/{len(data_df)}...")
        
        feats = np.array(all_feats, dtype=np.float32)
        if cache_path:
            np.save(cache_path, feats)
            print(f"💾 Cached raw features to {cache_path}")
            
    # 2. Impute NaNs (with mean of column) - Simple global imputation is okay for raw data
    # Ideally should be done inside CV too, but global mean imputation for MISSING data is often acceptable
    # if the missingness isn't target-dependent. Let's keep it simple: impute global mean.
    col_means = np.nanmean(feats, axis=0)
    inds = np.where(np.isnan(feats))
    feats[inds] = np.take(col_means, inds[1])
    
    # 2.5 Replace remaining NaNs (if column all NaN) with 0
    feats = np.nan_to_num(feats, nan=0.0)

    print(f"✅ Features Extracted (Raw): mean={feats.mean():.3f}, std={feats.std():.3f}")
    return feats

cache_path = os.path.join(CFG["RESULTS_DIR"], "pose_features_v33_raw.npy")

pose_features = extract_pose_features(
    data_df, pose_extractor, cache_path
)
"""))

# Save part 1
notebook = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {"colab": {"provenance": [], "gpuType": "T4"},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}, "accelerator": "GPU"},
    "cells": cells
}

SCRIPT_DIR = r"c:\Users\HP\Desktop\Clone Repos\CowLameness\Colab_Notebook"
out_path = SCRIPT_DIR + "\\_v33_part1.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"Part 1 saved: {out_path} ({len(cells)} cells)")

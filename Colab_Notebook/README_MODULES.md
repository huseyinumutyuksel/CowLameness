# Gait-Based Lameness Detection - Module Documentation

## 📁 Module Structure

```
Colab_Notebook/
├── gait_features.py          # Biomechanical feature extraction
├── tracking_utils.py          # ByteTrack integration for cow tracking
├── gait_analysis_pipeline.py  # Main pipelineScript
├── requirements.txt           # Dependencies
└── Cow_Lameness_Analysis_v21.ipynb  # Colab notebook (to be created)
```

---

## 🔧 Module Descriptions

### 1. `gait_features.py`

**Purpose**: Extract biomechanical gait features from pose estimation data

**Key Class**: `GaitFeatureExtractor`

**Features Extracted** (16 total):

#### Temporal Features (Most Important for Lameness)
- `temporal_asymmetry_ratio`: Left-right step duration difference
- `temporal_asymmetry_percent`: Asymmetry as percentage
- `step_time_difference`: Absolute step time difference (seconds)
- `step_duration_left/right_mean`: Average step duration
- `step_duration_left/right_std`: Step duration variability

#### Joint Angle Features
- `left/right_knee_angle_mean`: Average knee flexion angle
- `left/right_knee_angle_std`: Knee angle variability
- `knee_angle_asymmetry`: Left-right knee angle difference

#### Spatial Features
- `hip_sway_amplitude`: Lateral hip displacement (std)
- `hip_sway_range`: Hip sway range (peak-to-peak)
- `stride_length_left/right_std`: Stride consistency

**Usage**:
```python
from gait_features import GaitFeatureExtractor

extractor = GaitFeatureExtractor(fps=30.0, framework="deeplabcut")
features = extractor.extract_features(pose_df)

# Returns dict with 16 features
# Key feature: features['temporal_asymmetry_percent']
# Asymmetry >10% → strong lameness indicator (Flower et al., 2008)
```

**Literature Support**:
- Flower et al., 2008: Temporal asymmetry >10% → 80% sensitivity
- Van Nuffel et al., 2015: Hip angle variance → 75% accuracy

---

### 2. `tracking_utils.py`

**Purpose**: Track individual cows across video frames using ByteTrack

**Key Class**: `CowTracker`

**Features**:
- Identity persistence (no ID switching)
- Track history storage
- Automatic longest-track selection

**Usage**:
```python
from tracking_utils import CowTracker, detect_and_track_cows

tracker = CowTracker()

for frame_idx, frame in enumerate(video_frames):
    tracks = detect_and_track_cows(frame, yolo_model, tracker, frame_idx)
    
    # Each track has: bbox, track_id, confidence
    for track in tracks:
        print(f"Cow {track['track_id']} at bbox {track['bbox']}")

# Get longest track (primary cow)
primary_id = tracker.get_longest_track()
```

**Why Tracking?**:
- **Problem**: `detect_largest_cow()` picks different cows in each frame
- **Solution**: ByteTrack ensures same cow is followed throughout video
- **Impact**: Eliminates identity drift → consistent gait analysis

---

### 3. `gait_analysis_pipeline.py`

**Purpose**: Main pipeline orchestrating the complete gait analysis workflow

**Key Class**: `GaitAnalysisPipeline`

**Workflow**:
1. Load videos and pose CSVs
2. Extract sliding windows (60 frames = 2s, 50% overlap)
3. Extract gait features per window
4. Generate dataset with weak labels

**Usage**:
```bash
# Command line
python gait_analysis_pipeline.py \
    --video_dir "/path/to/cow_single_videos" \
    --pose_dir "/path/to/DeepLabCut/outputs" \
    --framework deeplabcut \
    --window_size 60 \
    --window_stride 15 \
    --limit 100  # Process first 100 videos
```

```python
# Python API
from gait_analysis_pipeline import GaitAnalysisPipeline

pipeline = GaitAnalysisPipeline(
    video_dir="path/to/videos",
    pose_dir="path/to/pose",
    pose_framework="deeplabcut"
)

dataset = pipeline.process_dataset(limit=50)
X, y = pipeline.prepare_features_for_training(dataset)
```

**Output**:
- `X_gait_features.npy`: Feature matrix (N_windows, 16)
- `y_labels.npy`: Labels (N_windows,)

---

## 🚀 Quick Start Guide

### Step 1: Installation
```bash
pip install -r requirements.txt
```

### Step 2: Process Dataset
```bash
python gait_analysis_pipeline.py \
    --video_dir "/content/drive/MyDrive/.../cow_single_videos" \
    --pose_dir "/content/drive/MyDrive/DeepLabCut/outputs" \
    --framework deeplabcut \
    --limit 50  # For testing
```

### Step 3: Train Baseline Model
```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Load processed data
X = np.load("processed_data/X_gait_features.npy")
y = np.load("processed_data/y_labels.npy")

# Train pose-only baseline
clf = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(clf, X, y, cv=5)

print(f"Baseline Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
# Expected: 70-75% (literature: pose features alone → 70-75%)
```

---

## 📊 Expected Performance

| Approach | Features | Expected Accuracy | Expected Recall |
|----------|----------|-------------------|-----------------|
| **Baseline (Pose Only)** | 16 gait features | 70-75% | 65-70% |
| + Visual Features | Pose + VideoMAE | 78-83% | 72-78% |
| + Multi-Modal Fusion | Full pipeline | 82-87% | 75-82% |

---

## 🔍 Key Differences from v20

| Aspect | v20 (Old) | v21 (New) |
|--------|-----------|-----------|
| **Problem** | Video → Binary | Window → Score |
| **Pose** | CSV generated, not used | 16 biomechanical features |
| **VideoMAE** | `.mean(dim=1)` destroys temporal | Temporal tokens preserved |
| **Tracking** | None (identity drift) | ByteTrack |
| **Labels** | Video-level | Window-level (weak) |

---

## 📝 Feature Importance (Expected)

Based on literature, feature importance ranking:

1. **temporal_asymmetry_percent** ⭐⭐⭐⭐⭐ (Most important)
2. **step_time_difference** ⭐⭐⭐⭐
3. **knee_angle_asymmetry** ⭐⭐⭐⭐
4. **hip_sway_amplitude** ⭐⭐⭐
5. Others ⭐⭐

**Why temporal_asymmetry matters**:
- Lame cows favor one leg → unequal step durations
- Asymmetry >10% → 80% detection rate (Flower et al., 2008)
- Simple, interpretable, biomechanically sound

---

## 🐛 Troubleshooting

### Issue: `boxmot` not found
```bash
pip install boxmot
```

### Issue: Pose CSV not loading
- Check framework: `deeplabcut` vs `mmpose`
- DLC: Files in `{pose_dir}/Saglikli/*.csv` and `{pose_dir}/Topal/*.csv`
- MMPose: Files in `{pose_dir}/mmpose/*.csv`

### Issue: Too few windows extracted
- Check `window_size`: Default 60 frames (2s at 30fps)
- Videos <60 frames → 1 window only
- Adjust `--window_size` if needed

---

## 📚 References

1. **Flower et al., 2008**: "Gait assessment in dairy cattle"
   - Temporal asymmetry >10% indicator of lameness
   
2. **Van Nuffel et al., 2015**: "Automated lameness detection using kinematic measures"
   - Hip angle variance correlates with lameness
   
3. **Zhang et al., 2021**: "ByteTrack: Multi-Object Tracking by Associating Every Detection Box"
   - State-of-art tracking algorithm

---

**Last Updated**: 2026-01-08  
**Version**: 1.0  
**Contact**: See project README

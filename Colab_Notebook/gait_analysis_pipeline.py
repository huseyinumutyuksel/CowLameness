"""
Gait-Based Lameness Detection Pipeline

This script implements the redesigned architecture for cow lameness detection
using temporal gait analysis instead of static classification.

Key Components:
1. ByteTrack for cow tracking (identity persistence)
2. Gait feature extraction from pose data
3. Sliding window approach (temporal segments)
4. Multi-modal fusion with proper temporal handling

Usage:
    python gait_analysis_pipeline.py --video_dir /path/to/videos --pose_dir /path/to/pose
"""

import os
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

# Import custom modules
from gait_features import GaitFeatureExtractor, extract_gait_features_from_csv
from tracking_utils import CowTracker, detect_and_track_cows, select_primary_cow


class GaitAnalysisPipeline:
    """
    Main pipeline for gait-based lameness detection
    """
    
    def __init__(self, video_dir: str, pose_dir: str, 
                 pose_framework: str = "deeplabcut",
                 window_size: int = 60, window_stride: int = 15):
        """
        Args:
            video_dir: Directory containing cow videos
            pose_dir: Directory containing pose CSV files
            pose_framework: "deeplabcut" or "mmpose"
            window_size: Temporal window size in frames (default: 60 = 2s at 30fps)
            window_stride: Window stride in frames (default: 15 = 50% overlap)
        """
        self.video_dir = video_dir
        self.pose_dir = pose_dir
        self.pose_framework = pose_framework
        self.window_size = window_size
        self.window_stride = window_stride
        
        # Initialize components
        self.gait_extractor = GaitFeatureExtractor(fps=30.0, framework=pose_framework)
        
        print(f"✅ Pipeline initialized")
        print(f"   Video dir: {video_dir}")
        print(f"   Pose dir: {pose_dir}")
        print(f"   Window size: {window_size} frames")
    
    def load_pose_csv(self, video_name: str, label: int) -> pd.DataFrame:
        """
        Load pose CSV file for a given video
        
        Args:
            video_name: Video filename (without extension)
            label: 0 for healthy, 1 for lame
        
        Returns:
            Pose DataFrame or None if not found
        """
        if self.pose_framework == "deeplabcut":
            # DeepLabCut pattern: {video}DLC*.csv in Saglikli/Topal folders
            sub_folder = "Saglikli" if label == 0 else "Topal"
            pose_pattern = f"{self.pose_dir}/{sub_folder}/{video_name}DLC*.csv"
            pose_files = glob.glob(pose_pattern)
            
            if pose_files:
                return pd.read_csv(pose_files[0], header=[1, 2])
            
        elif self.pose_framework == "mmpose":
            # MMPose pattern: {video}_MMPose.csv
            pose_file = f"{self.pose_dir}/{video_name}_MMPose.csv"
            if os.path.exists(pose_file):
                return pd.read_csv(pose_file, index_col=0)
        
        return None
    
    def extract_sliding_windows(self, pose_df: pd.DataFrame, 
                                video_label: int) -> List[Tuple[Dict, int]]:
        """
        Extract overlapping temporal windows from pose sequence
        
        Args:
            pose_df: Full pose DataFrame for video
            video_label: Video-level weak label (0 or 1)
        
        Returns:
            List of (gait_features, weak_label) tuples
        """
        windows = []
        num_frames = len(pose_df)
        
        if num_frames < self.window_size:
            # Video too short, extract features from entire sequence
            features = self.gait_extractor.extract_features(pose_df)
            windows.append((features, video_label))
            return windows
        
        # Sliding window extraction
        for start_idx in range(0, num_frames - self.window_size + 1, self.window_stride):
            end_idx = start_idx + self.window_size
            
            # Extract window
            window_pose = pose_df.iloc[start_idx:end_idx]
            
            # Extract gait features
            features = self.gait_extractor.extract_features(window_pose)
            
            # Weak label: if video is lame, any window might be lame
            windows.append((features, video_label))
        
        return windows
    
    def process_dataset(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Process all videos in dataset
        
        Args:
            limit: Optional limit on number of videos to process (for testing)
        
        Returns:
            List of dataset samples with features and labels
        """
        dataset = []
        
        # Get all videos
        video_files = []
        for label_folder in ['Saglikli', 'Topal']:
            folder_path = f"{self.video_dir}/{label_folder}"
            videos = glob.glob(f"{folder_path}/*.mp4")
            label = 0 if label_folder == 'Saglikli' else 1
            video_files.extend([(v, label) for v in videos])
        
        print(f"\n📊 Found {len(video_files)} videos")
        print(f"   Healthy: {sum(1 for _, l in video_files if l==0)}")
        print(f"   Lame: {sum(1 for _, l in video_files if l==1)}")
        
        # Limit for testing
        if limit:
            video_files = video_files[:limit]
            print(f"   Processing first {limit} videos (testing mode)")
        
        # Process videos
        for video_path, label in tqdm(video_files, desc="Processing videos"):
            try:
                video_name = Path(video_path).stem
                
                # Load pose CSV
                pose_df = self.load_pose_csv(video_name, label)
                
                if pose_df is None or len(pose_df) < 30:
                    # Skip if no pose data or too short
                    continue
                
                # Extract windows
                windows = self.extract_sliding_windows(pose_df, label)
                
                # Add to dataset
                for window_features, window_label in windows:
                    dataset.append({
                        'video': video_name,
                        'features': window_features,
                        'label': window_label
                    })
                
            except Exception as e:
                print(f"  ⚠️  Failed: {Path(video_path).name} - {e}")
        
        print(f"\n✅ Processed {len(dataset)} temporal windows")
        print(f"   from {len(set(d['video'] for d in dataset))} videos")
        
        return dataset
    
    def prepare_features_for_training(self, dataset: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert dataset to numpy arrays for training
        
        Returns:
            (X, y) where X is feature matrix, y is labels
        """
        # Convert feature dicts to arrays
        feature_names = list(dataset[0]['features'].keys())
        
        X = np.array([
            [d['features'][fname] for fname in feature_names]
            for d in dataset
        ])
        
        y = np.array([d['label'] for d in dataset])
        
        print(f"\n📊 Dataset prepared:")
        print(f"   Shape: {X.shape}")
        print(f"   Features: {len(feature_names)}")
        print(f"   Samples: {len(y)}")
        print(f"   Healthy: {sum(y == 0)} ({sum(y == 0)/len(y)*100:.1f}%)")
        print(f"   Lame: {sum(y == 1)} ({sum(y == 1)/len(y)*100:.1f}%)")
        
        return X, y


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Gait-based lameness detection')
    parser.add_argument('--video_dir', type=str, required=True,
                       help='Directory containing cow videos (with Saglikli/Topal subdirs)')
    parser.add_argument('--pose_dir', type=str, required=True,
                       help='Directory containing pose CSV files')
    parser.add_argument('--framework', type=str, default='deeplabcut',
                       choices=['deeplabcut', 'mmpose'],
                       help='Pose estimation framework')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of videos (for testing)')
    parser.add_argument('--window_size', type=int, default=60,
                       help='Temporal window size in frames')
    parser.add_argument('--window_stride', type=int, default=15,
                       help='Window stride in frames')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = GaitAnalysisPipeline(
        video_dir=args.video_dir,
        pose_dir=args.pose_dir,
        pose_framework=args.framework,
        window_size=args.window_size,
        window_stride=args.window_stride
    )
    
    # Process dataset
    dataset = pipeline.process_dataset(limit=args.limit)
    
    # Prepare for training
    X, y = pipeline.prepare_features_for_training(dataset)
    
    # Save processed data
    output_dir = Path(args.video_dir).parent / "processed_data"
    output_dir.mkdir(exist_ok=True)
    
    np.save(output_dir / "X_gait_features.npy", X)
    np.save(output_dir / "y_labels.npy", y)
    
    print(f"\n✅ Data saved to {output_dir}")
    print(f"   X_gait_features.npy: {X.shape}")
    print(f"   y_labels.npy: {y.shape}")


if __name__ == "__main__":
    main()

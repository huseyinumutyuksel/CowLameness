"""
Gait Feature Extraction Module for Cow Lameness Detection

This module extracts biomechanical features from pose estimation data (DeepLabCut/MMPose)
to detect gait asymmetry and lameness patterns.

Key Features:
- Step duration analysis (left/right)
- Temporal asymmetry detection
- Joint angle calculations
- Gait cycle detection

References:
- Flower et al., 2008: Temporal asymmetry >10% → 80% lameness detection
- Van Nuffel et al., 2015: Hip angle variance → 75% accuracy
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from typing import Dict, List, Tuple, Optional


class GaitFeatureExtractor:
    """
    Extract biomechanical gait features from pose keypoint trajectories
    """
    
    def __init__(self, fps: float = 30.0, framework: str = "deeplabcut"):
        """
        Args:
            fps: Frame rate of video
            framework: "deeplabcut" or "mmpose"
        """
        self.fps = fps
        self.framework = framework
        
        # Keypoint indices (DeepLabCut specific - adjust for your config)
        self.keypoints = {
            'left_hoof': 0,
            'right_hoof': 1,
            'left_knee': 2,
            'right_knee': 3,
            'left_hip': 4,
            'right_hip': 5,
        }
    
    def extract_features(self, pose_df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract all gait features from pose CSV data
        
        Args:
            pose_df: DataFrame with pose keypoints (DLC format with multi-level header)
        
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Parse pose data based on framework
        coords = self._parse_pose_data(pose_df)
        
        if coords is None or len(coords) < 30:  # Need at least 1 second of data
            return self._get_default_features()
        
        # 1. Heel strike detection
        steps_left = self._detect_heel_strikes(coords, 'left_hoof')
        steps_right = self._detect_heel_strikes(coords, 'right_hoof')
        
        # 2. Step duration features
        features.update(self._compute_step_duration(steps_left, steps_right))
        
        # 3. Temporal asymmetry (MOST IMPORTANT)
        features.update(self._compute_temporal_asymmetry(steps_left, steps_right))
        
        # 4. Joint angle features
        features.update(self._compute_joint_angles(coords))
        
        # 5. Hip sway amplitude
        features.update(self._compute_hip_sway(coords))
        
        # 6. Stride length variation
        features.update(self._compute_stride_variation(coords, steps_left, steps_right))
        
        return features
    
    def _parse_pose_data(self, pose_df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Parse pose DataFrame into numpy array
        
        Returns:
            Array of shape (num_frames, num_keypoints, 3) where 3 = (x, y, confidence)
        """
        try:
            if self.framework == "deeplabcut":
                # DLC has multi-level header: (scorer, bodypart, coords)
                # Extract x, y, likelihood for each bodypart
                num_frames = len(pose_df)
                num_keypoints = len(self.keypoints)
                coords = np.zeros((num_frames, num_keypoints, 3))
                
                for kp_name, kp_idx in self.keypoints.items():
                    # Find columns for this keypoint
                    # DLC format: (scorer, bodypart, x/y/likelihood)
                    kp_cols = [col for col in pose_df.columns if kp_name in str(col)]
                    
                    if len(kp_cols) >= 3:
                        coords[:, kp_idx, 0] = pose_df[kp_cols[0]].values  # x
                        coords[:, kp_idx, 1] = pose_df[kp_cols[1]].values  # y
                        coords[:, kp_idx, 2] = pose_df[kp_cols[2]].values  # confidence
                
                return coords
                
            elif self.framework == "mmpose":
                # MMPose has simpler format
                return pose_df.values.reshape(-1, len(self.keypoints), 3)
            
        except Exception as e:
            print(f"Error parsing pose data: {e}")
            return None
    
    def _detect_heel_strikes(self, coords: np.ndarray, hoof_name: str) -> np.ndarray:
        """
        Detect heel strikes (when hoof touches ground) using vertical position minima
        
        Args:
            coords: Pose coordinates
            hoof_name: 'left_hoof' or 'right_hoof'
        
        Returns:
            Array of frame indices where heel strikes occur
        """
        hoof_idx = self.keypoints[hoof_name]
        y_trajectory = coords[:, hoof_idx, 1]  # Vertical position
        confidence = coords[:, hoof_idx, 2]
        
        # Filter low confidence points
        y_trajectory = np.where(confidence > 0.5, y_trajectory, np.nan)
        
        # Interpolate NaN values
        mask = ~np.isnan(y_trajectory)
        if mask.sum() < 10:
            return np.array([])
        
        # Find local minima (heel strikes)
        # Use inverted y (higher values = lower position in image)
        peaks, _ = find_peaks(
            y_trajectory[mask],
            distance=int(0.3 * self.fps),  # Minimum 0.3s between steps
            prominence=5  # Minimum vertical displacement
        )
        
        return peaks
    
    def _compute_step_duration(self, steps_left: np.ndarray, steps_right: np.ndarray) -> Dict[str, float]:
        """
        Compute step duration statistics
        """
        features = {}
        
        if len(steps_left) > 1:
            durations_left = np.diff(steps_left) / self.fps
            features['step_duration_left_mean'] = np.median(durations_left)
            features['step_duration_left_std'] = np.std(durations_left)
        else:
            features['step_duration_left_mean'] = 0.0
            features['step_duration_left_std'] = 0.0
        
        if len(steps_right) > 1:
            durations_right = np.diff(steps_right) / self.fps
            features['step_duration_right_mean'] = np.median(durations_right)
            features['step_duration_right_std'] = np.std(durations_right)
        else:
            features['step_duration_right_mean'] = 0.0
            features['step_duration_right_std'] = 0.0
        
        return features
    
    def _compute_temporal_asymmetry(self, steps_left: np.ndarray, steps_right: np.ndarray) -> Dict[str, float]:
        """
        Compute temporal asymmetry - THE MOST IMPORTANT FEATURE FOR LAMENESS
        
        Asymmetry > 10% is strong indicator of lameness (Flower et al., 2008)
        """
        features = {}
        
        if len(steps_left) > 1 and len(steps_right) > 1:
            duration_left = np.median(np.diff(steps_left)) / self.fps
            duration_right = np.median(np.diff(steps_right)) / self.fps
            
            # Temporal asymmetry ratio
            if duration_left > 0:
                asymmetry = abs(duration_left - duration_right) / duration_left
                features['temporal_asymmetry_ratio'] = asymmetry
                features['temporal_asymmetry_percent'] = asymmetry * 100
            else:
                features['temporal_asymmetry_ratio'] = 0.0
                features['temporal_asymmetry_percent'] = 0.0
            
            # Step time difference
            features['step_time_difference'] = abs(duration_left - duration_right)
        else:
            features['temporal_asymmetry_ratio'] = 0.0
            features['temporal_asymmetry_percent'] = 0.0
            features['step_time_difference'] = 0.0
        
        return features
    
    def _compute_joint_angles(self, coords: np.ndarray) -> Dict[str, float]:
        """
        Compute joint angle statistics (hip-knee-hoof angles)
        """
        features = {}
        
        try:
            # Left leg angle (hip -> knee -> hoof)
            left_angles = self._calculate_angle_trajectory(
                coords, 'left_hip', 'left_knee', 'left_hoof'
            )
            
            # Right leg angle
            right_angles = self._calculate_angle_trajectory(
                coords, 'right_hip', 'right_knee', 'right_hoof'
            )
            
            # Statistics
            if len(left_angles) > 0:
                features['left_knee_angle_mean'] = np.nanmean(left_angles)
                features['left_knee_angle_std'] = np.nanstd(left_angles)
            else:
                features['left_knee_angle_mean'] = 0.0
                features['left_knee_angle_std'] = 0.0
            
            if len(right_angles) > 0:
                features['right_knee_angle_mean'] = np.nanmean(right_angles)
                features['right_knee_angle_std'] = np.nanstd(right_angles)
            else:
                features['right_knee_angle_mean'] = 0.0
                features['right_knee_angle_std'] = 0.0
            
            # Angle asymmetry
            if len(left_angles) > 0 and len(right_angles) > 0:
                angle_diff = abs(np.nanmean(left_angles) - np.nanmean(right_angles))
                features['knee_angle_asymmetry'] = angle_diff
            else:
                features['knee_angle_asymmetry'] = 0.0
                
        except Exception as e:
            print(f"Error computing joint angles: {e}")
            features.update({
                'left_knee_angle_mean': 0.0,
                'left_knee_angle_std': 0.0,
                'right_knee_angle_mean': 0.0,
                'right_knee_angle_std': 0.0,
                'knee_angle_asymmetry': 0.0
            })
        
        return features
    
    def _calculate_angle_trajectory(self, coords: np.ndarray, 
                                    joint1: str, joint2: str, joint3: str) -> np.ndarray:
        """
        Calculate angle formed by three joints over time
        """
        idx1 = self.keypoints[joint1]
        idx2 = self.keypoints[joint2]
        idx3 = self.keypoints[joint3]
        
        angles = []
        for frame in range(len(coords)):
            # Get coordinates
            p1 = coords[frame, idx1, :2]
            p2 = coords[frame, idx2, :2]
            p3 = coords[frame, idx3, :2]
            conf = coords[frame, idx2, 2]  # Use middle joint confidence
            
            if conf > 0.5:
                # Calculate angle
                v1 = p1 - p2
                v2 = p3 - p2
                
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
                angles.append(angle)
            else:
                angles.append(np.nan)
        
        return np.array(angles)
    
    def _compute_hip_sway(self, coords: np.ndarray) -> Dict[str, float]:
        """
        Compute hip sway amplitude (lateral displacement)
        """
        features = {}
        
        try:
            # Average hip position (left + right) / 2
            left_hip_idx = self.keypoints['left_hip']
            right_hip_idx = self.keypoints['right_hip']
            
            left_hip_x = coords[:, left_hip_idx, 0]
            right_hip_x = coords[:, right_hip_idx, 0]
            conf_left = coords[:, left_hip_idx, 2]
            conf_right = coords[:, right_hip_idx, 2]
            
            # Filter by confidence
            mask = (conf_left > 0.5) & (conf_right > 0.5)
            if mask.sum() > 10:
                hip_center_x = (left_hip_x[mask] + right_hip_x[mask]) / 2
                
                # Sway amplitude = std of lateral position
                features['hip_sway_amplitude'] = np.std(hip_center_x)
                features['hip_sway_range'] = np.ptp(hip_center_x)  # Peak-to-peak
            else:
                features['hip_sway_amplitude'] = 0.0
                features['hip_sway_range'] = 0.0
        except:
            features['hip_sway_amplitude'] = 0.0
            features['hip_sway_range'] = 0.0
        
        return features
    
    def _compute_stride_variation(self, coords: np.ndarray, 
                                   steps_left: np.ndarray, steps_right: np.ndarray) -> Dict[str, float]:
        """
        Compute stride length variation (consistency)
        """
        features = {}
        
        try:
            left_hoof_idx = self.keypoints['left_hoof']
            right_hoof_idx = self.keypoints['right_hoof']
            
            # Calculate stride lengths (distance between consecutive heel strikes)
            if len(steps_left) > 1:
                left_hoof_x = coords[steps_left, left_hoof_idx, 0]
                stride_lengths_left = np.abs(np.diff(left_hoof_x))
                features['stride_length_left_std'] = np.std(stride_lengths_left)
            else:
                features['stride_length_left_std'] = 0.0
            
            if len(steps_right) > 1:
                right_hoof_x = coords[steps_right, right_hoof_idx, 0]
                stride_lengths_right = np.abs(np.diff(right_hoof_x))
                features['stride_length_right_std'] = np.std(stride_lengths_right)
            else:
                features['stride_length_right_std'] = 0.0
        except:
            features['stride_length_left_std'] = 0.0
            features['stride_length_right_std'] = 0.0
        
        return features
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default features when pose data is insufficient"""
        return {
            'step_duration_left_mean': 0.0,
            'step_duration_left_std': 0.0,
            'step_duration_right_mean': 0.0,
            'step_duration_right_std': 0.0,
            'temporal_asymmetry_ratio': 0.0,
            'temporal_asymmetry_percent': 0.0,
            'step_time_difference': 0.0,
            'left_knee_angle_mean': 0.0,
            'left_knee_angle_std': 0.0,
            'right_knee_angle_mean': 0.0,
            'right_knee_angle_std': 0.0,
            'knee_angle_asymmetry': 0.0,
            'hip_sway_amplitude': 0.0,
            'hip_sway_range': 0.0,
            'stride_length_left_std': 0.0,
            'stride_length_right_std': 0.0
        }


# Utility function for easy usage
def extract_gait_features_from_csv(csv_path: str, fps: float = 30.0, 
                                     framework: str = "deeplabcut") -> Dict[str, float]:
    """
    Convenience function to extract features directly from CSV file
    
    Args:
        csv_path: Path to pose CSV file
        fps: Video frame rate
        framework: "deeplabcut" or "mmpose"
    
    Returns:
        Dictionary of gait features
    """
    try:
        if framework == "deeplabcut":
            pose_df = pd.read_csv(csv_path, header=[1, 2])  # Multi-level header
        else:
            pose_df = pd.read_csv(csv_path, index_col=0)
        
        extractor = GaitFeatureExtractor(fps=fps, framework=framework)
        features = extractor.extract_features(pose_df)
        
        return features
    except Exception as e:
        print(f"Error extracting features from {csv_path}: {e}")
        return GaitFeatureExtractor(fps, framework)._get_default_features()

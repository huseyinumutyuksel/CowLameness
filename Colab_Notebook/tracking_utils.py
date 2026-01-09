"""
ByteTrack Integration Utilities for Cow Tracking

This module provides wrapper functions for integrating ByteTrack
into the cow lameness detection pipeline.

Dependencies:
    pip install boxmot
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import cv2


class CowTracker:
    """
    Wrapper for ByteTrack to track individual cows across video frames
    """
    
    def __init__(self):
        """Initialize ByteTrack tracker"""
        try:
            from boxmot import ByteTrack
            self.tracker = ByteTrack(
                track_thresh=0.5,  # Detection confidence threshold
                track_buffer=30,   # Frames to keep track alive
                match_thresh=0.8,  # IOU matching threshold
                frame_rate=30      # Video FPS
            )
            self.tracks_history = {}  # Store track history
        except ImportError:
            raise ImportError(
                "boxmot not installed. Install with: pip install boxmot"
            )
    
    def update(self, detections: np.ndarray, frame_idx: int) -> List[Dict]:
        """
        Update tracker with new detections
        
        Args:
            detections: numpy array of shape (N, 6) with format:
                       [x1, y1, x2, y2, confidence, class_id]
            frame_idx: Current frame number
        
        Returns:
            List of tracked objects with format:
            [
                {
                    'bbox': (x1, y1, x2, y2),
                    'track_id': int,
                    'confidence': float,
                    'class_id': int
                },
                ...
            ]
        """
        if len(detections) == 0:
            # Update tracker with empty detections
            tracks = self.tracker.update(np.empty((0, 6)), None)
            return []
        
        # Update tracker
        tracks = self.tracker.update(detections, None)
        
        # Parse tracks
        tracked_objects = []
        for track in tracks:
            x1, y1, x2, y2, track_id, conf, cls = track[:7]
            
            track_info = {
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'track_id': int(track_id),
                'confidence': float(conf),
                'class_id': int(cls),
                'frame_idx': frame_idx
            }
            
            # Store in history
            if track_id not in self.tracks_history:
                self.tracks_history[track_id] = []
            self.tracks_history[track_id].append(track_info)
            
            tracked_objects.append(track_info)
        
        return tracked_objects
    
    def get_track_history(self, track_id: int) -> List[Dict]:
        """Get full history of a specific track"""
        return self.tracks_history.get(track_id, [])
    
    def get_longest_track(self) -> Optional[int]:
        """
        Get the track_id of the longest track (most frames)
        This is useful for selecting the primary cow in the video
        """
        if not self.tracks_history:
            return None
        
        longest_id = max(
            self.tracks_history.keys(),
            key=lambda tid: len(self.tracks_history[tid])
        )
        return longest_id
    
    def get_track_duration(self, track_id: int) -> int:
        """Get number of frames a track was active"""
        return len(self.tracks_history.get(track_id, []))


def yolo_to_tracker_format(yolo_results) -> np.ndarray:
    """
    Convert YOLO detection results to ByteTrack input format
    
    Args:
        yolo_results: YOLO model output (ultralytics format)
    
    Returns:
        numpy array of shape (N, 6): [x1, y1, x2, y2, confidence, class_id]
    """
    if len(yolo_results[0].boxes) == 0:
        return np.empty((0, 6))
    
    boxes = yolo_results[0].boxes.xyxy.cpu().numpy()  # (N, 4)
    confidences = yolo_results[0].boxes.conf.cpu().numpy()  # (N,)
    class_ids = yolo_results[0].boxes.cls.cpu().numpy()  # (N,)
    
    # Concatenate to (N, 6)
    detections = np.column_stack([
        boxes,
        confidences,
        class_ids
    ])
    
    return detections


def detect_and_track_cows(frame, yolo_model, tracker, frame_idx: int, 
                          cow_class_id: int = 19) -> List[Dict]:
    """
    Detect cows with YOLO and track them with ByteTrack
    
    Args:
        frame: Video frame (numpy array)
        yolo_model: YOLO model instance
        tracker: CowTracker instance
        frame_idx: Current frame number
        cow_class_id: COCO class ID for cow (default: 19)
    
    Returns:
        List of tracked cows with bbox and track_id
    """
    # YOLO detection
    results = yolo_model(frame, classes=[cow_class_id], verbose=False)
    
    # Convert to tracker format
    detections = yolo_to_tracker_format(results)
    
    # Update tracker
    tracks = tracker.update(detections, frame_idx)
    
    return tracks


def select_primary_cow(tracks: List[Dict]) -> Optional[Dict]:
    """
    Select the primary cow from multiple tracks
    Currently selects the largest bounding box (foreground cow)
    
    Args:
        tracks: List of tracked cows
    
    Returns:
        Primary cow track info, or None if no cows detected
    """
    if not tracks:
        return None
    
    # Select largest bbox (most likely foreground cow)
    def bbox_area(track):
        x1, y1, x2, y2 = track['bbox']
        return (x2 - x1) * (y2 - y1)
    
    primary_cow = max(tracks, key=bbox_area)
    return primary_cow


def process_video_with_tracking(video_path: str, yolo_model, 
                                fps: float = 30.0) -> Dict[int, List[Dict]]:
    """
    Process entire video with cow tracking
    
    Args:
        video_path: Path to video file
        yolo_model: YOLO model instance
        fps: Video frame rate
    
    Returns:
        Dictionary mapping track_id to list of frame-by-frame detections
    """
    cap = cv2.VideoCapture(video_path)
    tracker = CowTracker()
    
    frame_idx = 0
    all_tracks = {}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect and track
        tracks = detect_and_track_cows(frame, yolo_model, tracker, frame_idx)
        
        # Store tracks
        for track in tracks:
            track_id = track['track_id']
            if track_id not in all_tracks:
                all_tracks[track_id] = []
            all_tracks[track_id].append({
                'frame_idx': frame_idx,
                'bbox': track['bbox'],
                'confidence': track['confidence']
            })
        
        frame_idx += 1
    
    cap.release()
    
    return all_tracks

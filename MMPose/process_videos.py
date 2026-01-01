"""
MMPose Video Processing Pipeline
Processes cow videos using RTMPose model

Usage:
    python process_videos.py --test     # Test mode (single video)
    python process_videos.py --batch    # Batch mode (all videos)
"""
import os
import sys
import glob
import argparse
import logging
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from tqdm import tqdm

# Check imports
try:
    from mmpose.apis import MMPoseInferencer
    import cv2
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please activate the virtual environment:")
    print("   .venv\\Scripts\\Activate.ps1")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
VIDEO_DIR = Path("C:/Users/HP/Desktop/Yeni klasör/CowLameness_v15/cow_single_videos")
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Model selection - RTMPose for animal pose
MODEL_CONFIG = 'rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288'

def initialize_mmpose():
    """Initialize MMPose inferencer"""
    logger.info("🔧 Initializing MMPose inferencer...")
    try:
        inferencer = MMPoseInferencer(
            pose2d=MODEL_CONFIG,
            device='cuda'  # Use GPU if available
        )
        logger.info("✅ MMPose inferencer initialized")
        return inferencer
    except Exception as e:
        logger.error(f"❌ Failed to initialize MMPose: {e}")
        logger.info("   Trying CPU mode...")
        try:
            inferencer = MMPoseInferencer(
                pose2d=MODEL_CONFIG,
                device='cpu'
            )
            logger.info("✅ MMPose inferencer initialized (CPU mode)")
            return inferencer
        except Exception as e2:
            logger.error(f"❌ Failed to initialize MMPose on CPU: {e2}")
            sys.exit(1)

def process_video_to_csv(inferencer, video_path: Path, output_path: Path):
    """Process a single video and save as CSV"""
    # Run inference
    results_generator = inferencer(
        str(video_path),
        show=False,
        return_vis=False
    )
    
    # Collect all frames
    all_keypoints = []
    frame_indices = []
    
    for frame_idx, result in enumerate(results_generator):
        predictions = result['predictions'][0] if result['predictions'] else None
        
        if predictions:
            # Extract keypoints
            keypoints = predictions[0]['keypoints']  # Shape: (N_keypoints, 2)
            scores = predictions[0]['keypoint_scores']  # Shape: (N_keypoints,)
            
            # Flatten to single row
            row = []
            for kp_idx, (kp, score) in enumerate(zip(keypoints, scores)):
                row.extend([kp[0], kp[1], score])  # x, y, confidence
            
            all_keypoints.append(row)
            frame_indices.append(frame_idx)
    
    # Convert to DataFrame (matching DLC format)
    n_keypoints = len(keypoints)
    columns = []
    for kp_idx in range(n_keypoints):
        columns.extend([f'kp{kp_idx}_x', f'kp{kp_idx}_y', f'kp{kp_idx}_conf'])
    
    df = pd.DataFrame(all_keypoints, columns=columns, index=frame_indices)
    
    # Save to CSV
    df.to_csv(output_path)
    return df

def process_single_video(inferencer, video_path: Path) -> bool:
    """Process a single video (TEST MODE)"""
    logger.info(f"🎥 Processing test video: {video_path.name}")
    
    try:
        output_csv = OUTPUT_DIR / f"{video_path.stem}_MMPose.csv"
        
        df = process_video_to_csv(inferencer, video_path, output_csv)
        
        logger.info(f"✅ Test successful!")
        logger.info(f"   Output: {output_csv}")
        logger.info(f"   Shape: {df.shape} (frames x keypoints)")
        return True
            
    except Exception as e:
        logger.error(f"❌ Error processing video: {e}")
        return False

def process_batch(inferencer, video_list: List[Path], resume: bool = True):
    """Process all videos with resume capability"""
    logger.info(f"🚀 Starting batch processing: {len(video_list)} videos")
    
    # Check already processed videos
    processed = []
    if resume:
        existing_csvs = list(OUTPUT_DIR.glob("*_MMPose.csv"))
        processed = [csv.stem.replace("_MMPose", "") for csv in existing_csvs]
        logger.info(f"📊 Found {len(processed)} already processed videos (will skip)")
    
    # Filter out processed videos
    videos_to_process = [v for v in video_list if v.stem not in processed]
    
    if not videos_to_process:
        logger.info("✅ All videos already processed!")
        return
    
    logger.info(f"📹 Videos remaining: {len(videos_to_process)}")
    logger.info(f"⏱️  Estimated time: {len(videos_to_process) * 1}-{len(videos_to_process) * 2} minutes")
    
    # Process with progress bar
    failed = []
    for video_path in tqdm(videos_to_process, desc="Processing videos"):
        try:
            output_csv = OUTPUT_DIR / f"{video_path.stem}_MMPose.csv"
            process_video_to_csv(inferencer, video_path, output_csv)
            
            # Log milestone
            processed.append(video_path.stem)
            if len(processed) % 100 == 0:
                logger.info(f"✅ Milestone: {len(processed)} videos processed")
                
        except Exception as e:
            logger.error(f"❌ Failed: {video_path.name} - {e}")
            failed.append(video_path.name)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"BATCH PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"✅ Successfully processed: {len(processed) - len(failed)}")
    if failed:
        logger.warning(f"❌ Failed: {len(failed)}")
        logger.warning(f"Failed videos: {failed[:10]}...")
    logger.info(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(
        description="MMPose Batch Video Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--test", action="store_true",
                       help="Test mode: process single video")
    parser.add_argument("--batch", action="store_true",
                       help="Batch mode: process all videos")
    parser.add_argument("--no-resume", action="store_true",
                       help="Disable resume (reprocess all)")
    
    args = parser.parse_args()
    
    if not args.test and not args.batch:
        parser.print_help()
        sys.exit(1)
    
    # Initialize MMPose
    inferencer = initialize_mmpose()
    
    if args.test:
        # TEST MODE
        print("\n" + "="*60)
        print("TEST MODE - Processing Single Video")
        print("="*60)
        
        test_video = VIDEO_DIR / "Saglikli" / "cow_0001.mp4"
        if not test_video.exists():
            logger.error(f"❌ Test video not found: {test_video}")
            sys.exit(1)
        
        success = process_single_video(inferencer, test_video)
        
        if success:
            print("\n" + "="*60)
            print("✅ TEST SUCCESSFUL!")
            print("="*60)
            print(f"Output CSV: {OUTPUT_DIR / 'cow_0001_MMPose.csv'}")
            print("\nNext steps:")
            print("  1. Review the output CSV file")
            print("  2. If satisfied, run batch processing:")
            print("     python process_videos.py --batch")
            print("="*60)
        else:
            sys.exit(1)
    
    elif args.batch:
        # BATCH MODE
        print("\n" + "="*60)
        print("⚠️  BATCH PROCESSING MODE")
        print("="*60)
        print(f"Video directory: {VIDEO_DIR}")
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"Estimated time: 20-40 hours for 1167 videos")
        print("="*60)
        
        user_confirm = input("\nProceed with batch processing? (yes/no): ").strip().lower()
        if user_confirm != "yes":
            print("❌ Cancelled by user")
            sys.exit(0)
        
        # Collect all videos
        all_videos = []
        for folder in ["Saglikli", "Topal"]:
            folder_path = VIDEO_DIR / folder
            if folder_path.exists():
                videos = list(folder_path.glob("*.mp4"))
                all_videos.extend(videos)
                logger.info(f"📁 {folder}: {len(videos)} videos")
        
        if not all_videos:
            logger.error("❌ No videos found!")
            sys.exit(1)
        
        logger.info(f"📊 Total: {len(all_videos)} videos")
        
        # Process
        process_batch(inferencer, all_videos, resume=not args.no_resume)

if __name__ == "__main__":
    main()

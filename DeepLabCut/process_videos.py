"""
DeepLabCut Video Processing Pipeline
Processes cow videos using SuperAnimal-Quadruped model

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
from tqdm import tqdm

# Check imports
try:
    import deeplabcut
    import pandas as pd
    import numpy as np
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

# Configuration (Satır 16: local video directory)
VIDEO_DIR = Path("../cow_single_videos")  # Updated: videos now in project root
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def initialize_deeplabcut_project():
    """Initialize DeepLabCut SuperAnimal-Quadruped project"""
    project_name = "CowGaitAnalysis"
    experimenter = "Researcher"
    working_dir = Path("./dlc_project")
    working_dir.mkdir(exist_ok=True)
    
    # Check if project already exists
    existing_configs = list(working_dir.glob(f"{project_name}*/config.yaml"))
    if existing_configs:
        config_path = str(existing_configs[0])
        logger.info(f"✅ Using existing DLC project: {config_path}")
        return config_path
    
    # Create new project with SuperAnimal-Quadruped
    logger.info("🔧 Creating DeepLabCut SuperAnimal-Quadruped project...")
    
    # Use a dummy video for initialization
    dummy_video = VIDEO_DIR / "Saglikli" / "cow_0001.mp4"
    if not dummy_video.exists():
        logger.error(f"❌ Dummy video not found: {dummy_video}")
        logger.error("   Please check VIDEO_DIR path in script")
        sys.exit(1)
    
    result = deeplabcut.create_pretrained_project(
        project_name,
        experimenter,
        [str(dummy_video)],
        working_directory=str(working_dir),
        copy_videos=False,
        analyzevideo=False,
        model="superanimal_quadruped",
        videotype=".mp4"
    )
    
    # Handle tuple or string return
    if isinstance(result, tuple):
        config_path = result[0]
    else:
        config_path = result
    
    logger.info(f"✅ DLC project created: {config_path}")
    return config_path

def process_single_video(config_path: str, video_path: Path) -> bool:
    """
    Process a single video (TEST MODE - Satır 22)
    Returns True if successful, False otherwise
    """
    logger.info(f"🎥 Processing test video: {video_path.name}")
    
    try:
        deeplabcut.analyze_videos(
            config_path,
            [str(video_path)],
            videotype=".mp4",
            save_as_csv=True,
            destfolder=str(OUTPUT_DIR)
        )
        
        # Verify output - DeepLabCut creates files with model name in filename
        # Format: videonameDLC_modelname_projectname_shuffle_iteration.csv
        output_csvs = list(OUTPUT_DIR.glob(f"{video_path.stem}DLC*.csv"))
        
        if output_csvs:
            output_csv = output_csvs[0]  # Take first match
            # Check CSV structure
            df = pd.read_csv(output_csv, header=[1,2])
            logger.info(f"✅ Test successful!")
            logger.info(f"   Output: {output_csv.name}")
            logger.info(f"   Shape: {df.shape} (frames x keypoints)")
            return True
        else:
            logger.error(f"❌ No output CSV found for: {video_path.stem}")
            logger.error(f"   Expected pattern: {video_path.stem}DLC*.csv")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error processing video: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def process_batch(config_path: str, video_list: List[Path], resume: bool = True):
    """
    Process all videos with resume capability
    Resume: Skip already processed videos (default: True)
    """
    logger.info(f"🚀 Starting batch processing: {len(video_list)} videos")
    
    # Check already processed videos (resume capability)
    processed = []
    if resume:
        existing_csvs = list(OUTPUT_DIR.glob("*DLC*.csv"))
        # Extract video names from DLC output format
        processed = [csv.stem.split('DLC')[0] for csv in existing_csvs]
        logger.info(f"📊 Found {len(processed)} already processed videos (will skip)")
    
    # Filter out processed videos
    videos_to_process = [v for v in video_list if v.stem not in processed]
    
    if not videos_to_process:
        logger.info("✅ All videos already processed!")
        return
    
    logger.info(f"📹 Videos remaining: {len(videos_to_process)}")
    logger.info(f"⏱️  Estimated time: {len(videos_to_process) * 2}-{len(videos_to_process) * 3} minutes")
    
    # Process with progress bar
    failed = []
    for video_path in tqdm(videos_to_process, desc="Processing videos"):
        try:
            deeplabcut.analyze_videos(
                config_path,
                [str(video_path)],
                videotype=".mp4",
                save_as_csv=True,
                destfolder=str(OUTPUT_DIR)
            )
            
            # Log milestone every 100 videos
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
        logger.warning(f"Failed videos: {failed[:10]}...")  # Show first 10
    logger.info(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(
        description="DeepLabCut Batch Video Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python process_videos.py --test          # Test mode (required first)
    python process_videos.py --batch         # Batch mode
    python process_videos.py --batch --no-resume  # Reprocess all
        """
    )
    parser.add_argument("--test", action="store_true", 
                       help="Test mode: process single video (Satır 22)")
    parser.add_argument("--batch", action="store_true",
                       help="Batch mode: process all videos")
    parser.add_argument("--no-resume", action="store_true",
                       help="Disable resume (reprocess all videos)")
    
    args = parser.parse_args()
    
    if not args.test and not args.batch:
        parser.print_help()
        sys.exit(1)
    
    # Initialize DeepLabCut project
    config_path = initialize_deeplabcut_project()
    
    if args.test:
        # TEST MODE (Satır 22: önce test videosu)
        print("\n" + "="*60)
        print("TEST MODE - Processing Single Video")
        print("="*60)
        
        test_video = VIDEO_DIR / "Saglikli" / "cow_0001.mp4"
        if not test_video.exists():
            logger.error(f"❌ Test video not found: {test_video}")
            logger.error("   Please check VIDEO_DIR path")
            sys.exit(1)
        
        success = process_single_video(config_path, test_video)
        
        if success:
            print("\n" + "="*60)
            print("✅ TEST SUCCESSFUL!")
            print("="*60)
            print(f"Output CSV: {OUTPUT_DIR / 'cow_0001_DLC_SuperAnimal.csv'}")
            print("\nNext steps:")
            print("  1. Review the output CSV file")
            print("  2. If satisfied, run batch processing:")
            print("     python process_videos.py --batch")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("❌ TEST FAILED")
            print("="*60)
            print("Check processing.log for details")
            sys.exit(1)
    
    elif args.batch:
        # BATCH MODE
        print("\n" + "="*60)
        print("⚠️  BATCH PROCESSING MODE")
        print("="*60)
        print(f"Video directory: {VIDEO_DIR}")
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"Estimated time: 35-60 hours for 1167 videos")
        print(f"Resume enabled: {not args.no_resume}")
        print("="*60)
        
        user_confirm = input("\nProceed with batch processing? (yes/no): ").strip().lower()
        if user_confirm != "yes":
            print("❌ Cancelled by user")
            sys.exit(0)
        
        # Collect all videos
        all_videos = []
        for folder in ["Saglikli", "Topal"]:
            folder_path = VIDEO_DIR / folder
            if not folder_path.exists():
                logger.error(f"❌ Folder not found: {folder_path}")
                continue
            
            videos = list(folder_path.glob("*.mp4"))
            all_videos.extend(videos)
            logger.info(f"📁 {folder}: {len(videos)} videos")
        
        if not all_videos:
            logger.error("❌ No videos found!")
            logger.error(f"   Check VIDEO_DIR: {VIDEO_DIR}")
            sys.exit(1)
        
        logger.info(f"📊 Total: {len(all_videos)} videos")
        
        # Process
        process_batch(config_path, all_videos, resume=not args.no_resume)
        
        print("\n" + "="*60)
        print("✅ BATCH PROCESSING COMPLETE")
        print("="*60)
        print(f"Check processing.log for details")
        print(f"Output directory: {OUTPUT_DIR}")
        print("\nNext step: Sync outputs to Google Drive")
        print("  cd ../sync")
        print("  python sync_to_drive.py")
        print("="*60)

if __name__ == "__main__":
    main()

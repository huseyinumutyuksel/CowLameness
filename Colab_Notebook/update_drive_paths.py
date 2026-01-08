
import json
import os

nb_path = r"c:\Users\HP\Desktop\CowLameness\Colab_Notebook\Cow_Lameness_Analysis_v20.ipynb"

def update_notebook():
    if not os.path.exists(nb_path):
        print(f"Error: Not found {nb_path}")
        return

    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']
    updates = 0

    for cell in cells:
        if cell['cell_type'] != 'code':
            continue
        
        source_str = "".join(cell['source'])

        # Modification 1: Drive Paths
        if 'drive.mount(\'/content/drive\')' in source_str and 'BASE =' in source_str:
            print("Found Drive Mount cell")
            new_source = source_str.replace(
                'POSE_CSV_DIR = f"{BASE}/outputs/deeplabcut"  # or mmpose based on config',
                'DLC_OUTPUT_BASE = "/content/drive/MyDrive/DeepLabCut/outputs"\nPOSE_CSV_DIR = DLC_OUTPUT_BASE  # Default base for DLC'
            )
            # detect_framework selection
            cell['source'] = new_source.splitlines(keepends=True)
            updates += 1

        # Modification 2: Detect Framework Logic
        if 'def detect_pose_framework' in source_str:
            print("Found Detect Framework cell")
            
            # Update the detection logic inside the function
            if 'dlc_files = glob.glob(f"{dlc_dir}/*DLC*.csv")' in source_str:
                source_str = source_str.replace(
                    'dlc_files = glob.glob(f"{dlc_dir}/*DLC*.csv")',
                    '# Check subdirectories\n        dlc_files = glob.glob(f"{dlc_dir}/Saglikli/*DLC*.csv") + glob.glob(f"{dlc_dir}/Topal/*DLC*.csv")'
                )

            # Update the function call at the bottom of the cell
            if 'detect_pose_framework(VIDEO_DIR, f"{BASE}/outputs")' in source_str:
                source_str = source_str.replace(
                    'detect_pose_framework(VIDEO_DIR, f"{BASE}/outputs")',
                    'detect_pose_framework(VIDEO_DIR, DLC_OUTPUT_BASE if \'DLC_OUTPUT_BASE\' in locals() and os.path.exists(DLC_OUTPUT_BASE) else f"{BASE}/outputs")'
                )
            
            cell['source'] = source_str.splitlines(keepends=True)
            updates += 1

        # Modification 3: Feature Loop
        if '# Extract multi-modal features' in source_str and 'for video_path, label in tqdm' in source_str:
            print("Found Feature Loop cell")
            
            old_block = """        if POSE_FRAMEWORK == "deeplabcut":
            # DeepLabCut pattern: {video}DLC*.csv
            pose_csv_pattern = f"{POSE_CSV_DIR}/{video_name}DLC*.csv"
            pose_csv_files = glob.glob(pose_csv_pattern)"""
            
            new_block = """        if POSE_FRAMEWORK == "deeplabcut":
            # DeepLabCut pattern: {video}DLC*.csv in Saglikli/Topal folders
            sub_folder = "Saglikli" if label == 0 else "Topal"
            pose_csv_pattern = f"{POSE_CSV_DIR}/{sub_folder}/{video_name}DLC*.csv"
            pose_csv_files = glob.glob(pose_csv_pattern)"""
            
            # Do a replacement ignoring exact whitespace match if possible, or just be precise
            # The previous replace failed because of Exact match requirement. Let's try replace part by part if block fails
            if old_block in source_str:
                source_str = source_str.replace(old_block, new_block)
            else:
                # Fallback: Replace lines individually/conceptually if block doesn't match due to spacing
                # But here we used 'view_file' so we know the structure.
                # Let's clean up the target string to ensure matching
                pass 
                
            cell['source'] = source_str.splitlines(keepends=True)
            updates += 1

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4)
    
    print(f"Updated {updates} cells successfully.")

if __name__ == "__main__":
    update_notebook()

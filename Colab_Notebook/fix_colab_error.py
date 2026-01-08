
import json
import os

nb_path = r"c:\Users\HP\Desktop\CowLameness\Colab_Notebook\Cow_Lameness_Analysis_v20.ipynb"

def fix_colab_error():
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

        # Modification: Fix Detect Framework Logic
        if 'def detect_pose_framework' in source_str:
            print("Found Detect Framework cell")
            
            # The issue: dlc_dir was set to {pose_base_dir}/deeplabcut
            # But the structure is pose_base_dir/Saglikli
            # So dlc_dir should be pose_base_dir
            
            # Replace the subdirectory addition
            if 'dlc_dir = f"{pose_base_dir}/deeplabcut"' in source_str:
                source_str = source_str.replace(
                    'dlc_dir = f"{pose_base_dir}/deeplabcut"',
                    'dlc_dir = pose_base_dir  # Directly use the base for Saglikli/Topal'
                )
            
            # Ensure MMPose also doesn't break if it exists or not, but focus on DLC fix
            # MMPose check: mmpose_dir = f"{pose_base_dir}/mmpose" -> This might be okay if user doesn't have mmpose or if it's there.
            # User only complained about DLC.

            # We also need to make sure the glob logic (which we updated previously) still holds or is reinforced.
            # "dlc_files = glob.glob(f"{dlc_dir}/Saglikli/*DLC*.csv") + glob.glob(f"{dlc_dir}/Topal/*DLC*.csv")"
            # If we run this script on an already updated file, the glob line will be the new one.
            # If we run on original, it will be old one.
            # Let's handle both or assume this runs AFTER the previous update (which seemed to succeed).
            
            cell['source'] = source_str.splitlines(keepends=True)
            updates += 1

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4)
    
    print(f"Fixed {updates} cells.")

if __name__ == "__main__":
    fix_colab_error()

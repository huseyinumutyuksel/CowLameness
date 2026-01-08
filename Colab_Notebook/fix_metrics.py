
import json
import os

nb_path = r"c:\Users\HP\Desktop\CowLameness\Colab_Notebook\Cow_Lameness_Analysis_v20.ipynb"

def fix_metrics_and_KeyError():
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

        # Modification 1: Fix Data Imbalance (Shuffle in Demo Mode)
        # Look for the DEMO_MODE block
        if "DEMO_MODE = True" in source_str and "video_files = video_files[:50]" in source_str:
            print("Found Demo Mode cell")
            
            # Add shuffle before slicing
            old_code = "video_files = video_files[:50]  # Process 50 videos for demo"
            new_code = "import random\n    random.seed(42)\n    random.shuffle(video_files) # Shuffle to get mix of classes\n    video_files = video_files[:50]  # Process 50 videos for demo"
            
            if old_code in source_str:
                source_str = source_str.replace(old_code, new_code)
                cell['source'] = source_str.splitlines(keepends=True)
                updates += 1

        # Modification 2: Fix KeyError: 'features'
        # Look for the final JSON dump block
        if "'features_used': list(config['features'].keys())," in source_str:
            print("Found Final Metrics cell")
            
            source_str = source_str.replace(
                "'features_used': list(config['features'].keys()),",
                "'features_used': list(config.get('features', {}).keys()),"
            )
            cell['source'] = source_str.splitlines(keepends=True)
            updates += 1

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4)
    
    print(f"Fixed {updates} issues (Data Shuffle & KeyError).")

if __name__ == "__main__":
    fix_metrics_and_KeyError()

import json
import os

def main():
    base_dir = r"c:\Users\HP\Desktop\Clone Repos\CowLameness\Colab_Notebook"
    parts = ["_v33_part1.json", "_v33_part2.json", "_v33_part3.json"]
    
    final_cells = []
    
    print("Assembling notebook (JSON mode)...")
    
    for part in parts:
        path = os.path.join(base_dir, part)
        if not os.path.exists(path):
            print(f"❌ Missing part: {path}")
            return
            
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)
            cells = nb.get("cells", [])
            final_cells.extend(cells)
            print(f"  + Added {len(cells)} cells from {part}")

    # Create final notebook structure manually
    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU"
        },
        "cells": final_cells
    }

    out_path = os.path.join(base_dir, "Cow_Lameness_Analysis_v33.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        
    print(f"Notebook assembled: {out_path}")
    print(f"Total cells: {len(final_cells)}")

if __name__ == "__main__":
    main()

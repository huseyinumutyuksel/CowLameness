"""
Assemble Cow_Lameness_Analysis_v32.ipynb from part builder scripts.
"""
import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Run each part builder
for part in ["build_v32_part1.py", "build_v32_part2.py", "build_v32_part3.py"]:
    path = os.path.join(SCRIPT_DIR, part)
    print(f"Running {part}...")
    result = subprocess.run([sys.executable, path], capture_output=True, text=True, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        sys.exit(1)
    print(f"  {result.stdout.strip()}")

# Load and combine cells
print("\nAssembling notebook...")
all_cells = []
for part_file in ["_v32_part1.json", "_v32_part2.json", "_v32_part3.json"]:
    path = os.path.join(SCRIPT_DIR, part_file)
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    all_cells.extend(nb["cells"])
    print(f"  {part_file}: {len(nb['cells'])} cells")

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "A100", "toc_visible": True},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "accelerator": "GPU"
    },
    "cells": all_cells
}

out_path = os.path.join(SCRIPT_DIR, "Cow_Lameness_Analysis_v32.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"\n✅ Notebook: {out_path} ({len(all_cells)} cells)")

# Cleanup
for tmp in ["_v32_part1.json", "_v32_part2.json", "_v32_part3.json"]:
    p = os.path.join(SCRIPT_DIR, tmp)
    if os.path.exists(p):
        os.remove(p)

print("Done!")

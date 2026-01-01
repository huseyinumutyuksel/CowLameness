"""
Update Colab Notebook with Flexible Pose Framework Detection
"""
import json
from pathlib import Path

# Load notebook
notebook_path = Path(r"c:\Users\HP\Desktop\CowLameness\Colab_Notebook\Cow_Lameness_Analysis_v18.ipynb")
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# New cell for pose framework detection
detection_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 2.5 Detect Available Pose Framework"
    ]
}

detection_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "def detect_pose_framework(video_dir, pose_base_dir):\n",
        "    \"\"\"\n",
        "    Detect which pose estimation framework outputs are available.\n",
        "    \n",
        "    Returns:\n",
        "        tuple: (framework_name, pose_csv_dir)\n",
        "            - framework_name: 'deeplabcut', 'mmpose', or None\n",
        "            - pose_csv_dir: path to the pose CSV directory\n",
        "    \"\"\"\n",
        "    dlc_dir = f\"{pose_base_dir}/deeplabcut\"\n",
        "    mmpose_dir = f\"{pose_base_dir}/mmpose\"\n",
        "    \n",
        "    # Check for DeepLabCut outputs\n",
        "    dlc_available = False\n",
        "    dlc_files = []\n",
        "    if os.path.exists(dlc_dir):\n",
        "        dlc_files = glob.glob(f\"{dlc_dir}/*DLC*.csv\")\n",
        "        dlc_available = len(dlc_files) > 0\n",
        "    \n",
        "    # Check for MMPose outputs\n",
        "    mmpose_available = False\n",
        "    mmpose_files = []\n",
        "    if os.path.exists(mmpose_dir):\n",
        "        mmpose_files = glob.glob(f\"{mmpose_dir}/*_MMPose.csv\")\n",
        "        mmpose_available = len(mmpose_files) > 0\n",
        "    \n",
        "    # Decision logic\n",
        "    if dlc_available and mmpose_available:\n",
        "        print(\"⚠️  Both DeepLabCut and MMPose outputs detected!\")\n",
        "        print(f\"   DeepLabCut: {len(dlc_files)} files\")\n",
        "        print(f\"   MMPose: {len(mmpose_files)} files\")\n",
        "        print(\"\\nWhich framework would you like to use?\")\n",
        "        print(\"  1. DeepLabCut\")\n",
        "        print(\"  2. MMPose\")\n",
        "        \n",
        "        choice = input(\"Enter choice (1 or 2): \").strip()\n",
        "        \n",
        "        if choice == \"1\":\n",
        "            return \"deeplabcut\", dlc_dir\n",
        "        elif choice == \"2\":\n",
        "            return \"mmpose\", mmpose_dir\n",
        "        else:\n",
        "            print(\"❌ Invalid choice. Defaulting to DeepLabCut.\")\n",
        "            return \"deeplabcut\", dlc_dir\n",
        "    \n",
        "    elif dlc_available:\n",
        "        print(f\"✅ DeepLabCut outputs detected: {len(dlc_files)} files\")\n",
        "        return \"deeplabcut\", dlc_dir\n",
        "    \n",
        "    elif mmpose_available:\n",
        "        print(f\"✅ MMPose outputs detected: {len(mmpose_files)} files\")\n",
        "        return \"mmpose\", mmpose_dir\n",
        "    \n",
        "    else:\n",
        "        print(\"❌ ERROR: No pose estimation outputs found!\")\n",
        "        print(f\"   Checked directories:\")\n",
        "        print(f\"     - {dlc_dir}\")\n",
        "        print(f\"     - {mmpose_dir}\")\n",
        "        print(\"\\n   Please run pose estimation first:\")\n",
        "        print(\"     - DeepLabCut: python DeepLabCut/process_videos.py --batch\")\n",
        "        print(\"     - MMPose: python MMPose/process_videos.py --batch\")\n",
        "        return None, None\n",
        "\n",
        "# Detect framework\n",
        "POSE_FRAMEWORK, POSE_CSV_DIR = detect_pose_framework(VIDEO_DIR, f\"{BASE}/outputs\")\n",
        "\n",
        "if POSE_FRAMEWORK is None:\n",
        "    raise RuntimeError(\"No pose estimation data found. Cannot proceed.\")\n",
        "\n",
        "print(f\"\\n🎯 Using pose framework: {POSE_FRAMEWORK.upper()}\")\n",
        "print(f\"   CSV directory: {POSE_CSV_DIR}\")\n",
        "\n",
        "# Update config\n",
        "config['pose_framework'] = POSE_FRAMEWORK\n",
        "print(f\"\\n✅ Config updated with detected framework\")"
    ]
}

# Find the index to insert (after cell with "Mount Drive & Load Config")
insert_index = None
for i, cell in enumerate(notebook['cells']):
    if cell.get('cell_type') == 'code' and any('drive.mount' in line for line in cell.get('source', [])):
        insert_index = i + 1
        break

if insert_index is None:
    print("❌ Could not find mount drive cell")
    exit(1)

# Insert new cells
notebook['cells'].insert(insert_index, detection_cell)
notebook['cells'].insert(insert_index + 1, detection_code_cell)

print(f"✅ Inserted detection cells at index {insert_index}")

# Update pose loading logic (find the cell with pose CSV loading)
for i, cell in enumerate(notebook['cells']):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        source_text = ''.join(source)
        
        # Find the cell with pose CSV loading
        if 'pose_csv = f"{POSE_CSV_DIR}/{video_name}_DLC_SuperAnimal.csv"' in source_text:
            print(f"✅ Found pose loading cell at index {i}")
            
            # Replace the pose loading logic
            new_source = []
            skip_lines = False
            for line in source:
                # Start of pose loading section
                if '# 5. Load pose CSV' in line:
                    new_source.append(line)
                    new_source.append("        video_name = Path(video_path).stem\n")
                    new_source.append("        \n")
                    new_source.append("        if POSE_FRAMEWORK == \"deeplabcut\":\n")
                    new_source.append("            # DeepLabCut pattern: {video}DLC*.csv\n")
                    new_source.append("            pose_csv_pattern = f\"{POSE_CSV_DIR}/{video_name}DLC*.csv\"\n")
                    new_source.append("            pose_csv_files = glob.glob(pose_csv_pattern)\n")
                    new_source.append("            \n")
                    new_source.append("            if pose_csv_files:\n")
                    new_source.append("                pose_csv = pose_csv_files[0]  # Take first match\n")
                    new_source.append("                pose_df = pd.read_csv(pose_csv, header=[1,2])  # DLC has multi-level header\n")
                    new_source.append("                pose_features = pose_df.values.mean(axis=0)[:50]\n")
                    new_source.append("            else:\n")
                    new_source.append("                pose_features = np.zeros(50)\n")
                    new_source.append("                \n")
                    new_source.append("        elif POSE_FRAMEWORK == \"mmpose\":\n")
                    new_source.append("            # MMPose pattern: {video}_MMPose.csv\n")
                    new_source.append("            pose_csv = f\"{POSE_CSV_DIR}/{video_name}_MMPose.csv\"\n")
                    new_source.append("            \n")
                    new_source.append("            if os.path.exists(pose_csv):\n")
                    new_source.append("                pose_df = pd.read_csv(pose_csv, index_col=0)  # MMPose has simple header\n")
                    new_source.append("                pose_features = pose_df.values.mean(axis=0)[:50]\n")
                    new_source.append("            else:\n")
                    new_source.append("                pose_features = np.zeros(50)\n")
                    skip_lines = True
                    continue
                
                # Skip old pose loading lines
                if skip_lines:
                    if '# Combine all features' in line:
                        skip_lines = False
                        new_source.append("        \n")
                        new_source.append(line)
                    continue
                
                new_source.append(line)
            
            cell['source'] = new_source
            print(f"✅ Updated pose loading logic")
            break

# Save updated notebook
output_path = notebook_path.parent / "Cow_Lameness_Analysis_v19.ipynb"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=4, ensure_ascii=False)

print(f"\n✅ Notebook updated successfully!")
print(f"   Output: {output_path}")
print(f"\nChanges made:")
print(f"  1. Added pose framework detection cell (automatic)")
print(f"  2. Updated pose CSV loading to support both DeepLabCut and MMPose")
print(f"  3. Created new version: v19")

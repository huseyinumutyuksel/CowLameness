
import json
import os

nb_path = r"c:\Users\HP\Desktop\CowLameness\Colab_Notebook\Cow_Lameness_Analysis_v20.ipynb"

def add_visualization_cell():
    if not os.path.exists(nb_path):
        print(f"Error: Not found {nb_path}")
        return

    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Check if cell already exists to avoid duplicates
    last_cell_source = "".join(nb['cells'][-1]['source'])
    if "Generate 10 Healthy and 10 Lame Sample Videos" in last_cell_source:
        print("Visualization cell already exists.")
        return

    new_cell_markdown = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 11. Generate Visual Evidence (Processed Videos)\n",
            "Generates 10 Healthy and 10 Lame sample videos with overlays:\n",
            "- **Bounding Box (YOLO)**\n",
            "- **Segmentation Mask (SAM)**\n",
            "- **Pose Skeleton (DeepLabCut/MMPose)**"
        ]
    }

    new_cell_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Create output directory for processed videos\n",
            "VIS_DIR = f\"{OUTPUT_DIR}/processed_video_samples\"\n",
            "os.makedirs(VIS_DIR, exist_ok=True)\n",
            "\n",
            "def process_and_visualize_video(video_path, label, model_yolo, predictor_sam, pose_base_dir, output_path):\n",
            "    cap = cv2.VideoCapture(video_path)\n",
            "    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))\n",
            "    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))\n",
            "    fps = cap.get(cv2.CAP_PROP_FPS)\n",
            "    \n",
            "    # Define Codec\n",
            "    fourcc = cv2.VideoWriter_fourcc(*'mp4v')\n",
            "    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))\n",
            "    \n",
            "    video_name = Path(video_path).stem\n",
            "    \n",
            "    # Load Pose CSV if available\n",
            "    pose_coords = None\n",
            "    if POSE_FRAMEWORK == \"deeplabcut\":\n",
            "        sub_folder = \"Saglikli\" if label == 0 else \"Topal\"\n",
            "        pose_pattern = f\"{pose_base_dir}/{sub_folder}/{video_name}DLC*.csv\"\n",
            "        pose_files = glob.glob(pose_pattern)\n",
            "        if pose_files:\n",
            "            df = pd.read_csv(pose_files[0], header=[1, 2])\n",
            "            pose_coords = df.values # Numpy array of coords\n",
            "            \n",
            "    elif POSE_FRAMEWORK == \"mmpose\":\n",
            "        pose_file = f\"{pose_base_dir}/mmpose/{video_name}_MMPose.csv\"\n",
            "        if os.path.exists(pose_file):\n",
            "             df = pd.read_csv(pose_file, index_col=0)\n",
            "             pose_coords = df.values\n",
            "    \n",
            "    frame_idx = 0\n",
            "    while True:\n",
            "        ret, frame = cap.read()\n",
            "        if not ret:\n",
            "            break\n",
            "            \n",
            "        # 1. YOLO Detection\n",
            "        bbox = detect_largest_cow(frame, model_yolo)\n",
            "        \n",
            "        overlay = frame.copy()\n",
            "        alpha = 0.4\n",
            "        \n",
            "        if bbox is not None:\n",
            "            x1, y1, x2, y2 = bbox\n",
            "            \n",
            "            # Draw BBox\n",
            "            color = (0, 255, 0) if label == 0 else (0, 0, 255) # Green for Healthy, Red for Lame\n",
            "            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)\n",
            "            \n",
            "            # 2. SAM Segmentation (Only on middle frame or stride for speed, but here do every frame for vis?)\n",
            "            # SAM is slow on every frame. Optimization: Use YOLO box as simple mask or run SAM every 5th frame.\n",
            "            # For full demo quality, let's run SAM.\n",
            "            try:\n",
            "                # We reuse the predictor but it needs 'set_image' every time which is slow.\n",
            "                # For 20 videos, this might take a while. Let's add 'Fast Mode' warning.\n",
            "                # Or just draw semi-transparent box as 'Segmentation Area' if SAM is too slow.\n",
            "                # Let's try to run SAM.\n",
            "                predictor_sam.set_image(frame)\n",
            "                masks, _, _ = predictor_sam.predict(box=bbox, multimask_output=False)\n",
            "                mask = masks[0]\n",
            "                \n",
            "                # Apply mask overlay\n",
            "                overlay[mask] = color\n",
            "                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)\n",
            "                \n",
            "            except Exception as e:\n",
            "                pass\n",
            "        \n",
            "        # 3. Pose Landmarks\n",
            "        if pose_coords is not None and frame_idx < len(pose_coords):\n",
            "            coords = pose_coords[frame_idx]\n",
            "            # DLC structure: [bodypart1_x, bodypart1_y, prob, bodypart2_x, ...]\n",
            "            # Iterate by 3\n",
            "            for i in range(0, len(coords), 3):\n",
            "                px, py, prob = coords[i], coords[i+1], coords[i+2]\n",
            "                if prob > 0.5:\n",
            "                    cv2.circle(frame, (int(px), int(py)), 4, (0, 255, 255), -1) # Yellow dots\n",
            "\n",
            "        # Label\n",
            "        text = \"HEALTHY\" if label == 0 else \"LAME\"\n",
            "        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)\n",
            "        \n",
            "        out.write(frame)\n",
            "        frame_idx += 1\n",
            "        \n",
            "    cap.release()\n",
            "    out.release()\n",
            "\n",
            "# Select 10 Random videos from each class\n",
            "import random\n",
            "\n",
            "healthy_videos = [v for v, l in video_files if l == 0]\n",
            "lame_videos = [v for v, l in video_files if l == 1]\n",
            "\n",
            "selected_healthy = random.sample(healthy_videos, min(10, len(healthy_videos)))\n",
            "selected_lame = random.sample(lame_videos, min(10, len(lame_videos)))\n",
            "\n",
            "print(f\"\\n\ud83c\udfa5 Generating Visual Evidence...\")\n",
            "print(f\"   - 10 Healthy Videos\")\n",
            "print(f\"   - 10 Lame Videos\")\n",
            "print(f\"   - Saving to: {VIS_DIR}\")\n",
            "\n",
            "# Process Healthy\n",
            "for v_path in tqdm(selected_healthy, desc=\"Generating Healthy Samples\"):\n",
            "    try:\n",
            "        out_name = f\"Healthy_{Path(v_path).stem}_vis.mp4\"\n",
            "        process_and_visualize_video(\n",
            "            v_path, 0, yolo_model, sam_predictor, POSE_CSV_DIR, f\"{VIS_DIR}/{out_name}\"\n",
            "        )\n",
            "    except Exception as e:\n",
            "        print(f\"Error processing {v_path}: {e}\")\n",
            "\n",
            "# Process Lame\n",
            "for v_path in tqdm(selected_lame, desc=\"Generating Lame Samples\"):\n",
            "    try:\n",
            "        out_name = f\"Lame_{Path(v_path).stem}_vis.mp4\"\n",
            "        process_and_visualize_video(\n",
            "            v_path, 1, yolo_model, sam_predictor, POSE_CSV_DIR, f\"{VIS_DIR}/{out_name}\"\n",
            "        )\n",
            "    except Exception as e:\n",
            "        print(f\"Error processing {v_path}: {e}\")\n",
            "\n",
            "print(\"\\n\u2705 Visualization Complete. Please check the drive folder.\")"
        ]
    }

    nb['cells'].append(new_cell_markdown)
    nb['cells'].append(new_cell_code)

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4)
    
    print("Added visualization cell successfully.")

if __name__ == "__main__":
    add_visualization_cell()

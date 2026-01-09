"""
Attention Visualization Module
==============================
Provides interpretable visualizations for lameness detection
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from typing import List, Tuple, Optional
import torch


def visualize_attention_bar(attention_weights: np.ndarray,
                            prediction: float,
                            true_label: float,
                            save_path: Optional[str] = None,
                            fps: float = 30.0,
                            window_frames: int = 60,
                            stride_frames: int = 15) -> None:
    """
    Visualize temporal attention weights as a bar chart.
    
    Args:
        attention_weights: Array of shape (N,) with attention weights
        prediction: Model prediction (0-3 severity or 0-1 probability)
        true_label: Ground truth label
        save_path: Optional path to save figure
        fps: Video FPS for time calculation
        window_frames: Frames per window
        stride_frames: Stride between windows
    """
    n_windows = len(attention_weights)
    
    # Calculate time positions
    window_duration = window_frames / fps
    stride_duration = stride_frames / fps
    time_starts = [i * stride_duration for i in range(n_windows)]
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Create bar chart
    colors = plt.cm.Reds(attention_weights / attention_weights.max())
    bars = ax.bar(range(n_windows), attention_weights, color=colors, edgecolor='black', alpha=0.8)
    
    # Highlight top-3 windows
    top_k = min(3, n_windows)
    top_indices = np.argsort(attention_weights)[-top_k:]
    for idx in top_indices:
        bars[idx].set_edgecolor('red')
        bars[idx].set_linewidth(3)
    
    # Labels
    ax.set_xlabel('Window Index (Time)', fontsize=12)
    ax.set_ylabel('Attention Weight', fontsize=12)
    
    # Prediction info
    label_text = "Lame" if true_label > 0.5 else "Healthy"
    ax.set_title(f'Temporal Attention Weights\nTrue: {label_text} | Pred: {prediction:.2f}', fontsize=14)
    
    # Add time axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    time_ticks = np.linspace(0, n_windows - 1, min(10, n_windows))
    ax2.set_xticks(time_ticks)
    ax2.set_xticklabels([f'{time_starts[int(t)]:.1f}s' for t in time_ticks])
    ax2.set_xlabel('Time', fontsize=10)
    
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention visualization to {save_path}")
    
    plt.show()


def visualize_attention_heatmap(attention_weights: np.ndarray,
                                video_path: str,
                                output_path: str,
                                fps: float = 30.0,
                                window_frames: int = 60,
                                stride_frames: int = 15) -> None:
    """
    Create video with attention heatmap overlay.
    
    Args:
        attention_weights: Array of shape (N,) with attention weights
        video_path: Path to original video
        output_path: Path to save annotated video
        fps: Video FPS
        window_frames: Frames per window
        stride_frames: Stride between windows
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height + 80))
    
    # Normalize attention
    attn_norm = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min() + 1e-6)
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Find which window this frame belongs to
        window_idx = min(frame_idx // stride_frames, len(attention_weights) - 1)
        attn_value = attn_norm[window_idx]
        
        # Create attention bar at bottom
        bar_height = 80
        bar_frame = np.zeros((bar_height, width, 3), dtype=np.uint8)
        
        # Draw attention timeline
        bar_width_per_window = width // len(attention_weights)
        for i, a in enumerate(attn_norm):
            color = (int(255 * a), int(50 * (1 - a)), int(255 * (1 - a)))  # Red gradient
            x1 = i * bar_width_per_window
            x2 = (i + 1) * bar_width_per_window
            cv2.rectangle(bar_frame, (x1, 20), (x2, bar_height - 10), color, -1)
            
            # Highlight current window
            if i == window_idx:
                cv2.rectangle(bar_frame, (x1, 15), (x2, bar_height - 5), (255, 255, 255), 2)
        
        # Add text
        cv2.putText(bar_frame, f"Attention: {attention_weights[window_idx]:.3f}", 
                    (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add subtle color overlay to frame based on attention
        overlay = frame.copy()
        if attn_value > 0.5:
            cv2.rectangle(overlay, (0, 0), (width, 30), (0, 0, 255), -1)
            cv2.putText(overlay, "HIGH ATTENTION", (10, 22), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # Combine frame and attention bar
        combined = np.vstack([frame, bar_frame])
        out.write(combined)
        
        frame_idx += 1
    
    cap.release()
    out.release()
    print(f"Saved attention video to {output_path}")


def generate_clinical_report(video_name: str,
                             attention_weights: np.ndarray,
                             prediction: float,
                             true_label: float,
                             save_path: str,
                             fps: float = 30.0,
                             stride_frames: int = 15) -> None:
    """
    Generate clinical report with lameness indicators.
    """
    # Find high attention windows
    threshold = np.percentile(attention_weights, 75)
    high_attn_windows = np.where(attention_weights >= threshold)[0]
    
    # Calculate time ranges
    time_ranges = []
    for win_idx in high_attn_windows:
        start_time = win_idx * stride_frames / fps
        end_time = start_time + 2.0  # Approx 2 sec window
        time_ranges.append((start_time, end_time, attention_weights[win_idx]))
    
    # Generate report
    severity_text = {
        0: "Healthy (Score: 0)",
        1: "Mild Lameness (Score: 1)",
        2: "Moderate Lameness (Score: 2)",
        3: "Severe Lameness (Score: 3)"
    }
    
    pred_category = min(3, max(0, int(round(prediction))))
    true_category = min(3, max(0, int(round(true_label))))
    
    report = f"""
# Clinical Lameness Report

## Video: {video_name}

### Prediction
- **Severity Score**: {prediction:.2f} / 3.0
- **Category**: {severity_text.get(pred_category, 'Unknown')}
- **Ground Truth**: {severity_text.get(true_category, 'Unknown')}

### Temporal Analysis
The model identified {len(high_attn_windows)} high-attention windows indicating potential lameness indicators:

| Time Range | Attention Weight | Priority |
|------------|------------------|----------|
"""
    
    for i, (start, end, attn) in enumerate(sorted(time_ranges, key=lambda x: -x[2])[:5]):
        priority = "🔴 High" if attn >= np.percentile(attention_weights, 90) else "🟡 Medium"
        report += f"| {start:.1f}s - {end:.1f}s | {attn:.4f} | {priority} |\n"
    
    report += f"""
### Recommendations
"""
    
    if pred_category >= 2:
        report += "- ⚠️ **Immediate veterinary examination recommended**\n"
        report += "- Review video segments at high-attention times\n"
    elif pred_category == 1:
        report += "- 👁️ Monitor closely over next 48 hours\n"
        report += "- Consider hoof inspection\n"
    else:
        report += "- ✅ No immediate action required\n"
        report += "- Continue routine monitoring\n"
    
    # Save report
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Clinical report saved to {save_path}")


if __name__ == "__main__":
    # Test visualization
    print("Testing attention visualization...")
    
    # Dummy data
    attn = np.random.rand(20)
    attn[5:8] = 0.8 + np.random.rand(3) * 0.2  # Simulate high attention region
    
    visualize_attention_bar(attn, prediction=2.3, true_label=3.0)
    print("✅ Visualization test passed!")


import torch
from transformers import VideoMAEModel, VideoMAEConfig

def verify_shape():
    print("--- Testing VideoMAE Input Shape ---")
    config = VideoMAEConfig(
        image_size=224, num_frames=16, 
        hidden_size=192, 
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=768
    )
    model = VideoMAEModel(config)
    
    # CASE 1: (B, T, C, H, W) - Likely Correct
    print("\nCase 1: (B=1, T=16, C=3, H=224, W=224)")
    x1 = torch.randn(1, 16, 3, 224, 224)
    try:
        model(pixel_values=x1)
        print("✅ Case 1 passed!")
    except Exception as e:
        print(f"❌ Case 1 failed: {e}")
        
    # CASE 2: (B, C, T, H, W) - My previous code
    print("\nCase 2: (B=1, C=3, T=16, H=224, W=224)")
    x2 = torch.randn(1, 3, 16, 224, 224)
    try:
        model(pixel_values=x2)
        print("✅ Case 2 passed!")
    except Exception as e:
        print(f"❌ Case 2 failed: {e}")

if __name__ == "__main__":
    verify_shape()

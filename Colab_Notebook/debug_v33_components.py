
import torch
import numpy as np
from transformers import VideoMAEImageProcessor, VideoMAEModel, VideoMAEConfig

def test_processor_behavior():
    print("\n--- Testing VideoMAEImageProcessor ---")
    model_id = "MCG-NJU/videomae-base"
    try:
        processor = VideoMAEImageProcessor.from_pretrained(model_id)
    except Exception as e:
        print(f"Failed to load processor: {e}")
        return

    # Simulate 8 clips * 16 frames = 128 frames
    # Random frames: (128, 224, 224, 3) - standard uint8 images
    num_frames = 128
    frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(num_frames)]
    
    print(f"Input: List of {len(frames)} frames.")
    
    # Process
    try:
        inputs = processor(frames, return_tensors="pt")
        pixel_values = inputs["pixel_values"] # (B, T, C, H, W)
        print(f"Output Shape: {pixel_values.shape}")
        
        if pixel_values.shape[1] != num_frames:
            print(f"⚠️ WARNING: Processor changed frame count! Expected {num_frames}, got {pixel_values.shape[1]}")
            print("Likely cause: implied truncation or fixed-size resizing.")
        else:
            print("✅ Processor preserved frame count.")
            
    except Exception as e:
        print(f"❌ Processor error: {e}")

def verify_cls_token():
    print("\n--- Testing VideoMAEModel Output ---")
    model_id = "MCG-NJU/videomae-base"
    try:
        # Load config only to be fast, or minimal model
        # using from_pretrained to be accurate to user's case
        model = VideoMAEModel.from_pretrained(model_id)
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Dummy input: (1, 16, 3, 224, 224) -> Standard pretraining shape
    # The user folds batch*clips, so it sends standard video chunks.
    pixel_values = torch.randn(1, 16, 3, 224, 224)
    
    with torch.no_grad():
        outputs = model(pixel_values)
        
    last_hidden = outputs.last_hidden_state
    print(f"Last Hidden State Shape: {last_hidden.shape}")
    # Shape should be (B, SequenceLength, Hidden)
    # SequenceLength = (T/2 * H/16 * W/16) + 1 (if CLS?)
    # 16/2=8, 224/16=14. 8*14*14 = 1568 patches.
    
    # Check if index 0 is CLS
    # We can't strictly know semantically, but we can check shape.
    
    # Note: VideoMAE uses MAE strategy. Standard MAE *does* add a CLS token.
    # Let's verify the sequence length.
    
    expected_patches = (16 // 2) * (224 // 16) * (224 // 16)
    print(f"Expected patches (excluding CLS): {expected_patches}")
    print(f"Actual tokens: {last_hidden.shape[1]}")
    
    if last_hidden.shape[1] == expected_patches:
         print("⚠️ WARNING: No CLS token found! (Sequence length == Patch count)")
         print("User code `last_hidden_state[:, 0, :]` might be taking the first patch, not CLS.")
    elif last_hidden.shape[1] == expected_patches + 1:
        print("✅ CLS token likely present (Sequence length == Patch count + 1)")
    else:
        print(f"❓ Unexpected token count.")

if __name__ == "__main__":
    test_processor_behavior()
    verify_cls_token()


import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))
from inference import predict, draw_rectangles_on_mask, DEVICE
from model import ResNetSegmentation
from PIL import Image
import torch
import numpy as np
import cv2

# Configuration
TEST_IMAGE = "data/cracks.v1-cracks-f.coco/test/2056_jpg.rf.c2c86bb2aa54ac0df349c42cbdfc1315.jpg"
MODEL_PATH = "best_model.pth"
PROMPT = "segment crack"

def test_visualization():
    print(f"Testing visualization on {TEST_IMAGE}...")
    
    # 1. Load model directly to inspect raw values if needed, otherwise use predict
    # We will use predict but inspect internals if possible or just rely on predict returning mask
    try:
        image, mask = predict(TEST_IMAGE, PROMPT, MODEL_PATH)
        print("Inference successful.")
    except Exception as e:
        print(f"Inference failed: {e}")
        return

    # Check mask stats
    print(f"Mask shape: {mask.shape}")
    print(f"Mask unique values: {np.unique(mask)}")
    print(f"Mask sum: {mask.sum()}")
    
    # Check predictions raw (re-run part of predict logic)
    # Load model
    model = ResNetSegmentation().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_pil = Image.open(TEST_IMAGE).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits)
        
    pred_np = probs[0, 0].cpu().numpy()
    print(f"Raw prediction range: {pred_np.min()} - {pred_np.max()}")
    threshold = 0.5 
    print(f"Threshold used: {threshold}")
    
    # Save the raw mask for debugging
    mask_debug_path = "mask_debug.png"
    Image.fromarray(mask).save(mask_debug_path)
    print(f"Saved raw mask to {mask_debug_path}")
    
    # 2. Draw rectangles
    # Convert mask to uint8 if not
    mask_uint8 = np.uint8(mask)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} contours.")
    
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        print(f"Contour {i}: size={len(contour)}, rect={w}x{h} at ({x},{y})")

    color = (0, 255, 0)  # Green
    marked_img = draw_rectangles_on_mask(image, mask, thickness=2, color=color)
    
    # 3. Save output
    output_path = "test_visualization_debug.png"
    marked_img.save(output_path)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    test_visualization()

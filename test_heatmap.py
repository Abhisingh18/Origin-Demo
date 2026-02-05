
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))
from inference import SegModel, DEVICE
from PIL import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Configuration
TEST_IMAGE = "data/cracks.v1-cracks-f.coco/test/2056_jpg.rf.c2c86bb2aa54ac0df349c42cbdfc1315.jpg"
MODEL_PATH = "best_model.pth"

def generate_heatmap():
    print(f"Generating heatmap for {TEST_IMAGE}...")
    
    # Load model
    model = SegModel().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model loaded.")
    else:
        print("Model not found!")
        return
    model.eval()
    
    # Prepare image
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    img_pil = Image.open(TEST_IMAGE).convert("RGB")
    original_size = img_pil.size
    img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        output = model(img_tensor)
        
    # Get raw probability map (0-1)
    probs = output[0, 0].cpu().numpy()
    
    print(f"Probability Stats: Min={probs.min():.4f}, Max={probs.max():.4f}, Mean={probs.mean():.4f}")
    
    # Resize probs to original size for visualization
    # We use bilinear here just for smooth visualization of the heatmap
    probs_uint8 = (probs * 255).astype(np.uint8)
    probs_img = Image.fromarray(probs_uint8).resize(original_size, Image.Resampling.BILINEAR)
    probs_np = np.array(probs_img)
    
    # Apply colormap
    heatmap = cv2.applyColorMap(probs_np, cv2.COLORMAP_JET)
    
    # Overlay on original image
    original_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(original_cv, 0.6, heatmap, 0.4, 0)
    
    # Save results
    cv2.imwrite("debug_heatmap_raw.png", heatmap)
    cv2.imwrite("debug_heatmap_overlay.png", overlay)
    
    # Also save a histogram of probabilities
    plt.figure()
    plt.hist(probs.flatten(), bins=100, log=True)
    plt.title("Prediction Probability Distribution")
    plt.savefig("debug_hist.png")
    
    print("Saved debug_heatmap_raw.png, debug_heatmap_overlay.png, debug_hist.png")

if __name__ == "__main__":
    generate_heatmap()

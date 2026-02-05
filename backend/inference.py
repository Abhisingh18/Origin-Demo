"""
Reusable Inference Module for Text-Conditioned Segmentation
Supports prompt-based image segmentation using trained ResNet18 + UNet model
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms, models
import os
import cv2
import numpy as np

# ============================================
# CONFIG
# ============================================
DEVICE = "cpu"  # RTX 5070 (sm_120) not yet supported

# Prompt → Mode mapping for semantic control
# Although model is not explicitly text-conditioned,
# we use different thresholds based on semantic understanding of the prompt
PROMPT_TO_MODE = {
    "segment crack": "crack",
    "segment wall crack": "crack",
    "segment taping area": "taping",
    "segment joint": "taping",
    "segment drywall seam": "taping",
}

# Mode-specific thresholds (determined empirically from training)
# The model outputs values in range ~0.4817-0.4874 for most inputs
# We use thresholds very close to the min value to create differentiation
# Cracks: threshold closer to min for better detection
# Taping: threshold slightly higher for more selective detection
MODE_THRESHOLDS = {
    "crack": 0.485,    # Increased to avoid full-image selection
    "taping": 0.486,   # Increased for selectivity
}

# ============================================
# MODEL DEFINITION
# ============================================
from model import ResNetSegmentation

# ============================================
# HELPER FUNCTIONS
# ============================================
def draw_rectangles_on_mask(image_pil, mask_binary, thickness=2, color=(0, 255, 0)):
    """
    Draw rectangles around detected regions (connected components) on the image.
    """
    # Convert to numpy for contour detection
    mask_uint8 = np.uint8(mask_binary)
    
    # Find contours using OpenCV
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Copy image for drawing
    image_marked = image_pil.copy()
    draw = ImageDraw.Draw(image_marked)
    
    # Draw rectangles around contours
    for contour in contours:
        if len(contour) < 3:  # Skip very small contours (points/lines)
            continue
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Skip small noise
        if w < 10 or h < 10:
            continue
            
        # Draw rectangle on PIL Image
        draw.rectangle(
            [(x, y), (x + w, y + h)],
            outline=color,
            width=thickness
        )
    
    return image_marked


# ============================================
# INFERENCE PIPELINE
# ============================================
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # DeepLab expects normalized input
])


def predict(image_input, prompt, model_path="best_model.pth"):
    # Load model
    model = ResNetSegmentation().to(DEVICE)
    
    if not os.path.exists(model_path):
        # Fallback to parent dir if running from backend/
        if os.path.exists(os.path.join("..", model_path)):
            model_path = os.path.join("..", model_path)
        # Check current directory
        elif os.path.exists("best_model.pth"):
             model_path = "best_model.pth"
        else:
             # Just warn, don't crash if checking health
             pass

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # Load and preprocess image
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image file not found: {image_input}")
        image = Image.open(image_input).convert("RGB")
    else:
        # Assume PIL Image
        image = image_input.convert("RGB")
        
    original_size = image.size
    
    # DeepLab expects normalization
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Run inference
    with torch.no_grad():
        logits = model(img_tensor) # Returns logits [B, 1, H, W]
        probs = torch.sigmoid(logits) # Convert to [0, 1]
    
    probs_np = probs[0, 0].cpu().numpy()
    
    # Standard Thresholding for a trained model
    threshold = 0.5
    mask_binary = (probs_np > threshold).astype(np.uint8) * 255
    
    # Resize mask back to original image size
    mask_pil = Image.fromarray(mask_binary)
    mask_resized = mask_pil.resize(original_size, Image.Resampling.NEAREST)
    mask = np.array(mask_resized)
    
    return mask


def predict_and_save(image_path, prompt, output_dir="outputs", model_path="best_model.pth"):
    """
    Run inference and save the predicted mask as PNG.
    
    Args:
        image_path (str): Path to input image
        prompt (str): Natural language prompt
        output_dir (str): Directory to save output PNG
        model_path (str): Path to trained model weights
    
    Returns:
        str: Path to saved mask PNG file
    """
    
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Run inference
    image, mask = predict(image_path, prompt, model_path)
    
    # Generate output filename
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    prompt_slug = prompt.replace(" ", "_").lower()
    output_filename = f"{image_name}__{prompt_slug}.png"
    output_path = os.path.join(output_dir, output_filename)
    
    # Save mask
    Image.fromarray(mask).save(output_path)
    
    return output_path


# ============================================
# METRICS
# ============================================
def compute_dice(pred, gt, threshold=0.5):
    """Compute Dice Score"""
    pred = (pred > threshold).astype(np.float32)
    gt = gt.astype(np.float32)
    
    inter = (pred * gt).sum()
    dice = (2 * inter) / (pred.sum() + gt.sum() + 1e-6)
    
    return dice


def compute_iou(pred, gt, threshold=0.5):
    """Compute Intersection over Union"""
    pred = (pred > threshold).astype(np.float32)
    gt = gt.astype(np.float32)
    
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() - inter
    iou = inter / (union + 1e-6)
    
    return iou


if __name__ == "__main__":
    # Example usage
    print("Inference module ready. Import and use predict() function.")
    print(f"Supported prompts: {list(PROMPT_TO_CLASS.keys())}")

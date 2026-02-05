
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))
from inference import predict, draw_rectangles_on_mask, PROMPT_TO_MODE
from PIL import Image
import torch

# Configuration
TEST_IMAGE = "data/cracks.v1-cracks-f.coco/test/2056_jpg.rf.c2c86bb2aa54ac0df349c42cbdfc1315.jpg"
MODEL_PATH = "best_model.pth"
PROMPT = "segment crack"

def test_visualization():
    print(f"Testing visualization on {TEST_IMAGE}...")
    
    if not os.path.exists(TEST_IMAGE):
        print("Error: Test image not found")
        return

    # 1. Run inference
    try:
        original_img, mask = predict(TEST_IMAGE, PROMPT, MODEL_PATH)
        print("Inference successful.")
    except Exception as e:
        print(f"Inference failed: {e}")
        return

    # 2. Draw rectangles
    color = (0, 255, 0)  # Green
    marked_img = draw_rectangles_on_mask(original_img, mask, thickness=2, color=color)
    
    # 3. Save output
    output_path = "test_visualization_output.png"
    marked_img.save(output_path)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    test_visualization()

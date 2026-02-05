"""
Evaluation Script for Text-Conditioned Segmentation Model
Computes mIoU, Dice Score, and generates visual examples
"""

import torch
import numpy as np
from PIL import Image
import json
import os
from pathlib import Path
from tqdm import tqdm
from model import SegModel
from inference import predict, PROMPT_TO_MODE
from dataset import SegmentationDataset
from pycocotools.coco import COCO

DEVICE = "cpu"
MODEL_PATH = "best_model.pth"
RESULTS_DIR = "../reports"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/visuals", exist_ok=True)


def compute_iou(pred, gt, threshold=0.5):
    """Compute Intersection over Union"""
    pred_binary = (pred > threshold).astype(np.float32)
    gt_binary = (gt > 0).astype(np.float32)
    
    intersection = np.sum(pred_binary * gt_binary)
    union = np.sum(pred_binary) + np.sum(gt_binary) - intersection
    
    if union == 0:
        return 1.0 if np.sum(pred_binary) == 0 and np.sum(gt_binary) == 0 else 0.0
    
    return intersection / union


def compute_dice(pred, gt, threshold=0.5):
    """Compute Dice Score (F1 for binary segmentation)"""
    pred_binary = (pred > threshold).astype(np.float32)
    gt_binary = (gt > 0).astype(np.float32)
    
    intersection = 2.0 * np.sum(pred_binary * gt_binary)
    total = np.sum(pred_binary) + np.sum(gt_binary)
    
    if total == 0:
        return 1.0 if np.sum(pred_binary) == 0 and np.sum(gt_binary) == 0 else 0.0
    
    return intersection / total


def load_ground_truth(image_id, dataset_path, coco_ann_file):
    """Load ground truth mask from COCO annotation"""
    try:
        coco = COCO(coco_ann_file)
        img_info = coco.loadImgs(image_id)
        if not img_info:
            return None
        
        ann_ids = coco.getAnnIds(imgIds=image_id)
        anns = coco.loadAnns(ann_ids)
        
        if not anns:
            return None
        
        # Create binary mask from all annotations
        img_data = img_info[0]
        mask = np.zeros((img_data['height'], img_data['width']), dtype=np.uint8)
        
        for ann in anns:
            rle = ann['segmentation']
            m = coco.annToMask(ann)
            mask = np.maximum(mask, m)
        
        return mask * 255
    except Exception as e:
        print(f"Error loading GT for image {image_id}: {e}")
        return None


def evaluate_dataset(dataset_type="crack", split="valid"):
    """Evaluate on a specific dataset and split"""
    
    print(f"\n{'='*70}")
    print(f"Evaluating {dataset_type.upper()} - {split.upper()} Set")
    print(f"{'='*70}")
    
    # Determine dataset paths
    if dataset_type == "crack":
        dataset_root = "../data/cracks.v1-cracks-f.coco"
        prompts = ["segment crack", "segment wall crack"]
    else:
        dataset_root = "../data/Drywall-Join-Detect.v2i.coco"
        prompts = ["segment taping area", "segment joint", "segment drywall seam"]
    
    coco_ann_file = f"{dataset_root}/{split}/_annotations.coco.json"
    image_dir = f"{dataset_root}/{split}"
    
    if not os.path.exists(coco_ann_file):
        print(f"⚠️  Annotation file not found: {coco_ann_file}")
        return None
    
    # Load COCO dataset
    coco = COCO(coco_ann_file)
    img_ids = coco.getImgIds()
    
    results = {
        "dataset": dataset_type,
        "split": split,
        "total_images": len(img_ids),
        "prompts": {},
        "timestamp": None
    }
    
    # Test each prompt
    for prompt in prompts:
        print(f"\n  Testing prompt: {prompt}")
        
        ious = []
        dices = []
        visual_samples = []
        
        for idx, img_id in enumerate(tqdm(img_ids[:50], desc=prompt)):  # Limit to first 50 for speed
            try:
                img_info = coco.loadImgs(img_id)[0]
                image_path = os.path.join(image_dir, img_info['file_name'])
                
                if not os.path.exists(image_path):
                    continue
                
                # Run inference
                img_orig, pred_mask = predict(image_path, prompt, MODEL_PATH)
                pred_array = np.array(pred_mask)
                
                # Load ground truth
                gt_mask = load_ground_truth(img_id, image_dir, coco_ann_file)
                if gt_mask is None:
                    continue
                
                # Resize GT to match prediction
                gt_mask = Image.fromarray(gt_mask).resize(
                    (pred_array.shape[1], pred_array.shape[0]),
                    Image.Resampling.BILINEAR
                )
                gt_array = np.array(gt_mask)
                
                # Compute metrics
                iou = compute_iou(pred_array, gt_array)
                dice = compute_dice(pred_array, gt_array)
                
                ious.append(iou)
                dices.append(dice)
                
                # Store visual samples (first 2 successful predictions)
                if len(visual_samples) < 2:
                    visual_samples.append({
                        "image_id": img_id,
                        "image_path": image_path,
                        "original": img_orig,
                        "gt": Image.fromarray(gt_array),
                        "pred": Image.fromarray(pred_array),
                        "iou": iou,
                        "dice": dice
                    })
            
            except Exception as e:
                continue
        
        # Compute statistics
        if ious:
            miou = np.mean(ious)
            dice_mean = np.mean(dices)
            
            results["prompts"][prompt] = {
                "mIoU": float(miou),
                "Dice": float(dice_mean),
                "samples_evaluated": len(ious),
                "visual_samples": visual_samples
            }
            
            print(f"    ✅ mIoU: {miou:.4f}")
            print(f"    ✅ Dice: {dice_mean:.4f}")
            print(f"    ✅ Samples: {len(ious)}")
    
    return results


def create_visual_comparison(results):
    """Create side-by-side comparison images"""
    
    print(f"\n{'='*70}")
    print("Creating Visual Examples")
    print(f"{'='*70}")
    
    for dataset_type in ["crack", "crack"]:  # Only evaluate what we have
        dataset_results = evaluate_dataset(dataset_type)
        if not dataset_results:
            continue
        
        for prompt, metrics in dataset_results["prompts"].items():
            for idx, sample in enumerate(metrics.get("visual_samples", [])):
                # Create comparison image
                orig = sample["original"]
                gt = sample["gt"]
                pred = sample["pred"]
                
                # Resize all to same size
                w, h = orig.size
                gt = gt.resize((w, h), Image.Resampling.NEAREST)
                pred = pred.resize((w, h), Image.Resampling.NEAREST)
                
                # Create 3-panel comparison
                comparison = Image.new('RGB', (w*3 + 20, h + 40), color='white')
                
                # Convert grayscale to RGB for display
                orig_rgb = orig.convert('RGB')
                gt_rgb = Image.new('RGB', gt.size, color='white')
                gt_rgb.paste(gt)
                pred_rgb = Image.new('RGB', pred.size, color='white')
                pred_rgb.paste(pred)
                
                comparison.paste(orig_rgb, (0, 40))
                comparison.paste(gt_rgb, (w + 10, 40))
                comparison.paste(pred_rgb, (w*2 + 20, 40))
                
                # Add labels
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(comparison)
                
                labels = [
                    f"Original (ID: {sample['image_id']})",
                    f"Ground Truth",
                    f"Prediction (IoU: {sample['iou']:.3f})"
                ]
                
                for i, label in enumerate(labels):
                    x = i * (w + 10) + 5
                    draw.text((x, 5), label, fill='black')
                
                # Save comparison
                prompt_slug = prompt.replace(" ", "_").lower()
                save_path = f"{RESULTS_DIR}/visuals/{dataset_type}__{prompt_slug}__{idx}.png"
                comparison.save(save_path)
                print(f"  Saved: {save_path}")


def generate_report():
    """Generate complete evaluation report"""
    
    print("\n" + "="*70)
    print("STARTING COMPREHENSIVE EVALUATION")
    print("="*70)
    
    # Evaluate both datasets
    crack_results = evaluate_dataset("crack", "valid")
    drywall_results = evaluate_dataset("drywall", "valid")
    
    # Generate report
    report = {
        "title": "Text-Conditioned Segmentation Evaluation Report",
        "model": "ResNet18 Encoder + UNet Decoder",
        "device": str(DEVICE),
        "datasets": {
            "crack": crack_results,
            "drywall": drywall_results
        }
    }
    
    # Save JSON report
    report_json = f"{RESULTS_DIR}/evaluation_metrics.json"
    with open(report_json, 'w') as f:
        # Convert non-serializable objects
        clean_report = {
            "title": report["title"],
            "model": report["model"],
            "device": report["device"],
            "datasets": {}
        }
        
        for ds_type, ds_data in report["datasets"].items():
            if ds_data:
                clean_report["datasets"][ds_type] = {
                    "dataset": ds_data["dataset"],
                    "split": ds_data["split"],
                    "total_images": ds_data["total_images"],
                    "prompts": {
                        prompt: {
                            "mIoU": metrics["mIoU"],
                            "Dice": metrics["Dice"],
                            "samples_evaluated": metrics["samples_evaluated"]
                        }
                        for prompt, metrics in ds_data["prompts"].items()
                    }
                }
        
        json.dump(clean_report, f, indent=2)
    
    print(f"\n✅ Report saved to: {report_json}")
    
    # Create visuals
    create_visual_comparison(report)
    
    return report


if __name__ == "__main__":
    report = generate_report()
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {RESULTS_DIR}/")

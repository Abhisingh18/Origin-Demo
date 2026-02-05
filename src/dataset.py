import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class COCOSegmentationDataset(Dataset):
    def __init__(self, root, split="train", img_size=256):
        self.root = root
        self.split = split
        self.img_size = img_size

        ann_path = os.path.join(root, split, "_annotations.coco.json")
        self.img_dir = os.path.join(root, split)

        self.coco = COCO(ann_path)
        self.img_ids = sorted(self.coco.getImgIds())
        
        print(f"{split}: {len(self.img_ids)} images found in COCO annotations")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        # 1. Load Image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for missing images (should catch in init, but safety here)
            print(f"Warning: Image not found {img_path}")
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Generate Mask from Annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        for ann in anns:
            try:
                # Some Roboflow exports have empty segmentation lists
                mask = np.maximum(mask, self.coco.annToMask(ann))
            except Exception as e:
                # print(f"Skipping bad annotation in image {img_id}: {e}")
                pass
            
        # 3. Resize both
        # Note: cv2.resize expects (width, height)
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        
        # 4. Normalize & Tensor
        # Standard ImageNet normalization matching inference
        # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Mask to float tensor [0, 1]
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        
        return image, mask

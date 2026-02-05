import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

class ResNetSegmentation(nn.Module):
    def __init__(self):
        super().__init__()
        # Load Pretrained DeepLabV3 with ResNet50 backbone
        # weights="DEFAULT" loads the best available weights (COCO)
        self.model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        
        # Replace the classifier head for binary segmentation (1 class + background? No, usually binary is 1 channel)
        # DeepLabV3 classifier is DeepLabHead sequentially:
        # 0: ASPP...
        # 1: Conv2d(256, 256, 3, padding=1)
        # 2: BatchNorm
        # 3: ReLU
        # 4: Conv2d(256, num_classes, 1)
        
        # We need final output channel = 1 (Binary mask)
        self.model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
        
        # Also auxiliary classifier needs to be changed if present (it is by default)
        self.model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        # DeepLabV3 returns an OrderedDict with keys 'out' and 'aux'
        return self.model(x)['out']

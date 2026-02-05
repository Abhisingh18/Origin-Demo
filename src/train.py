import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from dataset import COCOSegmentationDataset
from model import ResNetSegmentation
# from tqdm import tqdm
import os


# ---------------- CONFIG ----------------
DATA_ROOT = "data"
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-4
# RTX 5070 (sm_120) not supported by PyTorch yet - using CPU with optimizations
DEVICE = "cpu"
torch.set_num_threads(8)  # Optimize CPU performance

print("Device:", DEVICE)
print("⚠️  RTX 5070 (sm_120) requires PyTorch nightly build or future release")


# ---------------- DATA ----------------
crack_root = os.path.join(DATA_ROOT, "cracks.v1-cracks-f.coco")
drywall_root = os.path.join(DATA_ROOT, "Drywall-Join-Detect.v2i.coco")

train_full_dataset = ConcatDataset([
    COCOSegmentationDataset(crack_root, "train"),
    COCOSegmentationDataset(drywall_root, "train"),
])

# FAST TRAINING: Limit to 100 samples for quick feedback loop
# We select a random subset (or just first 500)
# subset_indices = torch.randperm(len(train_full_dataset))[:100]
# train_dataset = torch.utils.data.Subset(train_full_dataset, subset_indices)
train_dataset = train_full_dataset

val_dataset = ConcatDataset([
    COCOSegmentationDataset(crack_root, "valid"),
    COCOSegmentationDataset(drywall_root, "valid"),
])

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)

print("Train samples (Limited):", len(train_dataset))
print("Val samples:", len(val_dataset))


# ---------------- MODEL ----------------
model = ResNetSegmentation().to(DEVICE)
# DeepLabV3 outputs logits (no sigmoid), so we use BCEWithLogitsLoss
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


# ---------------- TRAIN LOOP ----------------
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    print(f"Epoch {epoch+1}/{EPOCHS}...")
    for step, (images, masks) in enumerate(train_loader):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        preds = model(images)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if step % 10 == 0:
            print(f"  Step {step}/{len(train_loader)} Loss: {loss.item():.4f}")

    avg_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}")
    
    # Save after every epoch so we can test early
    MODEL_PATH = "best_model.pth"
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Model saved to: {MODEL_PATH}")

print(f"✅ Training completed!")

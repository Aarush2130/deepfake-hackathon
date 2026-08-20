"""
VeriChain Forensic OS — Deepfake Model Training Pipeline
Fine-tunes EfficientNet-B0 on Large-Scale (140k+) Real vs. Fake Datasets.

Features:
- Automatic Train / Validation Split or Pre-split directory support
- PyTorch Mixed Precision (AMP FP16) for fast GPU throughput
- Deepfake-Specific Data Augmentation (JPEG compression simulation, color jitter, flip)
- Cosine Annealing LR Scheduler with Warmup
- Auto-saves the best checkpoint as 'best_model-v4.pt'
"""

import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

def get_transforms(img_size=224):
    """Deepfake-specific training and validation augmentations."""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform

def build_model(pretrained=True):
    """Builds EfficientNet-B0 with custom 2-class classifier."""
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 2)
    )
    return model

def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in dataloader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda" if "cuda" in device.type else "cpu"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda" if "cuda" in device.type else "cpu"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc = (correct / total) * 100.0
    return val_loss, val_acc

def main():
    parser = argparse.ArgumentParser(description="Train Deepfake Classifier on 140k Dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root folder (containing 'fake' and 'real' subfolders)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (Default: 10)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (Default: 64, use 32 if low VRAM)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate (Default: 3e-4)")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation split ratio if data is not pre-split (Default: 0.15)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker processes (Default: 4)")
    parser.add_argument("--output_path", type=str, default="best_model-v4.pt", help="Path to save best PyTorch model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    train_tf, val_tf = get_transforms(img_size=224)

    # Check if dataset has train/val subfolders or single root folder
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val") if os.path.exists(os.path.join(args.data_dir, "val")) else os.path.join(args.data_dir, "test")

    if os.path.exists(train_dir) and os.path.exists(val_dir):
        print(f"[*] Detected pre-split dataset: Train='{train_dir}', Val='{val_dir}'")
        train_dataset = datasets.ImageFolder(train_dir, transform=train_tf)
        val_dataset = datasets.ImageFolder(val_dir, transform=val_tf)
    else:
        print(f"[*] Loading unified dataset from '{args.data_dir}' with {args.val_split*100:.0f}% validation split...")
        full_dataset = datasets.ImageFolder(args.data_dir, transform=train_tf)
        val_size = int(len(full_dataset) * args.val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"[*] Total Samples: {len(train_dataset)} Train | {len(val_dataset)} Validation")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    model = build_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda" if "cuda" in device.type else "cpu")

    best_val_acc = 0.0
    print("\n" + "="*60)
    print("🚀 Starting Training Loop")
    print("="*60)

    for epoch in range(1, args.epochs + 1):
        start_t = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - start_t
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s, LR: {current_lr:.2e}) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output_path)
            print(f"  --> 🌟 New Best Model Saved to '{args.output_path}' (Val Acc: {val_acc:.2f}%)")

    print("\n" + "="*60)
    print(f"✅ Training Complete! Peak Validation Accuracy: {best_val_acc:.2f}%")
    print(f"📁 Checkpoint saved as: '{args.output_path}'")
    print("="*60)

if __name__ == "__main__":
    main()

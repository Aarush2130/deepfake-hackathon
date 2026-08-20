"""
VeriChain Forensic OS — Deepfake Model Training Pipeline
Fine-tunes EfficientNet-B0 on Large-Scale (140k+) Real vs. Fake Datasets.

Features:
- Automatic Train / Validation Split or Pre-split directory support
- PyTorch Mixed Precision (AMP FP16) for high GPU throughput
- Deepfake-Specific Data Augmentation (color jitter, flip, rotation, blur)
- Cosine Annealing LR Scheduler with AdamW optimizer
- Auto-saves the best checkpoint as 'best_model-v4.pt'
- Auto-generates 'model_metadata.json' with verified class index mapping
"""

import os
import sys
import json
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

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
    """Builds EfficientNet-B0 with custom 2-class classifier matching engine.py."""
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    return model

def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch=1, total_epochs=1):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    total_batches = len(dataloader)
    start_time = time.time()

    for batch_idx, (images, labels) in enumerate(dataloader, 1):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda" if "cuda" in device.type else "cpu"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_sz = images.size(0)
        running_loss += loss.item() * batch_sz
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += batch_sz

        if batch_idx % 100 == 0 or batch_idx == total_batches:
            cur_loss = running_loss / total
            cur_acc = (correct / total) * 100.0
            elapsed = time.time() - start_time
            img_per_sec = total / max(1e-5, elapsed)
            print(f"  [Epoch {epoch:02d}/{total_epochs:02d}] Batch {batch_idx:04d}/{total_batches:04d} | "
                  f"Loss: {cur_loss:.4f} | Acc: {cur_acc:.2f}% | Speed: {img_per_sec:.0f} img/s")

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = (correct / total) * 100.0 if total > 0 else 0.0
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

    val_loss = running_loss / total if total > 0 else 0.0
    val_acc = (correct / total) * 100.0 if total > 0 else 0.0
    return val_loss, val_acc

def main():
    parser = argparse.ArgumentParser(description="Train Deepfake Classifier on 140k Dataset from Scratch / Clean Slate")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root folder (containing 'real' and 'fake' subfolders)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (Default: 10)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (Default: 64, use 32 if VRAM < 6GB)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate (Default: 3e-4)")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation split ratio if data is not pre-split (Default: 0.15)")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker processes (Default: 0 for Windows safety)")
    parser.add_argument("--output_path", type=str, default="best_model-v4.pt", help="Path to save best PyTorch model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*70)
    print(f"🚀 VeriChain Model Training Pipeline (Clean Slate)")
    print(f"[*] Training on Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("="*70)

    train_tf, val_tf = get_transforms(img_size=224)

    # Smart case-insensitive discovery of train and validation subfolders
    train_dir = None
    val_dir = None
    if os.path.exists(args.data_dir):
        for d in os.listdir(args.data_dir):
            dp = os.path.join(args.data_dir, d)
            if os.path.isdir(dp):
                d_lower = d.lower()
                if d_lower in ["train", "training"] and train_dir is None:
                    train_dir = dp
                elif d_lower in ["val", "validation", "test", "testing"] and val_dir is None:
                    val_dir = dp

    if train_dir and val_dir:
        print(f"[*] Detected pre-split dataset:")
        print(f"    - Train Folder:      '{train_dir}'")
        print(f"    - Validation Folder: '{val_dir}'")
        raw_train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
        raw_val_ds = datasets.ImageFolder(val_dir, transform=val_tf)
        class_to_idx = raw_train_ds.class_to_idx
        classes = raw_train_ds.classes
        train_dataset = raw_train_ds
        val_dataset = raw_val_ds
    else:
        print(f"[*] Loading unified dataset from '{args.data_dir}' with {args.val_split*100:.0f}% validation split...")
        full_dataset = datasets.ImageFolder(args.data_dir, transform=train_tf)
        class_to_idx = full_dataset.class_to_idx
        classes = full_dataset.classes
        val_size = int(len(full_dataset) * args.val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"\n📊 Class Mapping Discovered:")
    for cls_name, idx in class_to_idx.items():
        print(f"   - Class '{cls_name}' --> Index {idx}")

    # Determine fake class index
    fake_idx = 1
    for cls_name, idx in class_to_idx.items():
        if any(term in cls_name.lower() for term in ["fake", "synth", "manipulated", "deepfake"]):
            fake_idx = idx
            break

    print(f"[*] Identified Fake Class Index: {fake_idx} ('{classes[fake_idx]}')")
    print(f"[*] Total Samples: {len(train_dataset):,} Train | {len(val_dataset):,} Validation\n")

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
    print("="*70)
    print("⚡ Starting Epoch Iterations")
    print("="*70)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    for epoch in range(1, args.epochs + 1):
        start_t = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch=epoch, total_epochs=args.epochs)
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
            
            # Save metadata
            meta = {
                "model_file": args.output_path,
                "architecture": "efficientnet_b0",
                "classes": classes,
                "class_to_idx": class_to_idx,
                "fake_index": fake_idx,
                "real_index": 1 - fake_idx if len(classes) == 2 else 0,
                "best_val_accuracy": float(val_acc),
                "epochs_trained": epoch,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            }
            with open("model_metadata.json", "w") as mf:
                json.dump(meta, mf, indent=2)

            print(f"  --> 🌟 New Best Model Saved to '{args.output_path}' (Val Acc: {val_acc:.2f}%)")

    print("\n" + "="*70)
    print(f"✅ Training Complete! Peak Validation Accuracy: {best_val_acc:.2f}%")
    print(f"📁 Checkpoint saved as: '{args.output_path}'")
    print(f"📄 Metadata written to: 'model_metadata.json'")
    print("="*70)

if __name__ == "__main__":
    main()

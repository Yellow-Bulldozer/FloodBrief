"""
FloodBrief - Training script.

Fine-tunes a TerraMind-small encoder + UPerNet head for binary flood
segmentation on Sen1Floods11.

Real-data training:
    python train.py --data-dir ./data/sen1floods11_v1.1 --epochs 50 --batch-size 8

With synthetic data (no real dataset needed):
    python train.py --synthetic --epochs 10

Resume training:
    python train.py --data-dir ./data/sen1floods11_v1.1 --resume ./checkpoints/best_model.pt
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import FloodBriefModel
from src.data_loader import get_dataloaders, IGNORE_INDEX
from src.metrics import SegmentationMetrics


# ---------------------------------------------------------------------------
# Dice Loss (recommended by IBM for binary segmentation)
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """Dice loss for binary/multiclass segmentation."""

    def __init__(self, smooth: float = 1.0, ignore_index: int = -1):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # Create mask for valid pixels
        valid_mask = (targets != self.ignore_index)
        targets_valid = targets.clone()
        targets_valid[~valid_mask] = 0  # temporary, will be masked

        # One-hot encode targets
        targets_onehot = F.one_hot(targets_valid, num_classes).permute(0, 3, 1, 2).float()

        # Apply valid mask
        valid_mask = valid_mask.unsqueeze(1).float()  # (B, 1, H, W)
        probs = probs * valid_mask
        targets_onehot = targets_onehot * valid_mask

        # Compute dice per class
        dims = (0, 2, 3)
        intersection = (probs * targets_onehot).sum(dims)
        cardinality = probs.sum(dims) + targets_onehot.sum(dims)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_score.mean()


class CombinedLoss(nn.Module):
    """Weighted combination of Dice + Cross-Entropy loss."""

    def __init__(
        self,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        class_weights: list = None,
        ignore_index: int = -1,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

        self.dice = DiceLoss(ignore_index=ignore_index)

        ce_weights = torch.tensor(class_weights) if class_weights else None
        self.ce = nn.CrossEntropyLoss(
            weight=ce_weights,
            ignore_index=ignore_index,
        )

    def forward(self, logits, targets):
        return (self.dice_weight * self.dice(logits, targets) +
                self.ce_weight * self.ce(logits, targets))


def parse_args():
    parser = argparse.ArgumentParser(description="FloodBrief training")
    # Data
    parser.add_argument("--data-dir", type=str, default="./data/sen1floods11_v1.1",
                        help="Sen1Floods11 v1.1 data directory")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (for testing)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit dataset size (for debugging)")
    parser.add_argument("--normalize", type=str, default="terramind",
                        choices=["terramind", "minmax"],
                        help="Normalization method")
    # Model
    parser.add_argument("--model-name", type=str, default="terramind_v1_small",
                        help="TerraMind backbone name")
    parser.add_argument("--freeze-encoder", action="store_true", default=False,
                        help="Freeze TerraMind encoder (faster but lower accuracy)")
    parser.add_argument("--no-freeze-encoder", dest="freeze_encoder",
                        action="store_false")
    # Training
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate (2e-5 recommended for fine-tuning)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--loss", type=str, default="combined",
                        choices=["dice", "ce", "combined"],
                        help="Loss function")
    parser.add_argument("--img-size", type=int, default=224, help="Input image size")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers")
    # Mixed precision
    parser.add_argument("--amp", action="store_true", default=False,
                        help="Enable automatic mixed precision")
    # Early stopping
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (0=disabled)")
    # Checkpointing
    parser.add_argument("--output-dir", type=str, default="./checkpoints",
                        help="Output directory for checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from checkpoint")
    # Misc
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str: str) -> torch.device:
    """Determine the best available device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def build_loss(args, device):
    """Build loss function based on args."""
    class_weights = [0.3, 0.7]  # upweight flood class

    if args.loss == "dice":
        return DiceLoss(ignore_index=IGNORE_INDEX)
    elif args.loss == "ce":
        return nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights).to(device),
            ignore_index=IGNORE_INDEX,
        )
    else:  # combined
        return CombinedLoss(
            dice_weight=0.5, ce_weight=0.5,
            class_weights=class_weights,
            ignore_index=IGNORE_INDEX,
        ).to(device)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch,
                    scaler=None):
    """Train for one epoch with optional mixed precision."""
    model.train()
    total_loss = 0.0
    metrics = SegmentationMetrics(num_classes=2)
    use_amp = scaler is not None

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast("cuda"):
                output = model(images)
                loss = criterion(output["logits"], labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(images)
            loss = criterion(output["logits"], labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)

        # Track metrics (skip ignored pixels)
        pred = output["logits"].argmax(dim=1).cpu().numpy()
        target_np = labels.cpu().numpy()
        # Mask out ignored pixels for metric computation
        valid = target_np != IGNORE_INDEX
        if valid.any():
            metrics.update(pred[valid], target_np[valid])

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / max(len(loader.dataset), 1)
    epoch_metrics = metrics.compute()
    epoch_metrics["loss"] = round(avg_loss, 6)
    return epoch_metrics


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    metrics = SegmentationMetrics(num_classes=2)

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        output = model(images)
        loss = criterion(output["logits"], labels)

        total_loss += loss.item() * images.size(0)
        pred = output["logits"].argmax(dim=1).cpu().numpy()
        target_np = labels.cpu().numpy()
        valid = target_np != IGNORE_INDEX
        if valid.any():
            metrics.update(pred[valid], target_np[valid])

    avg_loss = total_loss / max(len(loader.dataset), 1)
    epoch_metrics = metrics.compute()
    epoch_metrics["loss"] = round(avg_loss, 6)
    return epoch_metrics


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    # Use freeze_encoder=True for synthetic (demo), False for real data
    if args.synthetic and not args.freeze_encoder:
        args.freeze_encoder = True

    print(f"\n{'='*60}")
    print(f"  FloodBrief Training")
    print(f"  Device: {device}")
    print(f"  Data: {'Synthetic' if args.synthetic else args.data_dir}")
    print(f"  Loss: {args.loss} | LR: {args.lr} | Epochs: {args.epochs}")
    print(f"  Encoder freeze: {args.freeze_encoder}")
    print(f"  Mixed precision: {args.amp}")
    print(f"{'='*60}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Data ---
    train_loader, val_loader, _ = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        use_synthetic=args.synthetic,
        max_samples=args.max_samples,
        normalize=args.normalize,
    )

    # --- Model ---
    model = FloodBriefModel(
        model_name=args.model_name,
        pretrained=True,
        num_classes=2,
        freeze_encoder=args.freeze_encoder,
        img_size=args.img_size,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable / 1e6:.2f}M / {total / 1e6:.2f}M total\n")

    # --- Loss ---
    criterion = build_loss(args, device)

    # --- Optimizer ---
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # --- Scheduler ---
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5,
    )

    # --- Mixed precision ---
    scaler = None
    if args.amp and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
        print("[FloodBrief] Mixed precision training enabled.\n")

    # --- Resume ---
    start_epoch = 1
    best_miou = 0.0
    history = {"train": [], "val": [], "config": vars(args)}

    if args.resume:
        print(f"[FloodBrief] Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        if "val_metrics" in ckpt:
            best_miou = ckpt["val_metrics"].get("mIoU", 0.0)
        model = model.to(device)
        print(f"[FloodBrief] Resumed at epoch {start_epoch}, best mIoU: {best_miou:.4f}\n")

    # --- Training loop ---
    epochs_without_improvement = 0

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            scaler=scaler,
        )
        history["train"].append(train_metrics)

        val_metrics = validate(model, val_loader, criterion, device)
        history["val"].append(val_metrics)

        # Step scheduler
        scheduler.step(val_metrics["mIoU"])

        # Print summary
        print(
            f"[Epoch {epoch:3d}/{args.epochs}]  "
            f"Train Loss: {train_metrics['loss']:.4f}  "
            f"Train mIoU: {train_metrics['mIoU']:.4f}  |  "
            f"Val Loss: {val_metrics['loss']:.4f}  "
            f"Val mIoU: {val_metrics['mIoU']:.4f}  "
            f"Val F1: {val_metrics['f1_flood']:.4f}"
        )

        # Save best model
        if val_metrics["mIoU"] > best_miou:
            best_miou = val_metrics["mIoU"]
            epochs_without_improvement = 0
            checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "args": vars(args),
            }, checkpoint_path)
            print(f"  -> Best model saved (mIoU={best_miou:.4f})")
        else:
            epochs_without_improvement += 1

        # Early stopping
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"\n[FloodBrief] Early stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs)")
            break

    # Save final model
    final_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_metrics": val_metrics,
        "args": vars(args),
    }, final_path)

    # Save training history
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best validation mIoU: {best_miou:.4f}")
    print(f"  Checkpoints: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

# %% [markdown]
# # train.py — DeepLense GSoC 2026 Central Training Script
# Unified CLI trainer. Run from terminal or from Notebook 06 smoke test.
# Supports: baseline | transfer | vit | equivariant

# %% [Cell 1: Module Docstring & All Imports]
"""
train.py — DeepLense GSoC 2026 Central Training Script
=======================================================
Trains one of four models (baseline / transfer / vit / equivariant) with:
    • Per-epoch validation loss + accuracy tracking  (overfitting visibility)
    • Model-specific optimizer + LR scheduler        (correct training recipe)
    • Gradient clipping for ViT                      (prevents early divergence)
    • Best-model checkpointing on val accuracy       (saves peak, not last)
    • Full ROC-AUC + F1 report using metrics.py      (ML4SCI primary metrics)
    • Learning curve plots saved to assets/          (diagnostic evidence)
    • Deterministic seeding incl. DataLoader workers (reproducible science)

Usage (from notebook 06 or terminal):
    python src/train.py --model_name transfer \
                        --data_dir "."         \
                        --csv_path "metadata.csv" \
                        --zip_path "/content/drive/My Drive/.../dataset.zip" \
                        --epochs 10 \
                        --augment          (or --no-augment to disable)

Supported --model_name values:
    baseline     ResNet-18 from scratch,  1-ch grayscale, 64×64
    transfer     ResNet-18 + ImageNet,    3-ch RGB,       224×224
    vit          ViT-B/16  + ImageNet,    3-ch RGB,       224×224
    equivariant  C8-Equivariant CNN,      1-ch grayscale, 224×224  (Phase-2)
"""

import os
import random
import argparse
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

# FIX v2: wandb is optional — import failure does NOT crash the training script.
# Set WANDB_AVAILABLE=True only if wandb is installed and reachable.
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not installed — run tracking disabled. pip install wandb to enable.")

from dataset  import get_dataloaders, stage_data_locally
from models   import ResNetBaseline, ResNetTransfer, ViTChampion, EquivariantCNN
from metrics  import (
    save_confusion_matrix,
    generate_classification_report,
    plot_multiclass_roc_auc,
    plot_learning_curves,
    plot_calibration_curves,
)

# %% [Cell 2: Reproducibility Helpers]
# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# %% [Cell 3: Argument Parser]
# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepLense GSoC 2026 — Modular Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--model_name', type=str, default='baseline',
        choices=['baseline', 'transfer', 'vit', 'equivariant'],
        help='Architecture to train.',
    )

    parser.add_argument('--data_dir',  type=str, default='.',
                        help='Path to the image root directory.')
    parser.add_argument('--csv_path',  type=str, default='metadata.csv',
                        help='Path to metadata.csv.')
    parser.add_argument('--zip_path',  type=str, default=None,
                        help='(Optional) Path to zipped dataset on Drive for auto-staging.')

    parser.add_argument('--epochs',     type=int,   default=10)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=None,
                        help='Learning rate. If None, uses model-specific default.')

    parser.add_argument(
        '--augment',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Apply physics-motivated augmentation to training set.',
    )

    parser.add_argument(
        '--grad_clip', type=float, default=1.0,
        help='Max gradient norm for clipping. 0.0 disables clipping.',
    )

    parser.add_argument(
        '--scheduler', type=str, default='cosine',
        choices=['cosine', 'step', 'none'],
        help='LR scheduler. "cosine" = CosineAnnealingLR. "step" = StepLR. "none" = constant LR.',
    )

    parser.add_argument('--save_dir',   type=str, default='weights',
                        help='Directory to save model weights (.pth files).')
    parser.add_argument('--assets_dir', type=str, default='assets',
                        help='Directory to save plots (curves, confusion matrix, ROC).')

    # [GSOC UPGRADE 4] WandB project name flag
    parser.add_argument('--wandb_project', type=str, default='DeepLense_GSoC_2026',
                        help='Weights & Biases project name.')

    return parser.parse_args()

# %% [Cell 4: Model, Optimizer & Scheduler Factory]
# ─────────────────────────────────────────────────────────────────────────────
# MODEL + OPTIMIZER FACTORY
# ─────────────────────────────────────────────────────────────────────────────

MODEL_DEFAULTS = {
    'baseline':    {'mode': 'L',   'image_size': 64,  'lr': 1e-3,  'optimizer': 'adam',  'weight_decay': 0.0},
    'transfer':    {'mode': 'RGB', 'image_size': 224, 'lr': 1e-4,  'optimizer': 'adam',  'weight_decay': 0.0},
    'vit':         {'mode': 'RGB', 'image_size': 224, 'lr': 5e-5,  'optimizer': 'adamw', 'weight_decay': 0.01},
    'equivariant': {'mode': 'L',   'image_size': 128, 'lr': 1e-4,  'optimizer': 'adam',  'weight_decay': 1e-4},  # Changed from 224
}

GRAD_CLIP_DEFAULTS = {
    'baseline':    0.0,
    'transfer':    0.0,
    'vit':         1.0,
    'equivariant': 0.0,
}

def build_model(model_name: str) -> nn.Module:
    if model_name == 'baseline':
        return ResNetBaseline(num_classes=3)
    elif model_name == 'transfer':
        return ResNetTransfer(num_classes=3)
    elif model_name == 'vit':
        return ViTChampion(num_classes=3)
    elif model_name == 'equivariant':
        return EquivariantCNN(num_classes=3, n_rotations=8)   # C8 group
    else:
        raise ValueError(f"Unknown model_name: '{model_name}'")

def build_optimizer(model: nn.Module, model_name: str, lr: float) -> optim.Optimizer:
    cfg          = MODEL_DEFAULTS[model_name]
    opt_type     = cfg['optimizer']
    weight_decay = cfg['weight_decay']

    if opt_type == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def build_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str,
    epochs: int,
) -> Optional[optim.lr_scheduler.LRScheduler]:
    if scheduler_name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-7
        )
    elif scheduler_name == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=5, gamma=0.5
        )
    else:
        return None

# %% [Cell 5: train_one_epoch Function]
# ─────────────────────────────────────────────────────────────────────────────
# SINGLE EPOCH — TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:        nn.Module,
    loader:       torch.utils.data.DataLoader,
    criterion:    nn.Module,
    optimizer:    optim.Optimizer,
    device:       torch.device,
    epoch:        int,
    total_epochs: int,
    scaler:       torch.cuda.amp.GradScaler,   # [GSOC UPGRADE 2] Pass scaler
    grad_clip:    float = 0.0,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct      = 0
    total        = 0

    progress_bar = tqdm(
        loader,
        desc=f"  Train Epoch {epoch+1}/{total_epochs}",
        leave=False,
    )

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # FIX v2: Updated to torch.amp.autocast (torch.cuda.amp.autocast deprecated in 2.x)
        with torch.amp.autocast(device_type='cuda', enabled=device.type == 'cuda'):
            logits = model(images)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()

        if grad_clip > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # ── [GSOC UPGRADE 3] MIXUP ACCURACY FIX ──────────────────────────
        running_loss += loss.item()
        _, predicted  = torch.max(logits, dim=1)
        total        += labels.size(0)

        # Handle 2D soft labels from MixUp/CutMix
        if labels.dim() > 1:
            true_classes = torch.argmax(labels, dim=1)
        else:
            true_classes = labels

        correct += (predicted == true_classes).sum().item()

        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc':  f"{100 * correct / total:.1f}%",
        })

    avg_loss = running_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

# %% [Cell 6: validate_one_epoch Function]
# ─────────────────────────────────────────────────────────────────────────────
# SINGLE EPOCH — VALIDATE
# ─────────────────────────────────────────────────────────────────────────────

def validate_one_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> Tuple[float, float, List, List, np.ndarray]:
    model.eval()
    running_loss = 0.0
    correct      = 0
    total        = 0
    all_preds    = []
    all_labels   = []
    all_probs    = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss   = criterion(logits, labels)

            probs = F.softmax(logits, dim=1)

            running_loss += loss.item()
            _, predicted  = torch.max(logits, dim=1)
            total        += labels.size(0)
            correct      += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss  = running_loss / len(loader)
    accuracy  = 100.0 * correct / total
    all_probs = np.array(all_probs)

    return avg_loss, accuracy, all_preds, all_labels, all_probs

# %% [Cell 7: main() — Full Training Pipeline]
# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    set_seed(42)
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── [GSOC UPGRADE 4] WANDB INITIALIZATION — guarded by availability check
    if WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.model_name}_run",
            config=vars(args)
        )

    grad_clip = args.grad_clip

    print("\n" + "="*60)
    print(f"  DeepLense GSoC 2026 — Training Pipeline")
    print(f"  Model      : {args.model_name.upper()}")
    print(f"  Device     : {device}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Scheduler  : {args.scheduler}")
    print(f"  Augment    : {args.augment}")
    print(f"  Grad Clip  : {grad_clip if grad_clip > 0 else 'disabled'}")
    print(f"  Recommended: {GRAD_CLIP_DEFAULTS[args.model_name]} "
          f"for {args.model_name}")
    print(f"  WandB      : {'enabled' if WANDB_AVAILABLE else 'disabled (pip install wandb)'}")
    print("="*60 + "\n")

    if args.zip_path:
        staged_dir = stage_data_locally(args.zip_path)
        if staged_dir:
            args.data_dir = staged_dir
            args.csv_path = os.path.join(staged_dir, os.path.basename(args.csv_path))

    cfg        = MODEL_DEFAULTS[args.model_name]
    mode       = cfg['mode']
    image_size = cfg['image_size']
    lr         = args.lr if args.lr is not None else cfg['lr']

    print(f"📐 Input config — mode: {mode} | size: {image_size}×{image_size} | lr: {lr}")

    g = torch.Generator()
    g.manual_seed(42)

    # ── [GSOC UPGRADE 1] DataLoader Unpacking & ViT Trigger ───────────────
    train_loader, val_loader, test_loader, _, _, _ = get_dataloaders(
        csv_path         = args.csv_path,
        base_dir         = args.data_dir,
        mode             = mode,
        image_size       = image_size,
        batch_size       = args.batch_size,
        augment          = args.augment,
        worker_init_fn   = _seed_worker,
        generator        = g,
        apply_mixup      = (args.model_name == 'vit')  # Trigger MixUp only for ViT
    )

    model     = build_model(args.model_name).to(device)
    optimizer = build_optimizer(model, args.model_name, lr)
    scheduler = build_scheduler(optimizer, args.scheduler, args.epochs)
    criterion = nn.CrossEntropyLoss()

    # FIX v2: Updated to torch.amp.GradScaler (torch.cuda.amp.GradScaler deprecated in 2.x)
    scaler = torch.amp.GradScaler(device='cuda', enabled=device.type == 'cuda')

    os.makedirs(args.save_dir,   exist_ok=True)
    os.makedirs(args.assets_dir, exist_ok=True)

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    best_val_acc = 0.0
    best_epoch   = 0
    best_weights = None
    best_path    = os.path.join(args.save_dir, f"{args.model_name}_best.pth")

    print(f"\n🚀 Starting training — {args.model_name.upper()} | {args.epochs} epochs\n")

    for epoch in range(args.epochs):

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, args.epochs,
            scaler=scaler,          # Pass the initialized scaler
            grad_clip=grad_clip,
        )

        val_loss, val_acc, val_preds, val_labels, val_probs = validate_one_epoch(
            model, val_loader, criterion, device
        )

        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # ── [GSOC UPGRADE 4] WandB Iteration Logging ──────────────────────
        if WANDB_AVAILABLE:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": current_lr
            })

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_epoch    = epoch + 1
            best_weights  = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_weights, best_path)

        print(
            f"  Epoch {epoch+1:>3}/{args.epochs} │ "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.1f}% │ "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.1f}% │ "
            f"LR: {current_lr:.2e}"
            + (" ← best" if epoch + 1 == best_epoch else "")
        )

    final_path = os.path.join(args.save_dir, f"{args.model_name}_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\n💾 Last-epoch weights → {final_path}")
    print(f"💾 Best checkpoint   → {best_path}  (epoch {best_epoch}, val_acc={best_val_acc:.1f}%)")

    print(f"\n🔄 Restoring best weights (epoch {best_epoch}) for final evaluation...")
    model.load_state_dict(best_weights)

    # ── [CRITICAL FIX] Evaluation strictly on test set ────────────────────
    _, _, final_preds, final_labels, final_probs = validate_one_epoch(
        model, test_loader, criterion, device
    )

    classes = ['No Sub', 'CDM', 'Vortex']

    report = generate_classification_report(
        final_labels, final_preds,
        classes    = classes,
        model_name = args.model_name.upper(),
    )

    cm_path = os.path.join(args.assets_dir, f"{args.model_name}_confusion_matrix.png")
    save_confusion_matrix(
        final_labels, final_preds,
        classes   = classes,
        save_path = cm_path,
        title     = f'{args.model_name.upper()} — Confusion Matrix (Test Set)',
    )

    roc_path = os.path.join(args.assets_dir, f"{args.model_name}_roc_auc.png")
    auc_scores = plot_multiclass_roc_auc(
        final_labels, final_probs,
        classes    = classes,
        save_path  = roc_path,
        model_name = args.model_name.upper(),
    )

    curves_path = os.path.join(args.assets_dir, f"{args.model_name}_learning_curves.png")
    plot_learning_curves(
        train_losses = train_losses,
        val_losses   = val_losses,
        train_accs   = train_accs,
        val_accs     = val_accs,
        save_path    = curves_path,
        model_name   = args.model_name.upper(),
    )

    # ── [FIX APPLIED] Generate Calibration Curves ───────────────────────
    calib_path = os.path.join(args.assets_dir, f"{args.model_name}_calibration.png")
    plot_calibration_curves(
        final_labels, final_probs,
        classes    = classes,
        save_path  = calib_path,
        model_name = args.model_name.upper(),
    )

    # ── [FIX APPLIED] WandB Final Summary Logging ───────────────────────
    if WANDB_AVAILABLE:
        wandb.log({
            "best_val_acc":  best_val_acc,
            "macro_auc":     auc_scores['macro'],
            "fpr_90_macro":  auc_scores.get('fpr_90_macro', 0.0),
            "macro_f1":      report['f1_macro'],
            "weighted_f1":   report['f1_weighted']
        })
        wandb.finish()

    print("\n" + "="*60)
    print(f"  TRAINING COMPLETE — {args.model_name.upper()}")
    print(f"  Best Val Accuracy : {best_val_acc:.2f}%  (epoch {best_epoch})")
    print(f"  Macro AUC         : {auc_scores['macro']:.4f}  ← ML4SCI primary")
    print(f"  FPR @ 90% TPR     : {auc_scores.get('fpr_90_macro', 0.0):.4f}  ← Physics threshold")
    print(f"  Macro F1          : {report['f1_macro']:.4f}")
    print(f"  Weighted F1       : {report['f1_weighted']:.4f}")
    print(f"  {'─'*38}")
    for i, cls in enumerate(classes):
        print(f"  {cls:<12} AUC : {auc_scores['per_class'][i]:.4f}  "
              f"F1 : {report['f1_per_class'][cls]:.4f}")
    print("="*60 + "\n")

# %% [Cell 8: Entry Point]
if __name__ == "__main__":
    main()
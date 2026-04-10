"""
evaluate_ensemble.py — DeepLense GSoC 2026 Ensemble Evaluation Script
======================================================================
Evaluates the DeepLenseEnsemble (ResNetTransfer + ViTChampion) in two modes:

    Mode 1 — Standard Evaluation:
        Passes the clean 224×224 test set through the ensemble once.
        Produces: classification report, confusion matrix, ROC-AUC curves, 
                  and Calibration Curves.

    Mode 2 — Rotational TTA Diagnostic:
        Evaluates the same ensemble across 4 rotations (0°/90°/180°/270°).
        Averages predictions across all rotations per sample.
        Produces: TTA confusion matrix, per-class F1 degradation plot.

        Scientific purpose: Quantifies the spatial orientation bias of standard
        CNN+ViT architectures. The accuracy drop under rotation is the core
        evidence motivating E(2)-Equivariant Neural Networks for this task.

BUG FIXES & UPGRADES:
    1. [UPGRADE] The Ensemble is now a Stacking Meta-Learner. This script 
       trains the fusion head on the Val set before evaluating on the Test set.
    2. [UPGRADE] Ensemble now returns logits. Softmax is safely applied here.
    3. [UPGRADE] Calibration Curves and Physics FPR thresholds added.
    4. TTA rotation uses torch.rot90() for exact 90° multiples (lossless).

Usage:
    python src/evaluate_ensemble.py \
        --resnet_weights weights/transfer_best.pth \
        --vit_weights    weights/vit_best.pth      \
        --data_dir       "."                        \
        --csv_path       "metadata.csv"             \
        --zip_path       "/content/drive/My Drive/.../dataset.zip"
"""

from __future__ import annotations   # Enables Python 3.8/3.9 compatible type hints

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm

from dataset import get_dataloaders, stage_data_locally
from models  import ResNetTransfer, ViTChampion, DeepLenseEnsemble, load_model
from metrics import (
    save_confusion_matrix,
    generate_classification_report,
    plot_multiclass_roc_auc,
    plot_tta_degradation,
    plot_calibration_curves, # [GSOC UPGRADE 4] Added new metric
)


# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepLense GSoC 2026 — Ensemble Evaluation + TTA Diagnostic",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--data_dir',  type=str, default=".",
                        help='Path to the image root directory.')
    parser.add_argument('--csv_path',  type=str, default="metadata.csv",
                        help='Path to metadata.csv.')
    parser.add_argument('--zip_path',  type=str, default=None,
                        help='(Optional) Path to zipped dataset on Drive.')

    parser.add_argument('--resnet_weights', type=str,
                        default="weights/transfer_best.pth",
                        help='Path to ResNetTransfer weights (.pth).')
    parser.add_argument('--vit_weights', type=str,
                        default="weights/vit_best.pth",
                        help='Path to ViTChampion weights (.pth).')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--skip_tta',  action='store_true', default=False,
                        help='If set, skips the TTA diagnostic.')
    parser.add_argument('--tta_angles', type=int, nargs='+',
                        default=[0, 90, 180, 270],
                        help='Rotation angles (degrees) for TTA diagnostic.')

    parser.add_argument('--assets_dir', type=str, default='assets',
                        help='Directory to save all output plots.')

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# ROTATION UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def rotate_batch(images: torch.Tensor, angle: int) -> torch.Tensor:
    angle_mod = angle % 360
    if angle_mod == 0:
        return images
    elif angle_mod == 90:
        return torch.rot90(images, k=1, dims=[2, 3])
    elif angle_mod == 180:
        return torch.rot90(images, k=2, dims=[2, 3])
    elif angle_mod == 270:
        return torch.rot90(images, k=3, dims=[2, 3])
    else:
        return TF.rotate(images, angle=angle,
                         interpolation=TF.InterpolationMode.NEAREST)


# ─────────────────────────────────────────────────────────────────────────────
# [GSOC UPGRADE 1] META-LEARNER TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_meta_learner(
    ensemble:   DeepLenseEnsemble,
    val_loader: torch.utils.data.DataLoader,
    device:     torch.device,
    epochs:     int = 5,
) -> None:
    """
    Trains the linear fusion head of the DeepLenseEnsemble using the Val set.
    The massive ResNet and ViT base models are frozen. We only train the 
    weights that learn how to combine their predictions.
    """
    print("\n🧠 Training Stacking Meta-Learner on Validation Set...")
    
    # [CRITICAL FIX]: Keep base models in eval mode so BatchNorm/Dropout are disabled
    ensemble.eval() 
    ensemble.fusion_head.train() # ONLY the linear head goes into train mode
    
    # Only the fusion head requires gradients
    optimizer = torch.optim.Adam(ensemble.fusion_head.parameters(), lr=0.005)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = ensemble(images) 
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(logits, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100.0 * correct / total
        print(f"    Epoch {epoch+1}/{epochs} │ Loss: {epoch_loss:.4f} │ Acc: {epoch_acc:.1f}%")
        
    ensemble.eval() # Return everything to eval mode for inference
    print("✅ Meta-Learner training complete.")


# ─────────────────────────────────────────────────────────────────────────────
# STANDARD EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def run_standard_evaluation(
    ensemble:   DeepLenseEnsemble,
    loader:     torch.utils.data.DataLoader,
    device:     torch.device,
) -> tuple[list, list, np.ndarray]:
    ensemble.eval()
    all_preds  = []
    all_labels = []
    all_probs  = []

    print("\n📊 Running Standard Ensemble Evaluation...")
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Standard Eval"):
            images, labels = images.to(device), labels.to(device)

            # [GSOC UPGRADE 3] Convert new Logit output to Probabilities safely
            logits = ensemble(images)                   # (B, 3) logits
            ensemble_probs = F.softmax(logits, dim=1)   # (B, 3) probabilities
            _, predicted   = torch.max(logits, dim=1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(ensemble_probs.cpu().numpy())

    all_probs = np.array(all_probs)
    return all_preds, all_labels, all_probs


# ─────────────────────────────────────────────────────────────────────────────
# TTA DIAGNOSTIC
# ─────────────────────────────────────────────────────────────────────────────

def run_tta_evaluation(
    ensemble:   DeepLenseEnsemble,
    loader:     torch.utils.data.DataLoader,
    device:     torch.device,
    angles:     list[int],
) -> tuple[list, list, np.ndarray]:
    ensemble.eval()
    all_preds  = []
    all_labels = []
    all_probs  = []

    print(f"\n🔄 Running TTA Diagnostic — Angles: {angles}")
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"  TTA {angles}"):
            images, labels = images.to(device), labels.to(device)

            accumulated_probs: torch.Tensor | None = None

            for angle in angles:
                rotated = rotate_batch(images, angle)

                # [GSOC UPGRADE 3] Convert Logits to Probs
                logits = ensemble(rotated)   
                probs = F.softmax(logits, dim=1) 

                if accumulated_probs is None:
                    accumulated_probs = torch.zeros_like(probs)
                accumulated_probs += probs

            final_probs  = accumulated_probs / len(angles)  
            _, predicted = torch.max(final_probs, dim=1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(final_probs.cpu().numpy())

    all_probs = np.array(all_probs)
    return all_preds, all_labels, all_probs


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    set_seed(42)
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.assets_dir, exist_ok=True)

    print("\n" + "="*60)
    print("  DeepLense GSoC 2026 — Ensemble Evaluation Pipeline")
    print(f"  Device     : {device}")
    print(f"  ResNet wts : {args.resnet_weights}")
    print(f"  ViT wts    : {args.vit_weights}")
    if not args.skip_tta:
        print(f"  TTA angles : {args.tta_angles}")
    else:
        print(f"  TTA        : skipped (--skip_tta)")
    print("="*60)

    if args.zip_path:
        staged_dir = stage_data_locally(args.zip_path)
        if staged_dir:
            args.data_dir = staged_dir
            args.csv_path = os.path.join(staged_dir, os.path.basename(args.csv_path))

    # ── [GSOC UPGRADE 2] Unpack full 6-item split to get Test Loader ──────
    _, val_loader, test_loader, _, _, _ = get_dataloaders(
        csv_path   = args.csv_path,
        base_dir   = args.data_dir,
        mode       = 'RGB',
        image_size = 224,
        batch_size = args.batch_size,
        augment    = False,     
    )

    print("\n🔧 Loading component models...")
    resnet = load_model(ResNetTransfer(num_classes=3), args.resnet_weights, device)
    vit = load_model(ViTChampion(num_classes=3), args.vit_weights, device)

    print("\n🔀 Fusing models into DeepLenseEnsemble...")
    ensemble = DeepLenseEnsemble(
        resnet_model = resnet,
        vit_model    = vit,
        freeze_base  = True,    
    )
    ensemble = ensemble.to(device)
    
    # ── [GSOC UPGRADE 1] Train the Meta-Learner ───────────────────────────
    if ensemble.learnable_fusion:
        train_meta_learner(ensemble, val_loader, device, epochs=5)

    classes   = ['No Sub', 'CDM', 'Vortex']
    cdm_index = classes.index('CDM')    

    # ── [GSOC UPGRADE 2] Standard evaluation on strictly isolated TEST SET
    std_preds, std_labels, std_probs = run_standard_evaluation(
        ensemble, test_loader, device
    )

    std_report = generate_classification_report(
        std_labels, std_preds,
        classes    = classes,
        model_name = 'ENSEMBLE (ResNet + ViT) — Standard',
    )

    save_confusion_matrix(
        std_labels, std_preds,
        classes   = classes,
        save_path = os.path.join(args.assets_dir, 'ensemble_confusion_matrix.png'),
        title     = 'DeepLense Ensemble — Test Set Evaluation',
        cmap      = 'Oranges',
    )

    std_auc = plot_multiclass_roc_auc(
        std_labels, std_probs,
        classes    = classes,
        save_path  = os.path.join(args.assets_dir, 'ensemble_roc_auc.png'),
        model_name = 'ENSEMBLE (ResNet + ViT)',
    )
    
    # ── [GSOC UPGRADE 4] Physics Calibration Curve ────────────────────────
    plot_calibration_curves(
        std_labels, std_probs,
        classes    = classes,
        save_path  = os.path.join(args.assets_dir, 'ensemble_calibration.png'),
        model_name = 'ENSEMBLE (ResNet + ViT)',
    )

    # ── 6. TTA Diagnostic on TEST SET ─────────────────────────────────────
    tta_auc    = None
    tta_report = None

    if not args.skip_tta:
        tta_preds, tta_labels, tta_probs = run_tta_evaluation(
            ensemble, test_loader, device, args.tta_angles
        )

        tta_report = generate_classification_report(
            tta_labels, tta_preds,
            classes    = classes,
            model_name = f'ENSEMBLE TTA {args.tta_angles}',
        )

        save_confusion_matrix(
            tta_labels, tta_preds,
            classes   = classes,
            save_path = os.path.join(args.assets_dir, 'tta_confusion_matrix.png'),
            title     = f'DeepLense Ensemble — TTA {args.tta_angles}',
            cmap      = 'Reds',
        )

        tta_auc = plot_multiclass_roc_auc(
            tta_labels, tta_probs,
            classes    = classes,
            save_path  = os.path.join(args.assets_dir, 'tta_roc_auc.png'),
            model_name = f'ENSEMBLE TTA {args.tta_angles}',
        )

        plot_tta_degradation(
            labels_before = std_labels,
            preds_before  = std_preds,
            labels_after  = tta_labels,
            preds_after   = tta_preds,
            classes       = classes,
            save_path     = os.path.join(args.assets_dir, 'tta_degradation.png'),
            model_name    = 'ENSEMBLE (ResNet + ViT)',
        )

    # ── 7. Final summary ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  ENSEMBLE EVALUATION COMPLETE")
    print(f"  {'─'*38}")
    print(f"  Standard Macro AUC   : {std_auc['macro']:.4f}  ← ML4SCI primary")
    print(f"  Standard FPR@90% TPR : {std_auc.get('fpr_90_macro', 0.0):.4f}  ← Physics standard")
    print(f"  Standard Macro F1    : {std_report['f1_macro']:.4f}")
    print(f"  Standard CDM AUC     : {std_auc['per_class'][cdm_index]:.4f}")

    if tta_auc is not None and tta_report is not None:
        print(f"  {'─'*38}")
        print(f"  TTA Macro AUC        : {tta_auc['macro']:.4f}")
        print(f"  TTA FPR@90% TPR      : {tta_auc.get('fpr_90_macro', 0.0):.4f}")
        print(f"  TTA Macro F1         : {tta_report['f1_macro']:.4f}")
        print(f"  TTA CDM AUC          : {tta_auc['per_class'][cdm_index]:.4f}")
        auc_delta = tta_auc['macro'] - std_auc['macro']
        f1_delta  = tta_report['f1_macro'] - std_report['f1_macro']
        print(f"  {'─'*38}")
        print(f"  AUC  Δ (rotation)    : {auc_delta:+.4f}  ← orientation bias (AUC)")
        print(f"  F1   Δ (rotation)    : {f1_delta:+.4f}  ← orientation bias (F1)")
        print(f"  → See assets/tta_degradation.png for per-class breakdown")

    print(f"\n  All plots saved to : {args.assets_dir}/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
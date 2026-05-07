import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve  # [GSOC UPGRADE 2] For reliability diagrams

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

CLASSES       = ['No Sub', 'CDM', 'Vortex']
CLASS_COLORS  = ['#2196F3', '#F44336', '#4CAF50']   # Blue, Red, Green

# FIX: Graceful matplotlib style fallback.
try:
    plt.style.use('seaborn-v0_8-whitegrid')
    PLOT_STYLE = 'seaborn-v0_8-whitegrid'
except OSError:
    PLOT_STYLE = 'seaborn-whitegrid'


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_dir(path):
    """Creates the parent directory of a file path if it does not exist."""
    if path:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  ROC-AUC  (PRIMARY ML4SCI METRIC) + FPR@TPR=0.90
# ─────────────────────────────────────────────────────────────────────────────

def plot_multiclass_roc_auc(
    all_labels,
    all_probs,
    classes=CLASSES,
    save_path=None,
    title='Receiver Operating Characteristic (ROC)',
    model_name='Model',
):
    """
    Generates and saves a publication-quality multi-class ROC curve.

    [GSOC UPGRADE 1] - Physics-Informed Metrics:
    In dark matter searches, physicists care deeply about operational thresholds.
    We now compute the False Positive Rate (FPR) at a guaranteed 90% True 
    Positive Rate (TPR). Minimizing this FPR while keeping a 90% detection 
    efficiency is a gold-standard physics requirement.
    """
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    n_classes   = len(classes)
    y_bin       = label_binarize(all_labels, classes=list(range(n_classes)))

    fpr, tpr, roc_auc, fpr_at_90 = {}, {}, {}, {}

    # ── Per-class curves ──────────────────────────────────────────────────
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc[i]        = auc(fpr[i], tpr[i])
        
        # Calculate FPR at exactly 90% TPR using linear interpolation
        fpr_at_90[i] = np.interp(0.90, tpr[i], fpr[i])

    # ── Micro-average curve (aggregate TP/FP across all classes) ─────────
    fpr['micro'], tpr['micro'], _ = roc_curve(y_bin.ravel(), all_probs.ravel())
    roc_auc['micro']              = auc(fpr['micro'], tpr['micro'])
    fpr_at_90['micro']            = np.interp(0.90, tpr['micro'], fpr['micro'])

    # ── Macro-average AUC (unweighted mean — the headline number) ────────
    roc_auc['macro'] = np.mean([roc_auc[i] for i in range(n_classes)])
    fpr_at_90['macro'] = np.mean([fpr_at_90[i] for i in range(n_classes)])

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  ROC-AUC & PHYSICS REPORT  —  {model_name}")
    print(f"{'='*65}")
    for i, cls in enumerate(classes):
        print(f"  {cls:<12}: AUC = {roc_auc[i]:.4f}  |  FPR @ 90% TPR = {fpr_at_90[i]:.4f}")
    print(f"  {'-'*61}")
    print(f"  {'Micro-Avg':<12}: AUC = {roc_auc['micro']:.4f}  |  FPR @ 90% TPR = {fpr_at_90['micro']:.4f}")
    print(f"  {'Macro-Avg':<12}: AUC = {roc_auc['macro']:.4f}  |  FPR @ 90% TPR = {fpr_at_90['macro']:.4f}")
    print(f"{'='*65}\n")

    # ── Plot ─────────────────────────────────────────────────────────────
    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 8))

        # Per-class curves
        for i, (cls, color) in enumerate(zip(classes, CLASS_COLORS)):
            ax.plot(
                fpr[i], tpr[i],
                color=color, lw=2.5,
                label=f'{cls}  (AUC = {roc_auc[i]:.3f})'
            )

        # Micro-average
        ax.plot(
            fpr['micro'], tpr['micro'],
            color='darkorange', lw=2, linestyle='--',
            label=f'Micro-avg  (AUC = {roc_auc["micro"]:.3f})'
        )

        # Random classifier baseline
        ax.plot(
            [0, 1], [0, 1],
            'k--', lw=1.2, label='Random Classifier (AUC = 0.500)'
        )

        # Macro AUC annotation box
        ax.text(
            0.62, 0.12,
            f'Macro-Avg AUC = {roc_auc["macro"]:.4f}\nFPR @ 90% TPR = {fpr_at_90["macro"]:.4f}',
            transform=ax.transAxes,
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.9)
        )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=13)
        ax.set_ylabel('True Positive Rate', fontsize=13)
        ax.set_title(f'{title}\n{model_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        _ensure_dir(save_path)
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"📊 ROC curve saved → {save_path}")
        plt.show()
        plt.close()

    return {
        'per_class': {i: roc_auc[i] for i in range(n_classes)},
        'macro':     roc_auc['macro'],
        'micro':     roc_auc['micro'],
        'fpr_90_macro': fpr_at_90['macro'], # Added to export
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def save_confusion_matrix(
    y_true,
    y_pred,
    classes=CLASSES,
    save_path=None,
    title='Confusion Matrix',
    cmap='Blues',
    normalize=False,
):
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt        = '.2f'
        title      = title + ' (Normalised)'
    else:
        cm_display = cm
        fmt        = 'd'

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            cm_display,
            annot=True, fmt=fmt, cmap=cmap,
            xticklabels=classes, yticklabels=classes,
            linewidths=0.5, linecolor='gray',
            ax=ax,
        )

        ax.set_title(title, fontsize=14, fontweight='bold', pad=14)
        ax.set_ylabel('True Physics Label', fontsize=12)
        ax.set_xlabel('Model Predicted Label', fontsize=12)
        ax.tick_params(axis='x', rotation=0)
        ax.tick_params(axis='y', rotation=0)

        plt.tight_layout()
        _ensure_dir(save_path)
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"📊 Confusion matrix saved → {save_path}")
        plt.show()
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CLASSIFICATION REPORT
# ─────────────────────────────────────────────────────────────────────────────

def generate_classification_report(
    y_true,
    y_pred,
    classes=CLASSES,
    model_name='Model',
):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print(f"\n{'='*52}")
    print(f"  CLASSIFICATION REPORT  —  {model_name}")
    print(f"{'='*52}")
    report_str = classification_report(y_true, y_pred, target_names=classes, digits=4)
    print(report_str)

    f1_macro    = f1_score(y_true, y_pred, average='macro',    zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # [FIX APPLIED]: Dynamic label scaling for F1 computation
    dynamic_labels = list(range(len(classes)))
    f1_per_cls  = f1_score(y_true, y_pred, average=None,       zero_division=0,
                           labels=dynamic_labels)

    f1_per_class_named = {cls: float(f1_per_cls[i]) for i, cls in enumerate(classes)}

    precision_macro = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    recall_macro    = float(recall_score(y_true, y_pred, average='macro',    zero_division=0))

    print(f"  Macro F1        : {f1_macro:.4f}  ← use in result dicts")
    print(f"  Weighted F1     : {f1_weighted:.4f}")
    print(f"  Macro Precision : {precision_macro:.4f}")
    print(f"  Macro Recall    : {recall_macro:.4f}")
    print(f"{'='*52}\n")

    return {
        'report_str':      report_str,
        'f1_macro':        float(f1_macro),
        'f1_weighted':     float(f1_weighted),
        'f1_per_class':    f1_per_class_named,
        'precision_macro': precision_macro,
        'recall_macro':    recall_macro,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  LEARNING CURVES  (per-epoch train vs val tracking)
# ─────────────────────────────────────────────────────────────────────────────

def plot_learning_curves(
    train_losses,
    val_losses,
    train_accs,
    val_accs,
    save_path=None,
    model_name='Model',
):
    epochs = range(1, len(train_losses) + 1)

    with plt.style.context(PLOT_STYLE):
        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Learning Curves — {model_name}', fontsize=14, fontweight='bold')

        # ── Loss subplot ─────────────────────────────────────────────────
        ax_loss.plot(epochs, train_losses, 'b-o', markersize=4,
                     linewidth=2, label='Train Loss')
        ax_loss.plot(epochs, val_losses, 'r-o', markersize=4,
                     linewidth=2, label='Val Loss')
        ax_loss.set_xlabel('Epoch', fontsize=12)
        ax_loss.set_ylabel('Cross-Entropy Loss', fontsize=12)
        ax_loss.set_title('Loss over Epochs', fontsize=12)
        ax_loss.legend(fontsize=11)
        ax_loss.grid(alpha=0.4)

        best_val_loss_epoch = int(np.argmin(val_losses)) + 1
        ax_loss.axvline(
            x=best_val_loss_epoch, color='red',
            linestyle='--', alpha=0.5,
            label=f'Best Val Loss (epoch {best_val_loss_epoch})'
        )
        ax_loss.legend(fontsize=10)

        # ── Accuracy subplot ─────────────────────────────────────────────
        ax_acc.plot(epochs, train_accs, 'b-o', markersize=4,
                    linewidth=2, label='Train Acc')
        ax_acc.plot(epochs, val_accs, 'r-o', markersize=4,
                    linewidth=2, label='Val Acc')
        ax_acc.set_xlabel('Epoch', fontsize=12)
        ax_acc.set_ylabel('Accuracy (%)', fontsize=12)
        ax_acc.set_title('Accuracy over Epochs', fontsize=12)
        ax_acc.legend(fontsize=11)
        ax_acc.grid(alpha=0.4)

        best_val_acc_epoch = int(np.argmax(val_accs)) + 1
        best_val_acc_value = max(val_accs)
        ax_acc.axvline(
            x=best_val_acc_epoch, color='red',
            linestyle='--', alpha=0.5,
        )
        ax_acc.annotate(
            f'Best: {best_val_acc_value:.1f}%\n(epoch {best_val_acc_epoch})',
            xy=(best_val_acc_epoch, best_val_acc_value),
            xytext=(best_val_acc_epoch + 0.5, best_val_acc_value - 5),
            fontsize=9,
            arrowprops=dict(arrowstyle='->', color='black'),
        )

        plt.tight_layout()
        _ensure_dir(save_path)
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"📊 Learning curves saved → {save_path}")
        plt.show()
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  TTA DEGRADATION ANALYSIS 
# ─────────────────────────────────────────────────────────────────────────────

def plot_tta_degradation(
    labels_before,
    preds_before,
    labels_after,
    preds_after,
    classes=CLASSES,
    save_path=None,
    model_name='Ensemble',
):
    # [FIX APPLIED]: Dynamic labels instead of hardcoded [0, 1, 2]
    dynamic_labels = list(range(len(classes)))
    
    f1_before = f1_score(labels_before, preds_before, average=None,
                         labels=dynamic_labels, zero_division=0)
    f1_after  = f1_score(labels_after,  preds_after,  average=None,
                         labels=dynamic_labels, zero_division=0)
    delta_f1  = f1_after - f1_before

    x        = np.arange(len(classes))
    bar_w    = 0.35

    with plt.style.context(PLOT_STYLE):
        fig, (ax_bar, ax_delta) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f'TTA Rotational Variance Analysis — {model_name}',
            fontsize=14, fontweight='bold'
        )

        bars1 = ax_bar.bar(x - bar_w/2, f1_before, bar_w,
                           label='Standard Eval', color='#2196F3', alpha=0.85)
        bars2 = ax_bar.bar(x + bar_w/2, f1_after,  bar_w,
                           label='After TTA (0°/90°/180°/270°)',
                           color='#F44336', alpha=0.85)

        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(classes, fontsize=12)
        ax_bar.set_ylabel('F1-Score', fontsize=12)
        ax_bar.set_ylim([0, 1.1])
        ax_bar.set_title('F1-Score: Before vs After Rotational TTA', fontsize=12)
        ax_bar.legend(fontsize=11)
        ax_bar.grid(axis='y', alpha=0.4)

        for bar in bars1:
            ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

        colors_delta = ['#4CAF50' if d >= 0 else '#F44336' for d in delta_f1]
        ax_delta.bar(classes, delta_f1, color=colors_delta, alpha=0.85, edgecolor='black')
        ax_delta.axhline(y=0, color='black', linewidth=1.2)
        ax_delta.set_ylabel('ΔF1 (After − Before)', fontsize=12)
        ax_delta.set_title('Per-Class F1 Degradation Under Rotation', fontsize=12)
        ax_delta.grid(axis='y', alpha=0.4)

        for i, (cls, d) in enumerate(zip(classes, delta_f1)):
            ax_delta.text(i, d + (0.005 if d >= 0 else -0.015),
                          f'{d:+.3f}', ha='center', va='bottom', fontsize=10,
                          fontweight='bold')

        worst_cls = classes[int(np.argmin(delta_f1))]
        ax_delta.set_xlabel(
            f'→ "{worst_cls}" suffers the largest degradation, motivating '
            f'E(2)-Equivariant Networks.',
            fontsize=9, style='italic'
        )

        plt.tight_layout()
        _ensure_dir(save_path)
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"📊 TTA degradation plot saved → {save_path}")
        plt.show()
        plt.close()

    print(f"\n{'='*52}")
    print(f"  TTA DEGRADATION SUMMARY  —  {model_name}")
    print(f"{'='*52}")
    print(f"  {'Class':<12}  {'F1 Before':>10}  {'F1 After':>10}  {'Delta':>8}")
    print(f"  {'-'*46}")
    for cls, fb, fa, d in zip(classes, f1_before, f1_after, delta_f1):
        print(f"  {cls:<12}  {fb:>10.4f}  {fa:>10.4f}  {d:>+8.4f}")
    print(f"{'='*52}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MODEL COMPARISON SUMMARY TABLE  
# ─────────────────────────────────────────────────────────────────────────────

def print_model_comparison_table(results: list[dict]):
    header = (
        f"\n{'='*82}\n" # Expanded width to accommodate new physics metric
        f"  {'Model':<26} {'Val Acc':>8} {'MacroAUC':>10} "
        f"{'CDM AUC':>9} {'FPR@90%':>9} {'F1 Macro':>10}\n"
        f"  {'-'*78}"
    )
    print(header)
    for r in results:
        # Gracefully handle missing metric if older result format is passed
        fpr_val = r.get('fpr_90_macro', 0.0) 
        print(
            f"  {r['model']:<26} "
            f"{r['val_acc']:>7.1f}% "
            f"{r['macro_auc']:>10.4f} "
            f"{r['cdm_auc']:>9.4f} "
            f"{fpr_val:>9.4f} "
            f"{r['f1_macro']:>10.4f}"
        )
    print(f"{'='*82}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  CALIBRATION CURVES (RELIABILITY DIAGRAM) - [GSOC UPGRADE 2]
# ─────────────────────────────────────────────────────────────────────────────

def plot_calibration_curves(
    y_true,
    y_probs,
    classes=CLASSES,
    save_path=None,
    model_name='Model',
):
    """
    Plots a Reliability Diagram (Calibration Curve) for the model.
    
    A perfectly calibrated model (the dotted line) outputs probabilities that 
    exactly match their empirical frequencies. Neural networks are often 
    overconfident. This diagnostic proves deep ML maturity by showing you 
    evaluate the *trustworthiness* of the probabilities, not just accuracy.
    """
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    n_classes = len(classes)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 8))

        # Perfectly calibrated reference line
        ax.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")

        for i, (cls, color) in enumerate(zip(classes, CLASS_COLORS)):
            # Computes fraction of positives and mean predicted probability per bin
            prob_true, prob_pred = calibration_curve(y_bin[:, i], y_probs[:, i], n_bins=10)
            ax.plot(prob_pred, prob_true, "s-", color=color, label=f"{cls}")

        ax.set_ylabel("Fraction of positives (Empirical True Probability)", fontsize=12)
        ax.set_xlabel("Mean predicted value (Model Confidence)", fontsize=12)
        ax.set_title(f"Reliability Diagram (Calibration Curves)\n{model_name}", fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        _ensure_dir(save_path)
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"📊 Calibration curves saved → {save_path}")
        plt.show()
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 8.  GRADCAM — ResNet models                          [GSOC UPGRADE: NEW]
# ─────────────────────────────────────────────────────────────────────────────
#
# GradCAM computes a class-discriminative localisation map by weighting the
# activation maps of the final convolutional layer by the gradients of the
# class score with respect to those activations.
#
# Reference: Selvaraju et al. (2017) — "Grad-CAM: Visual Explanations from
# Deep Networks via Gradient-based Localization"
#
# DESIGN NOTE — EquivariantCNN edge case:
#   Standard GradCAM hooks attach to the output of a nn.Conv2d layer and
#   capture a (B, C, H, W) activation tensor. escnn's R2Conv does NOT return
#   a plain nn.Conv2d — it returns a GeometricTensor wrapper. Registering
#   a backward hook on R2Conv raises a RuntimeError because GeometricTensor
#   gradients are tracked on the `.tensor` attribute, not on the module output
#   directly.
#
#   Solution: For EquivariantCNN we hook the GROUP POOLING output instead
#   (self.group_pool), which returns a plain tensor after the pooling step.
#   This is still spatially meaningful — it captures the equivariant feature
#   map after the final group symmetry is collapsed, giving a valid CAM.
#   The function detects EquivariantCNN by class name and routes accordingly.
# ─────────────────────────────────────────────────────────────────────────────

def compute_gradcam(
    model,
    image_tensor,
    class_idx=None,
    device=None,
):
    """
    Computes a GradCAM heatmap for a single image.

    Supports:
        - ResNetBaseline / ResNetTransfer  → hooks model.model.layer4[-1]
        - ViTChampion                      → delegates to compute_attention_rollout()
        - EquivariantCNN                   → hooks model.group_pool (see DESIGN NOTE)

    Args:
        model        : One of the four DeepLense model classes.
        image_tensor : torch.Tensor of shape (1, C, H, W) — single image, batched.
        class_idx    : int — target class index (0/1/2). If None, uses argmax.
        device       : torch.device. If None, inferred from model parameters.

    Returns:
        cam          : np.ndarray of shape (H, W) — GradCAM heatmap in [0, 1].
        pred_class   : int — predicted class index.
        pred_probs   : np.ndarray of shape (3,) — softmax probabilities.
    """
    import torch
    import torch.nn.functional as F

    if device is None:
        device = next(model.parameters()).device

    model_class = type(model).__name__

    # ── ViT: delegate to Attention Rollout ───────────────────────────────
    if model_class == 'ViTChampion':
        return compute_attention_rollout(model, image_tensor, class_idx, device)

    image_tensor = image_tensor.to(device)
    model.eval()

    # ── Storage for hooks ─────────────────────────────────────────────────
    activations = {}
    gradients   = {}

    # ── Select hook strategy based on model class ───────────────────────────
    if model_class == 'EquivariantCNN':
        # escnn shares internal FieldType buffers across forward passes.
        # Neither module hooks nor .backward() can be safely reused across
        # multiple calls without triggering "graph freed" errors.
        #
        # Solution: torch.autograd.grad() with a CLONED fresh input each call.
        # autograd.grad() computes gradients without accumulating into .grad
        # attributes and without retaining the graph — it is a pure functional
        # call that does not interact with escnn's internal state across calls.
        #
        # We use input-gradient saliency: d(class_score)/d(input_pixels).
        # Reference: Simonyan et al. (2013) "Deep Inside Convolutional Networks"

        import cv2

        # Clone + detach — completely fresh tensor, no graph history
        inp = image_tensor.detach().clone().to(device).requires_grad_(True)

        # Get prediction first with no_grad (fast, no graph)
        with torch.no_grad():
            logits_nd = model(inp.detach())
            probs     = F.softmax(logits_nd, dim=1)
            if class_idx is None:
                class_idx = int(logits_nd.argmax(dim=1).item())
            pred_probs = probs.squeeze(0).cpu().numpy()

        # Now run a fresh forward pass WITH grad just for saliency
        inp2   = image_tensor.detach().clone().to(device).requires_grad_(True)
        logits2 = model(inp2)
        score   = logits2[0, class_idx]

        # autograd.grad: functional, does not touch .grad, no graph retention
        grads = torch.autograd.grad(
            outputs=score,
            inputs=inp2,
            retain_graph=False,
            create_graph=False,
        )[0]  # (1, C, H, W)

        # Saliency map: abs gradient averaged over channels → (H, W)
        cam_tensor = grads.detach().abs().squeeze(0).mean(dim=0)

        cam_min = cam_tensor.min()
        cam_max = cam_tensor.max()
        if cam_max > cam_min:
            cam_tensor = (cam_tensor - cam_min) / (cam_max - cam_min)
        else:
            cam_tensor = torch.zeros_like(cam_tensor)

        cam_np      = cam_tensor.cpu().numpy()
        h_in, w_in  = image_tensor.shape[2], image_tensor.shape[3]
        cam_resized = cv2.resize(cam_np, (w_in, h_in), interpolation=cv2.INTER_LINEAR)
        return cam_resized, class_idx, pred_probs

    elif model_class in ('ResNetBaseline', 'ResNetTransfer'):

        def _save_activation(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook

        def _save_gradient(name):
            def hook(module, grad_input, grad_output):
                gradients[name] = grad_output[0].detach()
            return hook

        target_module = model.model.layer4[-1]
        fwd_hook = target_module.register_forward_hook(_save_activation('target'))
        bwd_hook = target_module.register_full_backward_hook(_save_gradient('target'))

        try:
            model.zero_grad()
            image_tensor = image_tensor.detach().requires_grad_(True)

            with torch.enable_grad():
                logits = model(image_tensor)
                probs  = F.softmax(logits, dim=1)

                if class_idx is None:
                    class_idx = int(logits.argmax(dim=1).item())

                pred_probs = probs.squeeze(0).detach().cpu().numpy()

                score = logits[0, class_idx]
                score.backward()

        finally:
            fwd_hook.remove()
            bwd_hook.remove()

    else:
        raise ValueError(
            f"compute_gradcam: unsupported model class '{model_class}'. "
            "Use ResNetBaseline, ResNetTransfer, ViTChampion, or EquivariantCNN."
        )

    # ── Compute GradCAM ───────────────────────────────────────────────────
    act  = activations['target'].squeeze(0)   # (C, H, W)
    grad = gradients['target'].squeeze(0)     # (C, H, W)

    # Global average pool the gradients → importance weights per channel
    weights = grad.mean(dim=(1, 2))           # (C,)

    # Weighted combination of activation maps
    # act lives on GPU when CUDA is available — cam must match its device
    cam = torch.zeros(act.shape[1:], dtype=torch.float32, device=act.device)
    for c, w in enumerate(weights):
        cam += w * act[c]

    # ReLU — only keep positive contributions
    cam = F.relu(cam)

    # Normalise to [0, 1]
    cam_min = cam.min()
    cam_max = cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = torch.zeros_like(cam)

    cam_np = cam.detach().cpu().numpy()

    # Resize to input spatial resolution
    import cv2
    h_in = image_tensor.shape[2]
    w_in = image_tensor.shape[3]
    cam_resized = cv2.resize(cam_np, (w_in, h_in), interpolation=cv2.INTER_LINEAR)

    return cam_resized, class_idx, pred_probs


# ─────────────────────────────────────────────────────────────────────────────
# 9.  ATTENTION ROLLOUT — ViT models                   [GSOC UPGRADE: NEW]
# ─────────────────────────────────────────────────────────────────────────────
#
# Standard GradCAM cannot directly be applied to ViT because there is no
# spatial feature map equivalent to a CNN's layer4. Instead we use
# Attention Rollout (Abnar & Zuidema, 2020), which recursively multiplies
# the attention weights across all transformer layers and extracts the
# CLS-token row to obtain a per-patch importance map.
#
# Reference: Abnar & Zuidema (2020) — "Quantifying Attention Flow in
# Transformers"
# ─────────────────────────────────────────────────────────────────────────────

def compute_attention_rollout(
    model,
    image_tensor,
    class_idx=None,
    device=None,
):
    """
    Computes an Attention Rollout map for a ViTChampion model.

    The ViT-B/16 divides the 224×224 image into a 14×14 grid of patches.
    Attention Rollout gives an importance score per patch, which we reshape
    to (14, 14) and upsample to (224, 224) for overlay.

    Args:
        model        : ViTChampion instance.
        image_tensor : torch.Tensor of shape (1, 3, 224, 224).
        class_idx    : int — target class (unused by rollout, included for API compat).
        device       : torch.device.

    Returns:
        rollout_map  : np.ndarray of shape (H, W) — attention map in [0, 1].
        pred_class   : int — predicted class index.
        pred_probs   : np.ndarray of shape (3,) — softmax probabilities.
    """
    import torch
    import torch.nn.functional as F
    import cv2

    if device is None:
        device = next(model.parameters()).device

    image_tensor = image_tensor.to(device)
    model.eval()

    attention_maps = []

    def _make_attn_hook(layer_idx):
        def hook(module, input, output):
            # torchvision ViT: the attention module is nn.MultiheadAttention
            # We need to re-run the attention computation to extract weights.
            # However, torchvision's implementation does not expose weights
            # through the standard hook. We capture the input Q/K/V tensors
            # and recompute scaled dot-product attention manually.
            q, k, v = input[0], input[0], input[0]   # self-attention: Q=K=V
            B, S, D = q.shape
            H = module.num_heads
            d_h = D // H

            # Reshape to (B, H, S, d_h)
            q = q.reshape(B, S, H, d_h).transpose(1, 2)
            k = k.reshape(B, S, H, d_h).transpose(1, 2)

            scale = d_h ** -0.5
            attn  = torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
            # attn: (B, H, S, S) — average over heads
            attn_avg = attn.mean(dim=1).detach().cpu()   # (B, S, S)
            attention_maps.append(attn_avg)
        return hook

    # Register hooks on every encoder block's self_attention module
    hooks = []
    for i, encoder_block in enumerate(model.model.encoder.layers):
        # torchvision ViT encoder block has .self_attention attribute
        h = encoder_block.self_attention.register_forward_hook(_make_attn_hook(i))
        hooks.append(h)

    try:
        with torch.no_grad():
            logits = model(image_tensor)
            probs  = F.softmax(logits, dim=1)
    finally:
        for h in hooks:
            h.remove()

    pred_class = int(logits.argmax(dim=1).item())
    pred_probs = probs.squeeze(0).cpu().numpy()

    if class_idx is None:
        class_idx = pred_class

    # ── Rollout computation ───────────────────────────────────────────────
    # attention_maps: list of (B, S, S) tensors, one per transformer layer
    # S = 197 (196 patches + 1 CLS token), B = 1
    rollout = torch.eye(attention_maps[0].shape[-1])   # (S, S) identity

    for attn in attention_maps:
        # Add residual connection: A_hat = 0.5 * A + 0.5 * I
        attn_layer = attn[0]                           # (S, S)
        attn_with_res = 0.5 * attn_layer + 0.5 * torch.eye(attn_layer.shape[0])
        # Row-normalise
        attn_with_res = attn_with_res / attn_with_res.sum(dim=-1, keepdim=True)
        rollout = attn_with_res @ rollout

    # CLS token (row 0) attention to all patches (columns 1:)
    cls_attention = rollout[0, 1:]                     # (196,)
    grid_size     = int(cls_attention.shape[0] ** 0.5) # 14 for ViT-B/16
    attn_map      = cls_attention.reshape(grid_size, grid_size).numpy()

    # Normalise
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    # Upsample to original image resolution
    h_in = image_tensor.shape[2]
    w_in = image_tensor.shape[3]
    rollout_resized = cv2.resize(attn_map, (w_in, h_in), interpolation=cv2.INTER_LINEAR)

    return rollout_resized, pred_class, pred_probs


# ─────────────────────────────────────────────────────────────────────────────
# 10. GRADCAM OVERLAY VISUALISATION                    [GSOC UPGRADE: NEW]
# ─────────────────────────────────────────────────────────────────────────────

def overlay_gradcam(
    image_np,
    cam,
    alpha=0.45,
    colormap='jet',
):
    """
    Overlays a GradCAM / Attention Rollout heatmap onto the original image.

    Args:
        image_np  : np.ndarray of shape (H, W) or (H, W, 3), float in [0, 1].
        cam       : np.ndarray of shape (H, W), float in [0, 1].
        alpha     : float — heatmap opacity (default 0.45).
        colormap  : str — matplotlib colormap name (default 'jet').

    Returns:
        blended   : np.ndarray of shape (H, W, 3), float in [0, 1].
    """
    import matplotlib.cm as cm_module

    # Ensure image is (H, W, 3)
    if image_np.ndim == 2:
        image_rgb = np.stack([image_np] * 3, axis=-1)
    elif image_np.shape[2] == 1:
        image_rgb = np.concatenate([image_np] * 3, axis=-1)
    else:
        image_rgb = image_np.copy()

    # Normalise image to [0, 1]
    image_rgb = image_rgb.astype(float)
    img_min, img_max = image_rgb.min(), image_rgb.max()
    if img_max > img_min:
        image_rgb = (image_rgb - img_min) / (img_max - img_min)

    # Apply colormap to CAM
    cmap_fn = cm_module.get_cmap(colormap)
    heatmap = cmap_fn(cam)[:, :, :3]                  # drop alpha channel → (H, W, 3)

    # Alpha blend
    blended = (1 - alpha) * image_rgb + alpha * heatmap
    blended = np.clip(blended, 0.0, 1.0)

    return blended


def save_gradcam_visualization(
    model,
    dataloader,
    device,
    classes=CLASSES,
    save_dir=None,
    model_name='Model',
    n_samples_per_class=2,
    denormalize_fn=None,
):
    """
    Generates and saves GradCAM / Attention Rollout visualisations for
    n_samples_per_class images from each class.

    Creates one subplot grid per class showing:
        [Original Image | GradCAM Overlay | CAM Heatmap]

    Handles the escnn / EquivariantCNN edge case automatically via
    compute_gradcam() (hooks group_pool instead of a conv layer).

    Args:
        model            : Any DeepLense model.
        dataloader       : A DataLoader returning (images, labels).
        device           : torch.device.
        classes          : list[str] — class names.
        save_dir         : str — directory to save plots. If None, shows only.
        model_name       : str — used in plot titles and filenames.
        n_samples_per_class : int — how many images to visualise per class.
        denormalize_fn   : callable(tensor) → np.ndarray, optional.
                           If None, uses the standard ImageNet denorm for RGB
                           or the grayscale denorm automatically detected from
                           the image shape.

    Returns:
        None
    """
    import torch

    model.eval()
    _ensure_dir(save_dir + '/placeholder.png' if save_dir else None)

    # ── Collect n_samples_per_class images per class ──────────────────────
    collected = {i: [] for i in range(len(classes))}
    n_needed  = n_samples_per_class * len(classes)
    n_found   = 0

    for images, labels in dataloader:
        for img, lbl in zip(images, labels):
            c = int(lbl.item())
            if len(collected[c]) < n_samples_per_class:
                collected[c].append(img)
                n_found += 1
            if n_found >= n_needed:
                break
        if n_found >= n_needed:
            break

    # ── Default denormalise ───────────────────────────────────────────────
    def _default_denorm(t):
        """Undo ImageNet normalisation for RGB or grayscale tensors."""
        t = t.cpu().numpy()
        if t.shape[0] == 3:                         # RGB: (3, H, W)
            mean = np.array([0.485, 0.456, 0.406])
            std  = np.array([0.229, 0.224, 0.225])
            t    = t.transpose(1, 2, 0) * std + mean # (H, W, 3)
        else:                                        # Grayscale: (1, H, W)
            t    = t[0] * 0.5 + 0.5                 # undo Normalize(0.5, 0.5)
        return np.clip(t, 0, 1)

    denorm = denormalize_fn if denormalize_fn is not None else _default_denorm

    # ── Generate plots ────────────────────────────────────────────────────
    for class_idx, class_name in enumerate(classes):
        imgs = collected[class_idx]
        if not imgs:
            print(f"  ⚠️  No samples found for class '{class_name}' — skipping.")
            continue

        n_cols = 3                                   # Original | Overlay | Heatmap
        n_rows = len(imgs)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
        if n_rows == 1:
            axes = axes[np.newaxis, :]               # ensure 2-D indexing

        fig.suptitle(
            f'GradCAM — {model_name} | Class: {class_name}',
            fontsize=14, fontweight='bold'
        )

        for row_idx, img_tensor in enumerate(imgs):
            input_batch = img_tensor.unsqueeze(0)    # (1, C, H, W)
            img_np      = denorm(img_tensor)

            cam, pred_cls, pred_probs = compute_gradcam(
                model, input_batch, class_idx=class_idx, device=device
            )

            overlay = overlay_gradcam(img_np, cam)

            # Original
            ax_orig = axes[row_idx, 0]
            if img_np.ndim == 2 or (img_np.ndim == 3 and img_np.shape[-1] == 1):
                ax_orig.imshow(img_np if img_np.ndim == 2 else img_np[:, :, 0], cmap='gray')
            else:
                ax_orig.imshow(img_np)
            ax_orig.set_title(f'Original\nTrue: {class_name}', fontsize=10)
            ax_orig.axis('off')

            # Overlay
            ax_ovr = axes[row_idx, 1]
            ax_ovr.imshow(overlay)
            ax_ovr.set_title(
                f'GradCAM Overlay\nPred: {classes[pred_cls]} '
                f'({pred_probs[pred_cls]*100:.1f}%)',
                fontsize=10
            )
            ax_ovr.axis('off')

            # Pure heatmap
            ax_cam = axes[row_idx, 2]
            im     = ax_cam.imshow(cam, cmap='jet', vmin=0, vmax=1)
            ax_cam.set_title('CAM Heatmap', fontsize=10)
            ax_cam.axis('off')
            fig.colorbar(im, ax=ax_cam, fraction=0.046, pad=0.04)

        plt.tight_layout()

        if save_dir:
            safe_name = class_name.lower().replace(' ', '_')
            save_path = os.path.join(
                save_dir, f'gradcam_{model_name.lower().replace(" ", "_")}_{safe_name}.png'
            )
            fig.savefig(save_path, bbox_inches='tight', dpi=200)
            print(f"📊 GradCAM saved → {save_path}")

        plt.show()
        plt.close()
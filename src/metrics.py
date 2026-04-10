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
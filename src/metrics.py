import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize

def plot_multiclass_roc_auc(all_labels, all_probs, classes=['No Sub', 'CDM', 'Vortex'], save_path=None):
    """Generates and saves a professional multi-class ROC Curve."""
    y_test_bin = label_binarize(all_labels, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]

    fpr, tpr, roc_auc = dict(), dict(), dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Background Noise)')
    plt.ylabel('True Positive Rate (Correct Detection)')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Close plot to free up memory on the remote GPU backend
    plt.close()

def save_confusion_matrix(y_true, y_pred, classes=['No Sub', 'CDM', 'Vortex'], save_path=None, title='Confusion Matrix', cmap='Blues'):
    """
    Generates and saves a production-grade confusion matrix.
    Automatically handles directory creation and memory management.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Physics Label')
    plt.xlabel('AI Predicted Label')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.close()

def generate_classification_report(y_true, y_pred, classes=['No Sub', 'CDM', 'Vortex']):
    """
    Prints a cleanly formatted classification report to the console.
    """
    print("\n" + "="*50)
    print(f" EVALUATION REPORT ")
    print("="*50)
    report = classification_report(y_true, y_pred, target_names=classes)
    print(report)
    return report
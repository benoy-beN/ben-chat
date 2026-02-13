"""
Confusion Matrix Generator (V5).
Visualizes prediction accuracy across in-scope vs out-of-scope.
Saves to data/confusion_matrix.png.

Usage:
    python eval/confusion_matrix.py
    (Or called from evaluate.py)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    labels: list[str] = None,
    save_path: str = None
):
    """
    Generate and save confusion matrix.
    
    Args:
        y_true: Ground truth labels (1 = should match, 0 = should reject)
        y_pred: Predicted labels (1 = matched, 0 = rejected)
        labels: Class names (default: ["Reject", "Accept"])
        save_path: Path to save PNG
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, classification_report
    except ImportError:
        print("⚠️  matplotlib or sklearn not installed. Skipping confusion matrix.")
        return

    save_path = save_path or os.path.join(config.DATA_DIR, "confusion_matrix.png")
    labels = labels or ["Reject (OOS)", "Accept (Match)"]
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=labels)
    print(f"\n📊 Classification Report:\n{report}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(7, 6))
    
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels,
           yticklabels=labels,
           ylabel='Actual',
           xlabel='Predicted',
           title='SOP Chatbot V5 — Confusion Matrix')
    
    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    fontsize=18, fontweight='bold',
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"📊 Confusion matrix saved → {save_path}")
    
    # Compute metrics
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


if __name__ == "__main__":
    # Demo
    demo_true = [1]*40 + [0]*20
    demo_pred = [1]*38 + [0]*2 + [0]*18 + [1]*2
    plot_confusion_matrix(demo_true, demo_pred)

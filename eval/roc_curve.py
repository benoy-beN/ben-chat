"""
ROC Curve Generator (V5).
Plots ROC curve from evaluation scores and labels.
Saves to data/roc_curve.png.

Usage:
    python eval/roc_curve.py
    (Or called from evaluate.py)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def plot_roc_curve(scores: list[float], labels: list[int], save_path: str = None):
    """
    Generate and save ROC curve.
    
    Args:
        scores: List of confidence scores from pipeline
        labels: List of ground truth labels (1 = correct match, 0 = incorrect/OOS)
        save_path: Path to save the PNG (default: data/roc_curve.png)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
    except ImportError:
        print("⚠️  matplotlib or sklearn not installed. Skipping ROC curve.")
        return

    save_path = save_path or os.path.join(config.DATA_DIR, "roc_curve.png")
    
    scores_arr = np.array(scores)
    labels_arr = np.array(labels)
    
    fpr, tpr, thresholds = roc_curve(labels_arr, scores_arr)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#667eea', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', alpha=0.5)
    plt.scatter([fpr[optimal_idx]], [tpr[optimal_idx]], 
                color='#764ba2', s=100, zorder=5,
                label=f'Optimal threshold = {optimal_threshold:.3f}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('SOP Chatbot V5 — ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"📊 ROC curve saved → {save_path}")
    print(f"   AUC: {roc_auc:.4f}")
    print(f"   Optimal threshold: {optimal_threshold:.4f}")
    
    return {
        "auc": roc_auc,
        "optimal_threshold": float(optimal_threshold),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }


if __name__ == "__main__":
    # Demo with synthetic data
    np.random.seed(42)
    demo_scores = list(np.concatenate([
        np.random.beta(8, 2, 50),   # Correct matches (high scores)
        np.random.beta(2, 8, 30),   # Incorrect / OOS (low scores)
    ]))
    demo_labels = [1] * 50 + [0] * 30
    plot_roc_curve(demo_scores, demo_labels)

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

def threshold_analysis(y_true, y_score, model_name="Model", save_dir="reports"):
    """
    Plots Precision, Recall, and F1 score as threshold varies from 0 to 1.
    Saves optimal threshold based on max F1 score to text file.
    """
    thresholds = np.linspace(0, 1, 100)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    # Find best threshold
    best_idx = int(np.argmax(f1s))
    best_threshold = thresholds[best_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, recalls, label='Recall')
    plt.plot(thresholds, f1s, label='F1 Score')
    plt.axvline(x=0.5, color='gray', linestyle='--', label='Default Threshold = 0.5')
    plt.axvline(x=best_threshold, color='red', linestyle='--', label=f'Best Threshold = {best_threshold:.2f}')
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Threshold Analysis for {model_name}")
    plt.legend()
    plt.grid(True)

    os.makedirs(save_dir, exist_ok=True)
    filename = f"{model_name.lower().replace(' ', '_')}_threshold_curve.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    plt.close()

    # Save best threshold to file
    txt_path = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_best_threshold.txt")
    with open(txt_path, "w") as f:
        f.write(f"Best Threshold for {model_name}: {best_threshold:.4f}\n")

    print(f"[✓] Saved threshold analysis for {model_name} → {filepath}")
    print(f"[✓] Best threshold saved → {txt_path}")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def plot_confusion_matrix(cm, labels, save_path=None):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_scatter(X, y, save_path=None):
    df = pd.DataFrame(X[:, :2], columns=['PC1', 'PC2'])
    df['label'] = y
    sns.scatterplot(data=df, x='PC1', y='PC2', hue='label', palette='Set2')
    plt.title('2D PCA Projection')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_roc_curve(y_true, y_scores, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_precision_recall(y_true, y_scores, save_path=None):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure()
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_roc_all(models: dict, X_test, y_test, save_path=None):
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        try:
            if hasattr(model, "decision_function"):
                y_score = model.decision_function(X_test)
            else:
                y_score = model.predict_proba(X_test)[:, 1]

            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
        except Exception as e:
            print(f"Skipping {name} due to error: {e}")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for All Models')
    plt.legend(loc='lower right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def save_classification_report(report_dict, filename='report.csv'):
    df = pd.DataFrame(report_dict).transpose()
    os.makedirs('reports', exist_ok=True)
    df.to_csv(f'reports/{filename}', index=True)


# ✅ NEW utility to save SHAP plots for consistency (optional use)
def show_and_save_plot(fig, save_path=None):
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    fig.show()
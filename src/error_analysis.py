import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
import os

def analyze_errors(model, X_test, y_test, class_names=['Healthy', 'Cancer'], save_path=None):
    """
    Analyze classification errors of a trained model on test data.
    Generates a classification report, confusion matrix heatmap, and
    prints some misclassified examples.

    Parameters:
    - model: trained classifier
        The machine learning model to evaluate.
    - X_test: array-like
        Feature matrix for testing.
    - y_test: array-like
        True labels for test set.
    - class_names: list of str, optional (default=['Healthy', 'Cancer'])
        Class names for display in reports and plots.
    - save_path: str or None, optional
        File path to save the confusion matrix figure. If None, figure is not saved.

    Returns:
    - df_report: pd.DataFrame
        Classification report as a DataFrame.
    - wrong_indices: np.ndarray
        Indices of misclassified samples.
    """

    # Try to get decision function scores, fallback to predicted probabilities
    try:
        y_score = model.decision_function(X_test)
    except:
        y_score = model.predict_proba(X_test)[:, 1]

    # Predict labels on test data
    y_pred = model.predict(X_test)

    # Generate detailed classification report as dictionary, then convert to DataFrame
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose()

    # Compute confusion matrix and plot it as heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix (Error Analysis)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    # Save the plot if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.show()

    # Identify indices where prediction is incorrect
    wrong_indices = np.where(y_pred != y_test)[0]

    print(f"\n❌ Number of misclassified samples: {len(wrong_indices)} out of {len(y_test)}")
    print("\nSome misclassified examples:")

    # Print a few misclassified sample details: index, true label, predicted label, and score
    for i in wrong_indices[:5]:
        print(f"Sample {i} → True: {class_names[y_test[i]]}, Predicted: {class_names[y_pred[i]]}, Score: {y_score[i]:.4f}")

    return df_report, wrong_indices

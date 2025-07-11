from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluates a classification model based on test data and decision threshold.

    Parameters:
    - model: Trained classifier
    - X_test: Test features
    - y_test: True test labels
    - threshold: Threshold for converting probabilities/scores to binary predictions

    Returns:
    - acc: Accuracy score
    - report: Classification report (as dict)
    - cm: Confusion matrix
    """
    try:
        y_score = model.decision_function(X_test)
    except AttributeError:
        y_score = model.predict_proba(X_test)[:, 1]

    y_pred = (y_score >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    return acc, report, cm
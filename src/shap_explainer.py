import shap
import matplotlib.pyplot as plt
import numpy as np
import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def get_shap_explainer(model, X):
    """
    Choose appropriate SHAP explainer based on model type.
    """
    base_model = model
    if isinstance(model, Pipeline):
        if 'xgb' in model.named_steps:
            base_model = model.named_steps['xgb']
        elif 'rf' in model.named_steps:
            base_model = model.named_steps['rf']
        elif 'dt' in model.named_steps:
            base_model = model.named_steps['dt']
        elif 'lr' in model.named_steps:
            base_model = model.named_steps['lr']
        elif 'knn' in model.named_steps:
            base_model = model.named_steps['knn']
        elif 'nb' in model.named_steps:
            base_model = model.named_steps['nb']
        elif 'svc' in model.named_steps:
            base_model = model.named_steps['svc']

    if isinstance(base_model, (RandomForestClassifier, XGBClassifier, DecisionTreeClassifier)):
        return shap.TreeExplainer(base_model), "tree"
    elif isinstance(base_model, LogisticRegression):
        return shap.LinearExplainer(base_model, X), "linear"
    elif isinstance(base_model, (KNeighborsClassifier, GaussianNB, SVC)):
        return shap.KernelExplainer(base_model.predict_proba, X, nsamples=100), "kernel"
    else:
        return shap.Explainer(base_model.predict_proba, X), "generic"


def explain_model(model_name: str, model, X: pd.DataFrame, feature_names=None, save_dir="reports"):
    """
    Generate SHAP summary plots (bar + beeswarm) for a trained model.

    Parameters:
    - model_name: str
    - model: trained model or pipeline
    - X: input data (must match training data distribution)
    - feature_names: list of column names
    - save_dir: where to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(X.shape[1])]

    explainer, explainer_type = get_shap_explainer(model, X)
    shap_values = explainer.shap_values(X)

    # Bar plot (mean abs SHAP value)
    plt.figure(figsize=(10, 4))
    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
    bar_path = os.path.join(save_dir, f"{model_name.lower()}_shap_bar.png")
    plt.tight_layout()
    plt.savefig(bar_path)
    plt.close()

    # Beeswarm plot (distribution)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    beeswarm_path = os.path.join(save_dir, f"{model_name.lower()}_shap_beeswarm.png")
    plt.tight_layout()
    plt.savefig(beeswarm_path)
    plt.close()

    return bar_path, beeswarm_path
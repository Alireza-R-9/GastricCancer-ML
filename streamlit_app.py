import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys, os
from sklearn.model_selection import StratifiedKFold

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import load_data, split_data
from src.feature_selector import apply_feature_engineering
from src.model_trainer import train_svm_aoa
from src.benchmark import benchmark_models
from src.evaluator import evaluate_model
from src.visualizer import (
    plot_confusion_matrix,
    plot_scatter,
    plot_roc_curve,
    plot_precision_recall,
    save_classification_report,
    plot_roc_all
)
from src.error_analysis import analyze_errors
from src.threshold_analysis import threshold_analysis
from src.shap_explainer import explain_model

# ------------------ Layout ------------------
st.set_page_config(page_title="Gastric Cancer Classifier", layout="wide")
st.title("🧬 Gastric Cancer Classifier")

# ------------------ Sidebar: Settings ------------------
st.sidebar.header("🔧 Model Settings")
feature_selection_method = st.sidebar.selectbox("Feature Selection Method", ["PCA", "SelectKBest", "None"])
n_components = st.sidebar.slider("Dimensionality (PCA / KBest)", 2, 8, 5)
thresh = st.sidebar.slider("Decision Threshold", 0.0, 1.0, 0.5, step=0.01)

use_cv_all = st.sidebar.checkbox("🔁 Use CV for All Models")
if use_cv_all:
    n_splits_all = st.sidebar.slider("Number of CV Folds", 3, 10, 5)

# ------------------ Sidebar: Dataset ------------------
st.sidebar.header("📦 Dataset Size")
data_options = {
    "1K": 1_000,
    "5K": 5_000,
    "10K": 10_000,
    "50K": 50_000,
    "100K": 100_000,
    "500K": 500_000,
    "1M": 1_000_000,
    "2M (Full)": 2_000_000
}
selected_label = st.sidebar.selectbox("Select Dataset Size", list(data_options.keys()), index=2)
selected_rows = data_options[selected_label]

# ------------------ Load Data ------------------
with st.spinner("📥 Loading data..."):
    X, y = load_data("data/gastric_big.csv")
    X = X[:selected_rows]
    y = y[:selected_rows]
    X_train, X_test, y_train, y_test = split_data(X, y)

    method_map = {
        "PCA": "pca",
        "SelectKBest": "kbest",
        "None": "none"
    }
    method_selected = method_map[feature_selection_method]
    X_train_fs, X_test_fs = apply_feature_engineering(
        X_train, X_test,
        y_train=y_train,  # 🛠️ Add y_train for KBest
        method=method_selected,
        n_components=n_components,
        k_best=n_components
    )

st.success(f"✅ Data Loaded: {selected_rows:,} samples")

# ------------------ Sidebar: Dataset Info ------------------
st.sidebar.header("📊 Dataset Overview")
num_total = len(y)
num_cancer = int(np.sum(y == 1))
num_healthy = int(np.sum(y == 0))
st.sidebar.write(f"Total Samples: {num_total}")
st.sidebar.write(f"Cancer: {num_cancer} ({100 * num_cancer / num_total:.1f}%)")
st.sidebar.write(f"Healthy: {num_healthy} ({100 * num_healthy / num_total:.1f}%)")
st.sidebar.write(f"Train Size: {len(y_train)}")
st.sidebar.write(f"Test Size: {len(y_test)}")

# ------------------ Class Distribution Chart ------------------
st.subheader("📈 Dataset Class Distribution")
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(x=["Healthy", "Cancer"], y=[num_healthy, num_cancer], palette=["#2ecc71", "#e74c3c"])
ax.set_ylabel("Count")
ax.set_title("Class Distribution")
st.pyplot(fig)

# ------------------ SVM + AOA ------------------
best_thresholds = []
model_accuracies = {}

if st.button("🚀 Run SVM + AOA Model"):
    with st.spinner("⏳ Training SVM + AOA..."):
        model, best_params = train_svm_aoa(X_train_fs, y_train)

        try:
            y_score = model.decision_function(X_test_fs)
        except:
            y_score = model.predict_proba(X_test_fs)[:, 1]

        os.makedirs("reports", exist_ok=True)
        threshold_analysis(y_test, y_score, model_name="SVM+AOA", save_dir="reports")

        threshold_path = "reports/svm+aoa_best_threshold.txt"
        best_threshold = thresh
        if os.path.exists(threshold_path):
            with open(threshold_path) as f:
                best_threshold = float(f.read().split(":")[-1])
                st.info(f"Auto-applied best threshold: {best_threshold:.2f}")
                best_thresholds.append(["SVM+AOA", f"{best_threshold:.2f}"])

        y_pred = (y_score >= best_threshold).astype(int)
        acc, report, cm = evaluate_model(model, X_test_fs, y_test, threshold=best_threshold)
        save_classification_report(report, filename="svm_aoa_report.csv")
        model_accuracies["SVM+AOA"] = acc

    st.subheader("🔍 SVM + AOA Results")
    st.write(f"**Accuracy:** {acc:.4f}")
    st.write(f"**Optimized Params:** C={best_params[0]:.3f}, gamma={best_params[1]:.4f}")
    plot_confusion_matrix(cm, ['Healthy', 'Cancer'], save_path="reports/svm_cm.png")
    plot_roc_curve(y_test, y_score, save_path="reports/svm_roc.png")
    plot_precision_recall(y_test, y_score, save_path="reports/svm_pr.png")
    plot_scatter(X_train_fs, y_train, save_path="reports/svm_scatter.png")
    _, wrong_idx = analyze_errors(model, X_test_fs, y_test, save_path="reports/svm_error_matrix.png")

    st.image("reports/svm_cm.png", caption="Confusion Matrix")
    st.image("reports/svm_roc.png", caption="ROC Curve")
    st.image("reports/svm_pr.png", caption="Precision-Recall Curve")
    st.image("reports/svm_scatter.png", caption="PCA Scatter")
    st.image("reports/svm_error_matrix.png", caption=f"Error Matrix ({len(wrong_idx)} errors)")
    st.image("reports/svm+aoa_threshold_curve.png", caption="Threshold Analysis")

# ------------------ Benchmark Models ------------------
if st.button("📊 Compare with Other Models"):
    with st.spinner("🧠 Training benchmark models..."):
        models, best_model_name, best_acc = benchmark_models(
            X_train_fs, y_train,
            use_cv=use_cv_all,
            n_splits=n_splits_all if use_cv_all else 5
        )

        st.subheader("📈 Model Accuracies")
        for name, model in models.items():
            try:
                y_score = model.decision_function(X_test_fs)
            except:
                y_score = model.predict_proba(X_test_fs)[:, 1]

            y_pred = (y_score >= thresh).astype(int)
            acc, report, cm = evaluate_model(model, X_test_fs, y_test, threshold=thresh)
            model_accuracies[name] = acc
            save_classification_report(report, filename=f"{name.lower()}_report.csv")
            threshold_analysis(y_test, y_score, model_name=name, save_dir="reports")

            plot_confusion_matrix(cm, ['Healthy', 'Cancer'], save_path=f"reports/{name.lower()}_cm.png")
            st.write(f"{name}: {acc:.4f}")
            st.image(f"reports/{name.lower()}_cm.png", caption=f"{name} - Confusion Matrix")
            st.image(f"reports/{name.lower()}_threshold_curve.png", caption=f"{name} - Threshold Analysis")

            threshold_txt = f"reports/{name.lower()}_best_threshold.txt"
            if os.path.exists(threshold_txt):
                with open(threshold_txt) as f:
                    best_thresh = f.read().strip().split(":")[-1].strip()
                    st.info(f"{name} → Best Threshold: {best_thresh}")
                    best_thresholds.append([name, best_thresh])

            with st.spinner(f"🔍 SHAP for {name}..."):
                try:
                    bar_path, beeswarm_path = explain_model(name, model, X_test_fs)
                    st.image(bar_path, caption=f"{name} - SHAP Bar")
                    st.image(beeswarm_path, caption=f"{name} - SHAP Beeswarm")
                except Exception as e:
                    st.warning(f"SHAP not supported: {e}")

        plot_roc_all(models, X_test_fs, y_test, save_path="reports/all_models_roc.png")
        st.image("reports/all_models_roc.png", caption="All Models ROC Curve")
        st.success(f"🏆 Best Model: {best_model_name} (Accuracy: {best_acc:.4f})")

# ------------------ Accuracy Table & Chart ------------------
if model_accuracies:
    df_acc = pd.DataFrame(list(model_accuracies.items()), columns=["Model", "Accuracy"])
    st.subheader("📊 Accuracy Table")
    st.dataframe(df_acc.style.format({"Accuracy": "{:.4f}"}).background_gradient(cmap="Greens"))

    st.sidebar.subheader("📈 Accuracy Chart")
    fig_acc, ax_acc = plt.subplots(figsize=(4, 2 + len(df_acc)*0.3))
    sns.barplot(x="Accuracy", y="Model", data=df_acc, palette="Blues_r", ax=ax_acc)
    ax_acc.set_xlim(0, 1)
    ax_acc.set_title("Model Accuracy")
    st.sidebar.pyplot(fig_acc)

# ------------------ Threshold Table ------------------
if best_thresholds:
    st.subheader("📊 Best Thresholds Summary")
    df_thresh = pd.DataFrame(best_thresholds, columns=["Model", "Best Threshold"])
    st.dataframe(df_thresh.style.background_gradient(cmap="coolwarm"))
    csv = df_thresh.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Thresholds", data=csv, file_name="best_thresholds.csv", mime="text/csv")

# ------------------ Optional PCA Scatter ------------------
if st.checkbox("📍 Show Data Scatter Plot (PCA)"):
    plot_scatter(X_train_fs, y_train)

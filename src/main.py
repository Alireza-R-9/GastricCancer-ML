from src.data_loader import load_data, split_data
from src.feature_selector import apply_pca
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
from src.logger import setup_logger
from src.error_analysis import analyze_errors
from src.threshold_analysis import threshold_analysis
import os

def run_pipeline(X_train, X_test, y_train, y_test, label_suffix="", threshold=0.5):
    logger = setup_logger()

    logger.info(f"Training SVM + AOA {label_suffix}...")
    svm_model, best_params = train_svm_aoa(X_train, y_train)
    acc, report, cm = evaluate_model(svm_model, X_test, y_test, threshold=threshold)
    logger.info(f"SVM+AOA Accuracy {label_suffix}: {acc:.4f}, Params: {best_params}")

    save_classification_report(report, filename=f'svm_aoa_report{label_suffix}.csv')
    plot_confusion_matrix(cm, labels=['Healthy', 'Cancer'], save_path=f'reports/svm_confusion_matrix{label_suffix}.png')
    plot_scatter(X_train, y_train, save_path=f'reports/svm_pca_scatter{label_suffix}.png')

    try:
        y_score = svm_model.decision_function(X_test)
    except:
        y_score = svm_model.predict_proba(X_test)[:, 1]

    plot_roc_curve(y_test, y_score, save_path=f'reports/svm_roc{label_suffix}.png')
    plot_precision_recall(y_test, y_score, save_path=f'reports/svm_pr{label_suffix}.png')

    logger.info(f"🔍 تحلیل خطای SVM + AOA {label_suffix}")
    _, wrong_idx = analyze_errors(svm_model, X_test, y_test, class_names=['Healthy', 'Cancer'], save_path=f"reports/svm_error_matrix{label_suffix}.png")
    logger.info(f"تعداد نمونه‌های اشتباه‌شده: {len(wrong_idx)}")

    logger.info(f"📊 آستانه تصمیم‌گیری SVM+AOA {label_suffix}")
    threshold_analysis(y_test, y_score, model_name=f"SVM+AOA{label_suffix}", save_dir="reports")

    return svm_model

def main(threshold=0.5):
    logger = setup_logger()
    os.makedirs("reports", exist_ok=True)

    logger.info("Loading data...")
    X, y = load_data("../data/gastric.csv")
    X_train, X_test, y_train, y_test = split_data(X, y)

    logger.info("Running pipeline without PCA...")
    model_raw = run_pipeline(X_train, X_test, y_train, y_test, label_suffix="_raw", threshold=threshold)

    logger.info("Applying PCA...")
    X_train_pca, X_test_pca = apply_pca(X_train, X_test)

    logger.info("Running pipeline with PCA...")
    model_pca = run_pipeline(X_train_pca, X_test_pca, y_train, y_test, label_suffix="_pca", threshold=threshold)

    logger.info("Training benchmark models on PCA data...")
    benchmarks = benchmark_models(X_train_pca, y_train)
    for name, model in benchmarks.items():
        try:
            y_score = model.decision_function(X_test_pca)
        except:
            y_score = model.predict_proba(X_test_pca)[:, 1]

        threshold_analysis(y_test, y_score, model_name=name, save_dir="reports")

    logger.info("Plotting combined ROC curve...")
    plot_roc_all(benchmarks | {"SVM+AOA (PCA)": model_pca, "SVM+AOA (Raw)": model_raw}, X_test_pca, y_test, save_path="reports/all_models_roc.png")

if __name__ == "__main__":
    main()
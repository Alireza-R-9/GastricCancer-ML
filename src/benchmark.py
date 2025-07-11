from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import pandas as pd


def benchmark_models(X_train, y_train, use_cv=True, n_splits=5):
    """
    Train and tune multiple classification models using GridSearchCV with stratified k-fold cross-validation.
    Saves the best trained models to disk.

    Parameters:
    - X_train: array-like
        Feature matrix for training.
    - y_train: array-like
        Target labels for training.
    - use_cv: bool
        Whether to use cross-validation (default: True)
    - n_splits: int
        Number of CV folds (default: 5)

    Returns:
    - dict
        A dictionary mapping model names to their best trained Pipeline instances.
    - str
        The name of the best model.
    - float
        The accuracy of the best model.
    """

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) if use_cv else None

    models = {
        'KNN': (
            Pipeline([
                ('scaler', StandardScaler()),
                ('knn', KNeighborsClassifier())
            ]),
            {'knn__n_neighbors': [3, 5, 7]}
        ),
        'RF': (
            Pipeline([
                ('rf', RandomForestClassifier())
            ]),
            {'rf__n_estimators': [50, 100]}
        ),
        'DT': (
            Pipeline([
                ('dt', DecisionTreeClassifier())
            ]),
            {'dt__max_depth': [3, 5, None]}
        ),
        'LR': (
            Pipeline([
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(max_iter=1000))
            ]),
            {'lr__C': [0.1, 1.0, 10.0]}
        ),
        'NB': (
            Pipeline([
                ('nb', GaussianNB())
            ]),
            {}  # No hyperparameters to tune for GaussianNB
        ),
        'XGB': (
            Pipeline([
                ('xgb', XGBClassifier(eval_metric='logloss'))
            ]),
            {'xgb__n_estimators': [50, 100]}
        )
    }

    results = {}
    best_model_name = None
    best_accuracy = -1
    logs = []

    for name, (pipeline, param_grid) in models.items():
        if param_grid:
            grid = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring='accuracy')
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            best_params = grid.best_params_
            acc = grid.best_score_ if use_cv else accuracy_score(y_train, best_model.predict(X_train))
        else:
            pipeline.fit(X_train, y_train)
            best_model = pipeline
            best_params = {}
            acc = accuracy_score(y_train, best_model.predict(X_train))

        results[name] = best_model
        joblib.dump(best_model, f"models/{name.lower()}_model.pkl")

        logs.append({"Model": name, "Accuracy": acc, **best_params})
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name

    pd.DataFrame(logs).to_csv("reports/gridsearch_best_params.csv", index=False)
    return results, best_model_name, best_accuracy
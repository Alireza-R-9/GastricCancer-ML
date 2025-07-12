import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer

def add_derived_features(X):
    """
    Create additional derived features from original features.
    Adds log and squared versions of each column.
    """
    X_new = X.copy()
    if isinstance(X_new, pd.DataFrame):
        for col in X_new.columns:
            # Add log1p to avoid log(0) issues
            X_new[f"log_{col}"] = np.log1p(X_new[col])
            X_new[f"{col}_squared"] = X_new[col] ** 2
    else:
        # Convert to DataFrame if input is array
        X_new = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        return add_derived_features(X_new)
    return X_new

def apply_feature_engineering(X_train, X_test, y_train=None, method="pca", n_components=5, k_best=None):
    """
    Apply feature engineering and dimensionality reduction.

    Parameters:
    - X_train, X_test: pd.DataFrame or np.ndarray
        Feature matrices.
    - y_train: np.ndarray or pd.Series
        Required if method='kbest'
    - method: str
        'pca', 'kbest', or 'none'
    - n_components: int
        Number of components for PCA or fallback default.
    - k_best: int or None
        Number of top features to select for SelectKBest.

    Returns:
    - X_train_trans, X_test_trans: np.ndarray
        Transformed datasets.
    """
    # Step 1: Add derived features
    X_train_new = add_derived_features(X_train)
    X_test_new = add_derived_features(X_test)

    # Step 2: Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train_new), columns=X_train_new.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test_new), columns=X_test_new.columns)

    # Step 3: Feature selection / reduction
    if method.lower() == "pca":
        pca = PCA(n_components=n_components)
        X_train_trans = pca.fit_transform(X_train_imputed)
        X_test_trans = pca.transform(X_test_imputed)
    elif method.lower() == "kbest":
        if y_train is None:
            raise ValueError("y_train is required for SelectKBest")
        selector = SelectKBest(score_func=f_classif, k=k_best or n_components)
        X_train_trans = selector.fit_transform(X_train_imputed, y_train)
        X_test_trans = selector.transform(X_test_imputed)
    elif method.lower() == "none":
        X_train_trans, X_test_trans = X_train_imputed.values, X_test_imputed.values
    else:
        raise ValueError(f"Unknown feature selection method: {method}")

    return X_train_trans, X_test_trans

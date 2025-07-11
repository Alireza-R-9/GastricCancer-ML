from sklearn.decomposition import PCA

def apply_pca(X_train, X_test, n_components=5):
    """
    Apply Principal Component Analysis (PCA) to reduce feature dimensionality.

    Parameters:
    - X_train: array-like
        Training feature matrix.
    - X_test: array-like
        Testing feature matrix.
    - n_components: int, optional (default=5)
        Number of principal components to keep.

    Returns:
    - X_train_pca: np.ndarray
        Transformed training data in reduced dimensional space.
    - X_test_pca: np.ndarray
        Transformed testing data using the same PCA transformation.
    """
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca

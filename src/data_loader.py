import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(path: str):
    """
    Load dataset from a CSV file, separate features and target variable,
    and encode target labels.

    Parameters:
    - path: str
        File path to the CSV dataset.

    Returns:
    - X: pd.DataFrame
        Feature matrix.
    - y: np.ndarray
        Encoded target labels (e.g., Cancer=1, Healthy=0).
    """
    df = pd.read_csv(path)

    if 'Diagnosis' not in df.columns:
        raise ValueError("❌ Column 'Diagnosis' not found in dataset.")

    X = df.drop(columns=['Diagnosis'])
    y = LabelEncoder().fit_transform(df['Diagnosis'])  # Cancer=1, Healthy=0

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split dataset into training and testing sets using stratified sampling
    to preserve class distribution.

    Parameters:
    - X: pd.DataFrame or np.ndarray
        Feature matrix.
    - y: np.ndarray
        Target labels.
    - test_size: float, optional (default=0.2)
        Proportion of data to be used as test set.
    - random_state: int, optional (default=42)
        Random seed for reproducibility.

    Returns:
    - X_train, X_test, y_train, y_test: tuple
        Split training and testing data.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

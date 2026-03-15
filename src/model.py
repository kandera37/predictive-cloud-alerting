import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_features_for_model(X: np.ndarray) -> np.ndarray:
    """Flatten sliding-window features into 2D format for sklearn models."""
    num_samples = X.shape[0]
    return X.reshape(num_samples, -1)


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the dataset into train and test parts."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )


def scale_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Scale train and test features using statistics from the training set only."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train a logistic regression classifier for incident prediction."""
    model = LogisticRegression(max_iter=3000)
    model.fit(X_train, y_train)
    return model
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | np.ndarray]:
    """Compute basic classification metrics for incident prediction."""
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": matrix,
    }

def apply_threshold(probabilities: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert predicted probabilities into binary labels using a decision threshold."""
    return (probabilities >= threshold).astype(int)
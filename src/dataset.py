import numpy as np
import pandas as pd


def create_sliding_window_dataset(
    df: pd.DataFrame,
    window_size: int = 20,
    horizon: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding-window features and binary incident targets."""
    feature_columns = ["cpu_usage", "memory_usage", "request_rate", "error_rate"]

    X: list[np.ndarray] = []
    y: list[int] = []

    values = df[feature_columns].values
    incident_values = df["incident"].values

    for start_idx in range(len(df) - window_size - horizon + 1):
        end_idx = start_idx + window_size
        horizon_end_idx = end_idx + horizon

        window = values[start_idx:end_idx]
        future_incident = incident_values[end_idx:horizon_end_idx]

        target = 1 if np.any(future_incident == 1) else 0

        X.append(window)
        y.append(target)

    return np.array(X), np.array(y)
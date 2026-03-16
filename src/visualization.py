import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _find_incident_intervals(incident_series: np.ndarray) -> list[tuple[int, int]]:
    """Find continuous incident intervals from a binary incident array."""
    intervals: list[tuple[int, int]] = []
    in_interval = False
    start_idx = 0

    for idx, value in enumerate(incident_series):
        if value == 1 and not in_interval:
            in_interval = True
            start_idx = idx
        elif value == 0 and in_interval:
            in_interval = False
            intervals.append((start_idx, idx - 1))

    if in_interval:
        intervals.append((start_idx, len(incident_series) - 1))

    return intervals


def plot_metrics_with_incidents(
    df: pd.DataFrame,
    output_path: str = "artifacts/metrics_with_incidents.png",
) -> None:
    """Plot synthetic service metrics and highlight incident intervals."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    metric_configs = [
        ("cpu_usage", "CPU Usage"),
        ("error_rate", "Error Rate"),
        ("request_rate", "Request Rate"),
    ]

    incident_intervals = _find_incident_intervals(df["incident"].values)

    for axis, (column_name, title) in zip(axes, metric_configs):
        axis.plot(df.index, df[column_name], label=column_name)
        axis.set_title(title)
        axis.set_ylabel(column_name)

        for start_idx, end_idx in incident_intervals:
            axis.axvspan(start_idx, end_idx, alpha=0.2)

        axis.legend()

    axes[-1].set_xlabel("Time step")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_prediction_probabilities(
    probabilities: np.ndarray,
    threshold: float,
    output_path: str = "artifacts/predicted_probabilities.png",
) -> None:
    """Plot predicted incident probabilities together with the alert threshold."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(12, 4))
    plt.plot(np.arange(len(probabilities)), probabilities, label="Incident probability")
    plt.axhline(threshold, linestyle="--", label=f"Threshold = {threshold}")
    plt.title("Predicted incident probabilities")
    plt.xlabel("Test sample index")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
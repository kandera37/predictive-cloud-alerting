import numpy as np
import pandas as pd


def generate_synthetic_metrics(
    num_steps: int = 1000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic cloud-service metrics with incident labels."""
    rng = np.random.default_rng(random_seed)

    time = np.arange(num_steps)

    cpu_usage = 0.4 + 0.1 * np.sin(time / 20) + rng.normal(0, 0.03, num_steps)
    memory_usage = 0.5 + 0.08 * np.sin(time / 25 + 1.0) + rng.normal(0, 0.02, num_steps)
    request_rate = 100 + 10 * np.sin(time / 15) + rng.normal(0, 5, num_steps)
    error_rate = 0.01 + rng.normal(0, 0.005, num_steps)

    incident = np.zeros(num_steps, dtype=int)

    incident_starts = rng.choice(np.arange(50, num_steps - 20), size=15, replace=False)

    for start in incident_starts:
        duration = rng.integers(5, 15)
        end = min(start + duration, num_steps)

        cpu_usage[start:end] += rng.uniform(0.25, 0.4)
        memory_usage[start:end] += rng.uniform(0.15, 0.25)
        request_rate[start:end] += rng.uniform(30, 60)
        error_rate[start:end] += rng.uniform(0.05, 0.12)

        incident[start:end] = 1

    cpu_usage = np.clip(cpu_usage, 0.0, 1.0)
    memory_usage = np.clip(memory_usage, 0.0, 1.0)
    error_rate = np.clip(error_rate, 0.0, 1.0)

    df = pd.DataFrame(
        {
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "request_rate": request_rate,
            "error_rate": error_rate,
            "incident": incident,
        }
    )

    return df
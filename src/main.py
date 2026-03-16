from data_generation import generate_synthetic_metrics
from dataset import create_sliding_window_dataset
from evaluation import evaluate_classification, apply_threshold
from visualization import plot_metrics_with_incidents, plot_prediction_probabilities
from model import (
    prepare_features_for_model,
    split_dataset,
    scale_features,
    train_logistic_regression,
)


def main() -> None:
    """Run a baseline predictive alerting experiment on synthetic cloud metrics."""
    df = generate_synthetic_metrics(num_steps=300)

    print(df.head())
    print()
    print("Raw dataframe shape:", df.shape)
    print("Incident count:", df["incident"].sum())

    X, y = create_sliding_window_dataset(df, window_size=20, horizon=5)

    print()
    print("Sliding-window dataset created.")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("First target values:", y[:10])

    X_flat = prepare_features_for_model(X)

    print()
    print("Flattened X shape:", X_flat.shape)

    X_train, X_test, y_train, y_test = split_dataset(X_flat, y, test_size=0.2)

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    print()
    print("Scaled train shape:", X_train_scaled.shape)
    print("Scaled test shape:", X_test_scaled.shape)

    model = train_logistic_regression(X_train_scaled, y_train)

    incident_probabilities = model.predict_proba(X_test_scaled)[:, 1]
    threshold = 0.7
    y_pred = apply_threshold(incident_probabilities, threshold=threshold)

    metrics = evaluate_classification(y_test, y_pred)

    print()
    print("Decision threshold:", threshold)
    print("First 10 incident probabilities:", incident_probabilities[:10])

    print()
    print("Evaluation metrics:")
    print("Precision:", metrics["precision"])
    print("Recall:", metrics["recall"])
    print("F1:", metrics["f1"])
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])
    plot_metrics_with_incidents(df)
    plot_prediction_probabilities(
        probabilities=incident_probabilities,
        threshold=threshold,
    )

    print()
    print("Plots saved to artifacts/")


if __name__ == "__main__":
    main()
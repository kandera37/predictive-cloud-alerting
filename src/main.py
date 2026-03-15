from data_generation import generate_synthetic_metrics
from dataset import create_sliding_window_dataset
from evaluation import evaluate_classification
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

    y_pred = model.predict(X_test_scaled)
    metrics = evaluate_classification(y_test, y_pred)

    print()
    print("Evaluation metrics:")
    print("Precision:", metrics["precision"])
    print("Recall:", metrics["recall"])
    print("F1:", metrics["f1"])
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])


if __name__ == "__main__":
    main()
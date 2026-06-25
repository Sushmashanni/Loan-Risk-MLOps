import joblib
import pandas as pd
import yaml
import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def load_params():
    with open("params.yaml", "r") as file:
        return yaml.safe_load(file)


def load_test_data():
    return pd.read_csv("data/test.csv")


def load_model():
    return joblib.load("models/model.pkl")


def main():

    params = load_params()

    df = load_test_data()

    X_test = df.drop("loan_status", axis=1)
    y_test = df["loan_status"]

    model = load_model()

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("\n========== MODEL EVALUATION ==========\n")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions))

    # -------------------------
    # MLflow Logging
    # -------------------------

    mlflow.set_experiment("Loan Risk Prediction")

    with mlflow.start_run():

        mlflow.log_param(
            "n_estimators",
            params["training"]["n_estimators"]
        )

        mlflow.log_param(
            "max_depth",
            params["training"]["max_depth"]
        )

        mlflow.log_param(
            "min_samples_split",
            params["training"]["min_samples_split"]
        )

        mlflow.log_param(
            "min_samples_leaf",
            params["training"]["min_samples_leaf"]
        )

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model"
        )

        print("\nMLflow logging completed successfully.")


if __name__ == "__main__":
    main()
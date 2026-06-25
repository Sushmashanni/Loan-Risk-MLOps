import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def load_test_data():
    return pd.read_csv("data/test.csv")


def load_model():
    return joblib.load("models/model.pkl")


def main():
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


if __name__ == "__main__":
    main()
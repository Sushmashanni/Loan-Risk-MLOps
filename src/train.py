import os
import joblib
import yaml
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def load_params():
    with open("params.yaml", "r") as file:
        return yaml.safe_load(file)


def load_data():
    return pd.read_csv("data/train.csv")


def train_model(X, y, params):
    model = RandomForestClassifier(
        n_estimators=params["training"]["n_estimators"],
        max_depth=params["training"]["max_depth"],
        min_samples_split=params["training"]["min_samples_split"],
        min_samples_leaf=params["training"]["min_samples_leaf"],
        random_state=params["model"]["random_state"]
    )

    model.fit(X, y)
    return model


def save_model(model):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")


def main():
    params = load_params()

    df = load_data()

    X = df.drop("loan_status", axis=1)
    y = df["loan_status"]

    model = train_model(X, y, params)

    save_model(model)

    print("Model trained successfully.")
    print("Model saved to models/model.pkl")


if __name__ == "__main__":
    main()
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_params():
    with open("params.yaml", "r") as file:
        return yaml.safe_load(file)


def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def preprocess_data(df):
    # Remove unnecessary column
    df = df.drop(columns=["loan_id"])

    # Encode categorical columns
    categorical_columns = ["education", "self_employed", "loan_status"]

    for column in categorical_columns:
        df[column] = df[column].astype(str).str.strip()
        df[column] = df[column].astype("category").cat.codes

    return df


def split_data(df, test_size, random_state):
    X = df.drop("loan_status", axis=1)
    y = df["loan_status"]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def main():
    params = load_params()

    df = load_data(params["data"]["path"])

    print("Original Shape:", df.shape)

    df = preprocess_data(df)

    print("Processed Shape:", df.shape)

    X_train, X_test, y_train, y_test = split_data(
        df,
        params["model"]["test_size"],
        params["model"]["random_state"]
    )

    print("\nTrain Shape :", X_train.shape)
    print("Test Shape  :", X_test.shape)

        # Combine features and target
    train_df = X_train.copy()
    train_df["loan_status"] = y_train

    test_df = X_test.copy()
    test_df["loan_status"] = y_test

    # Save processed datasets
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    print("\nProcessed datasets saved successfully.")


if __name__ == "__main__":
    main()
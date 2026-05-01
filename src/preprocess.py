"""
preprocess.py — Data loading, exploration, and preprocessing for credit card fraud detection.

Handles:
- Loading raw CSV data
- Exploratory data analysis (class distribution, feature stats)
- Feature scaling (StandardScaler on Time & Amount)
- Stratified train/test split
- Optional SMOTE oversampling for training set
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib


def load_data(data_path: str) -> pd.DataFrame:
    """Load the credit card fraud dataset from CSV."""
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def explore_data(df: pd.DataFrame) -> dict:
    """Print and return key dataset statistics."""
    stats = {}

    # Class distribution
    class_counts = df["Class"].value_counts()
    fraud_ratio = class_counts[1] / len(df)
    stats["total_samples"] = len(df)
    stats["fraud_count"] = int(class_counts[1])
    stats["legit_count"] = int(class_counts[0])
    stats["fraud_ratio"] = fraud_ratio

    print("\n=== Dataset Overview ===")
    print(f"Total transactions : {stats['total_samples']:,}")
    print(f"Legitimate (0)     : {stats['legit_count']:,} ({1 - fraud_ratio:.4%})")
    print(f"Fraudulent (1)     : {stats['fraud_count']:,} ({fraud_ratio:.4%})")
    print(f"\nImbalance ratio    : 1:{stats['legit_count'] // stats['fraud_count']}")

    # Basic stats
    print("\n=== Feature Statistics ===")
    print(df[["Time", "Amount"]].describe().round(2))

    # Missing values
    missing = df.isnull().sum().sum()
    stats["missing_values"] = missing
    print(f"\nMissing values     : {missing}")

    return stats


def preprocess(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    apply_smote: bool = True,
    smote_strategy: float = 0.5,
) -> dict:
    """
    Preprocess the dataset:
    1. Scale 'Time' and 'Amount' features
    2. Stratified train/test split
    3. Optionally apply SMOTE to the training set

    Args:
        df: Raw dataframe
        test_size: Fraction held out for testing
        random_state: Reproducibility seed
        apply_smote: Whether to oversample the minority class
        smote_strategy: Target ratio of minority to majority after SMOTE

    Returns:
        Dictionary with X_train, X_test, y_train, y_test, scaler, and metadata
    """
    # Separate features and target
    X = df.drop("Class", axis=1).copy()
    y = df["Class"].astype(int)

    # Scale Time and Amount (V1-V28 are already PCA-transformed and scaled)
    scaler = StandardScaler()
    X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

    # Stratified split — preserves the fraud ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\n=== Train/Test Split ===")
    print(f"Train set: {X_train.shape[0]:,} samples ({y_train.sum():,} fraud)")
    print(f"Test set : {X_test.shape[0]:,} samples ({y_test.sum():,} fraud)")

    result = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "smote_applied": False,
    }

    # Apply SMOTE to training data only
    if apply_smote:
        print(f"\n=== Applying SMOTE (strategy={smote_strategy}) ===")
        smote = SMOTE(sampling_strategy=smote_strategy, random_state=random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        print(f"Before SMOTE: {X_train.shape[0]:,} samples ({y_train.sum():,} fraud)")
        print(f"After SMOTE : {X_train_res.shape[0]:,} samples ({y_train_res.sum():,} fraud)")

        result["X_train"] = X_train_res
        result["y_train"] = y_train_res
        result["smote_applied"] = True

    return result


def save_processed_data(result: dict, output_dir: str):
    """Save processed arrays and scaler to disk."""
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "X_train.npy"), result["X_train"].values if hasattr(result["X_train"], "values") else result["X_train"])
    np.save(os.path.join(output_dir, "X_test.npy"), result["X_test"].values)
    np.save(os.path.join(output_dir, "y_train.npy"), result["y_train"].values if hasattr(result["y_train"], "values") else result["y_train"])
    np.save(os.path.join(output_dir, "y_test.npy"), result["y_test"].values)
    joblib.dump(result["scaler"], os.path.join(output_dir, "scaler.joblib"))

    # Save feature names
    feature_names = result["X_test"].columns.tolist()
    joblib.dump(feature_names, os.path.join(output_dir, "feature_names.joblib"))

    print(f"\nProcessed data saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess credit card fraud data")
    parser.add_argument("--data-path", type=str, default="creditcard.csv", help="Path to raw CSV")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for processed data")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--no-smote", action="store_true", help="Disable SMOTE oversampling")
    parser.add_argument("--smote-strategy", type=float, default=0.5, help="SMOTE target minority/majority ratio")
    args = parser.parse_args()

    df = load_data(args.data_path)
    explore_data(df)
    result = preprocess(df, test_size=args.test_size, apply_smote=not args.no_smote, smote_strategy=args.smote_strategy)
    save_processed_data(result, args.output_dir)


if __name__ == "__main__":
    main()

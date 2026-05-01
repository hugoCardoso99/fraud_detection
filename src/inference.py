"""
inference.py — Run fraud predictions on new transactions.

Loads a trained model and scaler, applies preprocessing,
and predicts fraud with a configurable decision threshold.
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib


class FraudDetector:
    """Production-style fraud detection inference wrapper."""

    def __init__(self, model_path: str, scaler_path: str, threshold: float = 0.5):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.threshold = threshold
        self.feature_names = None

        # Try to load feature names
        feature_path = os.path.join(os.path.dirname(scaler_path), "feature_names.joblib")
        if os.path.exists(feature_path):
            self.feature_names = joblib.load(feature_path)

        print(f"Loaded model: {type(self.model).__name__}")
        print(f"Decision threshold: {self.threshold}")

    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        """Apply the same preprocessing as training."""
        X = df.copy()

        # Scale Time and Amount
        if "Time" in X.columns and "Amount" in X.columns:
            X[["Time", "Amount"]] = self.scaler.transform(X[["Time", "Amount"]])

        # Ensure feature order matches training
        if self.feature_names:
            X = X[self.feature_names]

        return X.values

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict fraud probabilities and labels for new transactions.

        Returns a DataFrame with:
        - fraud_probability: model's confidence that the transaction is fraud
        - is_fraud: binary prediction at the configured threshold
        - risk_level: Low / Medium / High based on probability
        """
        X = self.preprocess(df)
        probas = self.model.predict_proba(X)[:, 1]

        results = df.copy()
        results["fraud_probability"] = probas
        results["is_fraud"] = (probas >= self.threshold).astype(int)

        # Risk stratification
        results["risk_level"] = pd.cut(
            probas,
            bins=[0, 0.3, 0.7, 1.0],
            labels=["Low", "Medium", "High"],
            include_lowest=True,
        )

        return results

    def predict_single(self, transaction: dict) -> dict:
        """Predict fraud for a single transaction."""
        df = pd.DataFrame([transaction])
        result = self.predict(df).iloc[0]
        return {
            "fraud_probability": float(result["fraud_probability"]),
            "is_fraud": bool(result["is_fraud"]),
            "risk_level": str(result["risk_level"]),
        }


def main():
    parser = argparse.ArgumentParser(description="Run fraud inference")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model .joblib")
    parser.add_argument("--scaler-path", type=str, default="data/scaler.joblib")
    parser.add_argument("--input-csv", type=str, required=True, help="CSV file with transactions")
    parser.add_argument("--output-csv", type=str, default="predictions.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    detector = FraudDetector(args.model_path, args.scaler_path, args.threshold)

    # Load and predict
    df = pd.read_csv(args.input_csv)

    # Remove Class column if present (we're predicting, not evaluating)
    if "Class" in df.columns:
        df = df.drop("Class", axis=1)

    results = detector.predict(df)

    # Summary
    n_fraud = results["is_fraud"].sum()
    n_total = len(results)
    print(f"\n=== Inference Results ===")
    print(f"Total transactions : {n_total:,}")
    print(f"Flagged as fraud   : {n_fraud:,} ({n_fraud/n_total:.2%})")
    print(f"\nRisk distribution:")
    print(results["risk_level"].value_counts().to_string())

    # Save
    results.to_csv(args.output_csv, index=False)
    print(f"\nPredictions saved to: {args.output_csv}")


if __name__ == "__main__":
    main()

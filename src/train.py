"""
train.py — Model training for credit card fraud detection.

Trains multiple classifiers and saves the best one:
- Logistic Regression (baseline, fast, interpretable)
- Random Forest (handles non-linearity)
- XGBoost (gradient boosting, strong on tabular data)

Includes stratified cross-validation with relevant metrics for imbalanced data.
"""

import os
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier
import joblib


def get_models(use_smote: bool = True) -> dict:
    """
    Return a dictionary of models to train.

    Args:
        use_smote: If True, the training data has already been resampled via SMOTE,
                   so we set class_weight=None to avoid double-correcting for imbalance.
                   If False, we rely on class_weight='balanced' to handle imbalance
                   inside the loss function instead.

    NOTE: You should use one strategy or the other -- not both.
    Stacking SMOTE + balanced class weights double-penalizes the majority class.
    """
    if use_smote:
        # SMOTE already rebalanced the training data -- no need for class weights
        return {
            "LogisticRegression": LogisticRegression(
                max_iter=1000,
                class_weight=None,
                random_state=42,
                solver="lbfgs",
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=100,
                class_weight=None,
                random_state=42,
                n_jobs=-1,
            ),
            "XGBoost": XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=1,
                eval_metric="aucpr",
                random_state=42,
                n_jobs=-1,
            ),
        }
    else:
        # No SMOTE -- let the model handle imbalance via class weights
        return {
            "LogisticRegression": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
                solver="lbfgs",
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            "XGBoost": XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=1,
                eval_metric="aucpr",
                random_state=42,
                n_jobs=-1,
            ),
        }


def cross_validate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    models: dict,
    cv_folds: int = 5,
) -> dict:
    """
    Run stratified k-fold cross-validation for each model.

    Reports metrics that matter for imbalanced classification:
    - AUPRC (average_precision): best single metric for imbalanced data
    - AUROC (roc_auc): overall discrimination ability
    - F1: harmonic mean of precision/recall
    - Precision & Recall: the core tradeoff
    """
    scoring = {
        "auprc": "average_precision",
        "auroc": "roc_auc",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = {}

    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Cross-validating: {name}")
        print(f"{'='*50}")

        cv_results = cross_validate(
            model, X_train, y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1,
        )

        results[name] = cv_results

        for metric in scoring:
            key = f"test_{metric}"
            scores = cv_results[key]
            print(f"  {metric:>12s}: {scores.mean():.4f} (+/- {scores.std():.4f})")

    return results


def train_final_model(
    model_name: str,
    models: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
):
    """Train the selected model on the full training set."""
    model = models[model_name]
    print(f"\nTraining final {model_name} on full training set...")
    model.fit(X_train, y_train)
    print("Done.")
    return model


def save_model(model, model_name: str, output_dir: str):
    """Save the trained model to disk."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{model_name}.joblib")
    joblib.dump(model, path)
    print(f"Model saved to: {path}")
    return path


def select_best_model(cv_results: dict, metric: str = "auprc") -> str:
    """Select the best model based on mean cross-validation score."""
    key = f"test_{metric}"
    best_name = max(cv_results, key=lambda name: cv_results[name][key].mean())
    best_score = cv_results[best_name][key].mean()
    print(f"\nBest model by {metric}: {best_name} ({best_score:.4f})")
    return best_name


def main():
    parser = argparse.ArgumentParser(description="Train fraud detection models")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory with processed data")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--model", type=str, default=None,
                        help="Force a specific model (LogisticRegression, RandomForest, XGBoost)")
    parser.add_argument("--train-all", action="store_true", help="Train and save all models")
    parser.add_argument("--no-smote", action="store_true",
                        help="Indicate training data was NOT resampled with SMOTE. "
                             "Models will use class_weight='balanced' instead.")
    args = parser.parse_args()

    # Load processed data
    X_train = np.load(os.path.join(args.data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(args.data_dir, "y_train.npy"))
    print(f"Loaded training data: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"Fraud in training set: {y_train.sum():,.0f} ({y_train.mean():.2%})")

    use_smote = not args.no_smote
    smote_suffix = "smote" if use_smote else "no_smote"

    if use_smote:
        print("Mode: SMOTE resampled data -> class_weight=None")
    else:
        print("Mode: Original imbalanced data -> class_weight='balanced'")

    models = get_models(use_smote=use_smote)

    # If no SMOTE, compute scale_pos_weight for XGBoost
    if not use_smote and "XGBoost" in models:
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        spw = n_neg / n_pos
        models["XGBoost"].set_params(scale_pos_weight=spw)
        print(f"XGBoost scale_pos_weight set to {spw:.1f} (={n_neg}:{n_pos})")

    # Cross-validate all models
    cv_results = cross_validate_models(X_train, y_train, models, cv_folds=args.cv_folds)

    # All training outputs go into models/train_smote/ or models/train_no_smote/
    train_output_dir = os.path.join(args.model_dir, "train_" + smote_suffix)
    os.makedirs(train_output_dir, exist_ok=True)

    if args.train_all:
        for name in models:
            model = train_final_model(name, models, X_train, y_train)
            save_model(model, name, train_output_dir)
    else:
        model_name = args.model if args.model else select_best_model(cv_results)
        model = train_final_model(model_name, models, X_train, y_train)
        save_model(model, model_name, train_output_dir)

    # Save CV results in the same folder
    cv_path = os.path.join(train_output_dir, "cv_results.joblib")
    joblib.dump(cv_results, cv_path)
    print("\nCV results saved to: " + cv_path)


if __name__ == "__main__":
    main()

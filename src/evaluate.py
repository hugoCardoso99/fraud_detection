"""
evaluate.py — Comprehensive evaluation for fraud detection models.

Key analyses for interview discussion:
1. Precision vs Recall tradeoff (why it matters for fraud)
2. Threshold tuning (finding the optimal operating point)
3. ROC curve vs PR curve (why PR is better for imbalanced data)
4. Confusion matrices at different thresholds
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
import joblib


# ─── 1. PRECISION VS RECALL TRADEOFF ────────────────────────────────────────

def plot_precision_recall_tradeoff(y_true, y_scores, output_dir):
    """
    Visualize how precision and recall change with the decision threshold.

    KEY INTERVIEW INSIGHT:
    - In fraud detection, a false negative (missed fraud) is far more costly
      than a false positive (flagging a legit transaction).
    - High recall = catch most fraud, but more false alarms.
    - High precision = fewer false alarms, but miss some fraud.
    - The business decides the acceptable tradeoff.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, precisions[:-1], "b-", label="Precision", linewidth=2)
    ax.plot(thresholds, recalls[:-1], "r-", label="Recall", linewidth=2)

    # Find and mark the threshold where F1 is maximized
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    ax.axvline(x=best_threshold, color="green", linestyle="--", alpha=0.7,
               label=f"Best F1 threshold = {best_threshold:.3f}")
    ax.scatter([best_threshold], [precisions[best_idx]], color="green", s=100, zorder=5)
    ax.scatter([best_threshold], [recalls[best_idx]], color="green", s=100, zorder=5)

    ax.set_xlabel("Decision Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Precision vs Recall Tradeoff", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    path = os.path.join(output_dir, "precision_recall_tradeoff.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    return best_threshold, f1_scores[best_idx]


# ─── 2. THRESHOLD TUNING ────────────────────────────────────────────────────

def tune_threshold(y_true, y_scores, output_dir):
    """
    Find optimal thresholds for different business objectives.

    KEY INTERVIEW INSIGHT:
    - Default threshold of 0.5 is almost never optimal for imbalanced data.
    - Different objectives need different thresholds:
      * Max F1: balanced precision/recall
      * Max F2: prioritize recall (catch more fraud)
      * Target recall >= 0.95: ensure we catch 95%+ of fraud
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

    # F1-optimal threshold
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    best_f1_idx = np.argmax(f1_scores)

    # F2-optimal threshold (weights recall 2x more than precision)
    f2_scores = 5 * (precisions[:-1] * recalls[:-1]) / (4 * precisions[:-1] + recalls[:-1] + 1e-8)
    best_f2_idx = np.argmax(f2_scores)

    # Threshold for target recall >= 0.95
    valid = recalls[:-1] >= 0.95
    if valid.any():
        # Among thresholds with recall >= 0.95, pick highest precision
        candidates = np.where(valid)[0]
        best_recall95_idx = candidates[np.argmax(precisions[:-1][candidates])]
    else:
        best_recall95_idx = 0

    results = {
        "best_f1": {
            "threshold": float(thresholds[best_f1_idx]),
            "precision": float(precisions[best_f1_idx]),
            "recall": float(recalls[best_f1_idx]),
            "f1": float(f1_scores[best_f1_idx]),
        },
        "best_f2": {
            "threshold": float(thresholds[best_f2_idx]),
            "precision": float(precisions[best_f2_idx]),
            "recall": float(recalls[best_f2_idx]),
            "f2": float(f2_scores[best_f2_idx]),
        },
        "recall_95": {
            "threshold": float(thresholds[best_recall95_idx]),
            "precision": float(precisions[best_recall95_idx]),
            "recall": float(recalls[best_recall95_idx]),
        },
    }

    print("\n=== Threshold Tuning Results ===")
    for strategy, vals in results.items():
        print(f"\n  {strategy}:")
        for k, v in vals.items():
            print(f"    {k}: {v:.4f}")

    # Visualization: metrics across thresholds
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (name, vals) in zip(axes, results.items()):
        t = vals["threshold"]
        y_pred = (y_scores >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
        ax.set_title(f"{name}\n(threshold={t:.3f})", fontsize=11)
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

    plt.suptitle("Confusion Matrices at Different Thresholds", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "threshold_confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    return results


# ─── 3. ROC CURVE VS PR CURVE ───────────────────────────────────────────────

def plot_roc_vs_pr_curve(y_true, y_scores, output_dir):
    """
    Compare ROC and Precision-Recall curves side by side.

    KEY INTERVIEW INSIGHT:
    - ROC curve can be misleadingly optimistic on imbalanced data because
      the FPR denominator (true negatives) is huge, making even many false
      positives look like a tiny FPR.
    - PR curve focuses on the positive (fraud) class and is more informative
      when the minority class is what we care about.
    - AUPRC (Average Precision) is a better single-number metric than AUROC
      for imbalanced problems.
    """
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # PR curve
    precisions, recalls, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)

    # Baseline for PR curve = prevalence of positive class
    baseline = y_true.mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ROC
    ax1.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random classifier")
    ax1.fill_between(fpr, tpr, alpha=0.1, color="blue")
    ax1.set_xlabel("False Positive Rate", fontsize=12)
    ax1.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax1.set_title("ROC Curve", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # PR
    ax2.plot(recalls, precisions, "r-", linewidth=2, label=f"PR (AUPRC = {pr_auc:.4f})")
    ax2.axhline(y=baseline, color="k", linestyle="--", alpha=0.5,
                label=f"Random (prevalence = {baseline:.4f})")
    ax2.fill_between(recalls, precisions, alpha=0.1, color="red")
    ax2.set_xlabel("Recall", fontsize=12)
    ax2.set_ylabel("Precision", fontsize=12)
    ax2.set_title("Precision-Recall Curve", fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("ROC vs Precision-Recall Curve", fontsize=15, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "roc_vs_pr_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    return {"auroc": roc_auc, "auprc": pr_auc}


# ─── 4. FULL EVALUATION REPORT ──────────────────────────────────────────────

def full_evaluation(y_true, y_scores, threshold, output_dir):
    """Generate a complete evaluation at a given threshold."""
    y_pred = (y_scores >= threshold).astype(int)

    print(f"\n{'='*50}")
    print(f"Classification Report (threshold={threshold:.3f})")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, target_names=["Legitimate", "Fraud"], digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"True Positives  (caught fraud)     : {tp:,}")
    print(f"False Negatives (missed fraud)      : {fn:,}")
    print(f"False Positives (false alarms)      : {fp:,}")
    print(f"True Negatives  (correct legit)     : {tn:,}")
    print(f"\nFraud catch rate (recall)           : {tp/(tp+fn):.4f}")
    print(f"Alert precision                     : {tp/(tp+fp):.4f}")

    # Detailed confusion matrix plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd", ax=ax,
                xticklabels=["Legitimate", "Fraud"],
                yticklabels=["Legitimate", "Fraud"])
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_title(f"Confusion Matrix (threshold={threshold:.3f})", fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    return {
        "threshold": threshold,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate fraud detection model")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model .joblib")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory for plots and results")
    parser.add_argument("--threshold", type=float, default=None, help="Custom threshold (default: auto-tuned)")
    args = parser.parse_args()

    # Load data and model
    X_test = np.load(os.path.join(args.data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(args.data_dir, "y_test.npy"))
    model = joblib.load(args.model_path)

    # Derive model name from filename and smote strategy from parent folder.
    # Expected path: models/train_<smote|no_smote>/<ModelName>.joblib
    # e.g. models/train_smote/XGBoost.joblib → output to models/XGBoost_smote/
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    parent_dir = os.path.basename(os.path.dirname(args.model_path))

    # Extract smote suffix from the training folder name
    if "no_smote" in parent_dir:
        smote_suffix = "no_smote"
    elif "smote" in parent_dir:
        smote_suffix = "smote"
    else:
        smote_suffix = "unknown"

    # Save evaluation outputs to: models/<ModelName>_<smote|no_smote>/
    output_dir = os.path.join(args.output_dir, f"{model_name}_{smote_suffix}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Evaluating: {model_name}")
    print(f"Test set: {X_test.shape[0]:,} samples ({y_test.sum():,.0f} fraud)")
    print(f"Output dir: {output_dir}")

    # Get probability scores for the positive class
    y_scores = model.predict_proba(X_test)[:, 1]

    # Run all analyses
    print("\n" + "=" * 60)
    print("1. PRECISION VS RECALL TRADEOFF")
    print("=" * 60)
    best_threshold, best_f1 = plot_precision_recall_tradeoff(y_test, y_scores, output_dir)

    print("\n" + "=" * 60)
    print("2. THRESHOLD TUNING")
    print("=" * 60)
    threshold_results = tune_threshold(y_test, y_scores, output_dir)

    print("\n" + "=" * 60)
    print("3. ROC VS PR CURVE")
    print("=" * 60)
    curve_metrics = plot_roc_vs_pr_curve(y_test, y_scores, output_dir)

    print("\n" + "=" * 60)
    print("4. FULL EVALUATION")
    print("=" * 60)
    threshold = args.threshold if args.threshold else threshold_results["best_f1"]["threshold"]
    eval_results = full_evaluation(y_test, y_scores, threshold, output_dir)

    # Save all results
    all_results = {
        "model_name": model_name,
        "curve_metrics": curve_metrics,
        "threshold_tuning": threshold_results,
        "evaluation": eval_results,
    }
    results_path = os.path.join(output_dir, "evaluation_results.joblib")
    joblib.dump(all_results, results_path)
    print(f"\nAll results saved to: {results_path}")


if __name__ == "__main__":
    main()

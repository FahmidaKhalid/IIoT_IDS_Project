# src/evaluate.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize
from scipy.stats import wilcoxon
from itertools import combinations
from .config import DATASETS, MODELS_DIR, RESULTS_DIR

MODEL_NAMES = [
    "NaiveBayes",
    "LogisticRegression",
    "KNN",
    "SVM",
    "RandomForest",
    "XGBoost",
    "MLP"
]


# ─────────────────────────────────────────────
# 1. CONFUSION MATRIX
# ─────────────────────────────────────────────
def plot_confusion_matrix(y_test, y_pred, dataset_name, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{dataset_name} — {model_name}\nConfusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    path = os.path.join(RESULTS_DIR, f"{dataset_name}_{model_name}_confusion_matrix.png")
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"  Confusion matrix saved: {path}")


# ─────────────────────────────────────────────
# 2. ROC CURVE
# ─────────────────────────────────────────────
def plot_roc_curve(model, X_test, y_test, dataset_name, model_name):
    """
    Plots ROC curve for binary or multiclass classification.
    """
    classes = np.unique(y_test)

    # Binary case
    if len(classes) == 2:
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            print(f"  ROC skipped for {model_name} (no probability support)")
            return None

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="darkorange", lw=2,
                 label=f"ROC curve (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{dataset_name} — {model_name}\nROC Curve")
        plt.legend(loc="lower right")
        path = os.path.join(RESULTS_DIR, f"{dataset_name}_{model_name}_roc_curve.png")
        plt.savefig(path, bbox_inches="tight")
        plt.show()
        plt.close()
        print(f"  ROC curve saved: {path}  (AUC={roc_auc:.4f})")
        return roc_auc

    else:
        print(f"  Multiclass ROC skipped for {model_name}")
        return None


# ─────────────────────────────────────────────
# 3. STATISTICAL SIGNIFICANCE TEST
#    Wilcoxon Signed-Rank Test between best model
#    and every other model (per dataset)
# ─────────────────────────────────────────────
def run_statistical_tests(dataset_name, model_predictions, y_test):
    """
    Compares all model pairs using Wilcoxon Signed-Rank Test.
    model_predictions: dict {model_name: y_pred array}
    """
    print(f"\n--- Statistical Tests ({dataset_name}) ---")
    print(f"{'Model A':<20} {'Model B':<20} {'p-value':<12} {'Significant?'}")
    print("-" * 65)

    # Convert predictions to binary correct/incorrect arrays
    correctness = {}
    for name, y_pred in model_predictions.items():
        correctness[name] = (y_pred == y_test).astype(int)

    results = []
    model_list = list(model_predictions.keys())

    for m1, m2 in combinations(model_list, 2):
        c1 = correctness[m1]
        c2 = correctness[m2]

        # Wilcoxon test requires differences — skip if identical
        if np.array_equal(c1, c2):
            print(f"{m1:<20} {m2:<20} {'N/A':<12} Identical predictions")
            continue

        try:
            stat, p_val = wilcoxon(c1, c2)
            significant = "YES ✓" if p_val < 0.05 else "NO"
            print(f"{m1:<20} {m2:<20} {p_val:<12.6f} {significant}")
            results.append({
                "Dataset": dataset_name,
                "Model_A": m1,
                "Model_B": m2,
                "p_value": p_val,
                "Significant (p<0.05)": significant
            })
        except Exception as e:
            print(f"{m1:<20} {m2:<20} Error: {e}")

    return results


# ─────────────────────────────────────────────
# 4. ACCURACY BAR CHART
# ─────────────────────────────────────────────
def plot_accuracy_comparison(summary_df, dataset_name):
    subset = summary_df[summary_df["Dataset"] == dataset_name]
    plt.figure(figsize=(10, 5))
    bars = plt.bar(subset["Model"], subset["Accuracy"] * 100,
                   color="steelblue", edgecolor="black")
    for bar, val in zip(bars, subset["Accuracy"]):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 f"{val*100:.2f}%", ha="center", va="bottom", fontsize=9)
    plt.title(f"Model Accuracy Comparison — {dataset_name}")
    plt.xlabel("Model")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 110)
    plt.xticks(rotation=15)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{dataset_name}_accuracy_comparison.png")
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"  Accuracy chart saved: {path}")


# ─────────────────────────────────────────────
# 5. MAIN EVALUATION PIPELINE
# ─────────────────────────────────────────────
def evaluate_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_results = []
    all_stat_results = []

    for dataset_name, paths in DATASETS.items():
        print(f"\n{'='*55}")
        print(f"  Dataset: {dataset_name}")
        print(f"{'='*55}")

        X_test = np.load(paths["X_test"])
        y_test = np.load(paths["y_test"], allow_pickle=True)

        model_predictions = {}   # for statistical test

        for model_name in MODEL_NAMES:
            model_path = os.path.join(MODELS_DIR, f"{model_name}_{dataset_name}.joblib")
            if not os.path.exists(model_path):
                print(f"  Model {model_name} not found — skipping.")
                continue

            print(f"\n--- Model: {model_name} ---")
            model = joblib.load(model_path)
            y_pred = model.predict(X_test)

            model_predictions[model_name] = y_pred

            acc  = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec  = recall_score(y_test, y_pred,    average="weighted", zero_division=0)
            f1   = f1_score(y_test, y_pred,        average="weighted", zero_division=0)

            print(f"  Accuracy : {acc:.4f}")
            print(f"  Precision: {prec:.4f}")
            print(f"  Recall   : {rec:.4f}")
            print(f"  F1-score : {f1:.4f}")
            print("\n  Classification Report:")
            print(classification_report(y_test, y_pred, zero_division=0))

            # Confusion Matrix
            plot_confusion_matrix(y_test, y_pred, dataset_name, model_name)

            # ROC Curve
            roc_auc = plot_roc_curve(model, X_test, y_test, dataset_name, model_name)

            summary_results.append({
                "Dataset":   dataset_name,
                "Model":     model_name,
                "Accuracy":  acc,
                "Precision": prec,
                "Recall":    rec,
                "F1-score":  f1,
                "ROC-AUC":   roc_auc if roc_auc else "N/A",
            })

        # Statistical Tests for this dataset
        if len(model_predictions) >= 2:
            stat_results = run_statistical_tests(dataset_name, model_predictions, y_test)
            all_stat_results.extend(stat_results)

    # ── Save summary CSV ──
    summary_df = pd.DataFrame(summary_results)
    summary_csv = os.path.join(RESULTS_DIR, "evaluation_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n Summary saved: {summary_csv}")

    # ── Save statistical test results ──
    if all_stat_results:
        stat_df = pd.DataFrame(all_stat_results)
        stat_csv = os.path.join(RESULTS_DIR, "statistical_test_results.csv")
        stat_df.to_csv(stat_csv, index=False)
        print(f" Statistical test results saved: {stat_csv}")

    # ── Accuracy comparison charts ──
    for dataset_name in summary_df["Dataset"].unique():
        plot_accuracy_comparison(summary_df, dataset_name)

    print("\n All evaluations complete!")


def main():
    evaluate_all()


if __name__ == "__main__":
    main()
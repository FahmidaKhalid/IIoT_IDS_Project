import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import label_binarize
from src.config import DATASETS, MODELS_DIR, RESULTS_DIR

def evaluate_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_results = []

    for dataset_name, paths in DATASETS.items():
        print(f"\n=== Dataset: {dataset_name} ===")

        X_test = np.load(paths["X_test"])
        y_test = np.load(paths["y_test"], allow_pickle=True)

        # Binarize labels for ROC/PR curves if multiclass
        classes = np.unique(y_test)
        y_test_bin = label_binarize(y_test, classes=classes)
        n_classes = y_test_bin.shape[1]

        for model_name in ["RandomForest", "XGBoost", "LightGBM"]:
            model_path = os.path.join(MODELS_DIR, f"{dataset_name}_{model_name}.joblib")
            if not os.path.exists(model_path):
                print(f"Model {model_name} not found for {dataset_name}, skipping.")
                continue

            model = joblib.load(model_path)
            y_pred = model.predict(X_test)
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            print(f"\n--- Model: {model_name} ---")
            print(f"Accuracy:  {acc:.4f}")
            print(f"Precision: {prec:.4f}")
            print(f"Recall:    {rec:.4f}")
            print(f"F1-score:  {f1:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, zero_division=0))

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"{dataset_name} - {model_name} Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.savefig(os.path.join(RESULTS_DIR, f"{dataset_name}_{model_name}_confusion_matrix.png"))
            plt.close()

            # ROC and Precision-Recall curves
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)
                plt.figure(figsize=(6,5))
                for i in range(n_classes):
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, label=f"Class {classes[i]} (AUC={roc_auc:.2f})")
                plt.plot([0,1],[0,1],'k--')
                plt.title(f"{dataset_name} - {model_name} ROC Curve")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.legend(loc="lower right")
                plt.savefig(os.path.join(RESULTS_DIR, f"{dataset_name}_{model_name}_roc_curve.png"))
                plt.close()

                # Precision-Recall curves
                plt.figure(figsize=(6,5))
                for i in range(n_classes):
                    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
                    plt.plot(recall, precision, label=f"Class {classes[i]}")
                plt.title(f"{dataset_name} - {model_name} Precision-Recall Curve")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.legend(loc="lower left")
                plt.savefig(os.path.join(RESULTS_DIR, f"{dataset_name}_{model_name}_pr_curve.png"))
                plt.close()

            summary_results.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-score": f1
            })

    # Save evaluation summary
    summary_df = pd.DataFrame(summary_results)
    summary_csv = os.path.join(RESULTS_DIR, "evaluation_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSummary saved to {summary_csv}")
    print(f"All plots saved in {RESULTS_DIR}")

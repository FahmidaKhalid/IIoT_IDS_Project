# src/evaluate.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from .config import DATASETS, MODELS_DIR, RESULTS_DIR  # <- relative import

def evaluate_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_results = []

    for dataset_name, paths in DATASETS.items():
        print(f"\n=== Dataset: {dataset_name} ===")

        X_test = np.load(paths["X_test"])
        y_test = np.load(paths["y_test"], allow_pickle=True)

        for model_name in ["RandomForest", "XGBoost", "LightGBM"]:
            model_path = os.path.join(MODELS_DIR, f"{dataset_name}_{model_name}.joblib")
            if not os.path.exists(model_path):
                print(f"Model {model_name} not found for {dataset_name}, skipping.")
                continue

            model = joblib.load(model_path)
            y_pred = model.predict(X_test)

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

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"{dataset_name} - {model_name} Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            
            # Save the plot
            plot_path = os.path.join(RESULTS_DIR, f"{dataset_name}_{model_name}_confusion_matrix.png")
            plt.savefig(plot_path)
            
            # Show the plot interactively
            plt.show()
            plt.close()

            summary_results.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-score": f1
            })

    # Save summary CSV
    summary_df = pd.DataFrame(summary_results)
    summary_csv = os.path.join(RESULTS_DIR, "evaluation_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSummary saved to {summary_csv}")


def main():
    evaluate_all()


if __name__ == "__main__":
    main()

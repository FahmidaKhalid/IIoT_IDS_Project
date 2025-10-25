import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from src.config import DATASETS, MODELS_DIR, RESULTS_DIR

warnings.filterwarnings("ignore")

def train_all():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        "LightGBM": LGBMClassifier(random_state=42)
    }

    summary_results = []

    for dataset_name, paths in DATASETS.items():
        print(f"\n=== Dataset: {dataset_name} ===")
        X_train = np.load(paths["X_train"])
        X_test = np.load(paths["X_test"])
        y_train = np.load(paths["y_train"], allow_pickle=True)
        y_test = np.load(paths["y_test"], allow_pickle=True)

        for model_name, model in models.items():
            print(f"\n--- Model: {model_name} ---")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-score: {f1:.4f}")
            joblib.dump(model, os.path.join(MODELS_DIR, f"{dataset_name}_{model_name}.joblib"))

            summary_results.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-score": f1
            })

    summary_df = pd.DataFrame(summary_results)
    summary_csv = os.path.join(RESULTS_DIR, "training_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nTraining summary saved to {summary_csv}")

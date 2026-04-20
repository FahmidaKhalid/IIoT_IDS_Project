import os
import numpy as np
import joblib
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.models import get_models

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data(dataset):
    base = os.path.join("data", "processed", dataset)

    dataset_file = dataset.replace("-", "_")

    X_train = np.load(os.path.join(base, f"X_train_{dataset_file}.npy"))
    X_test = np.load(os.path.join(base, f"X_test_{dataset_file}.npy"))
    y_train = np.load(os.path.join(base, f"y_train_{dataset_file}.npy"))
    y_test = np.load(os.path.join(base, f"y_test_{dataset_file}.npy"))

    return X_train, X_test, y_train, y_test


def train_and_evaluate(dataset):
    print(f"\n Training models for {dataset}")

    X_train, X_test, y_train, y_test = load_data(dataset)
    models = get_models()

    results = []

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred)
        }

        results.append(metrics)

        joblib.dump(
            model,
            os.path.join(MODEL_DIR, f"{name}_{dataset}.joblib")
        )

    np.save(
        os.path.join(RESULTS_DIR, f"metrics_{dataset}.npy"),
        results
    )

    print(f" Finished {dataset}")

def main():
    for dataset in ["ton-iot", "unsw-nb15"]:
        train_and_evaluate(dataset)

if __name__ == "__main__":
    main()
# For pipeline compatibility
def train_all():
    main()
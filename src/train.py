import os
import time
import numpy as np
import joblib
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer

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
    X_test  = np.load(os.path.join(base, f"X_test_{dataset_file}.npy"))
    y_train = np.load(os.path.join(base, f"y_train_{dataset_file}.npy"))
    y_test  = np.load(os.path.join(base, f"y_test_{dataset_file}.npy"))

    return X_train, X_test, y_train, y_test


def run_cross_validation(name, model, X_train, y_train, n_splits=5):
    """
    5-fold Stratified Cross-Validation on training data.
    Returns mean ± std for accuracy, precision, recall, f1.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scoring = {
        "accuracy":  make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, average="weighted", zero_division=0),
        "recall":    make_scorer(recall_score,    average="weighted", zero_division=0),
        "f1":        make_scorer(f1_score,        average="weighted", zero_division=0),
    }

    print(f"  Running {n_splits}-fold CV for {name}...")
    cv_results = cross_validate(model, X_train, y_train, cv=skf, scoring=scoring, n_jobs=-1)

    cv_summary = {
        "cv_accuracy_mean":  cv_results["test_accuracy"].mean(),
        "cv_accuracy_std":   cv_results["test_accuracy"].std(),
        "cv_precision_mean": cv_results["test_precision"].mean(),
        "cv_precision_std":  cv_results["test_precision"].std(),
        "cv_recall_mean":    cv_results["test_recall"].mean(),
        "cv_recall_std":     cv_results["test_recall"].std(),
        "cv_f1_mean":        cv_results["test_f1"].mean(),
        "cv_f1_std":         cv_results["test_f1"].std(),
    }

    print(f"  CV Accuracy : {cv_summary['cv_accuracy_mean']:.4f} ± {cv_summary['cv_accuracy_std']:.4f}")
    print(f"  CV F1-Score : {cv_summary['cv_f1_mean']:.4f} ± {cv_summary['cv_f1_std']:.4f}")

    return cv_summary


def train_and_evaluate(dataset):
    print(f"\n{'='*50}")
    print(f" Training models for: {dataset}")
    print(f"{'='*50}")

    X_train, X_test, y_train, y_test = load_data(dataset)
    models = get_models()

    results = []

    for name, model in models.items():
        print(f"\n[{name}]")

        # --- Cross-Validation (on training data) ---
        cv_summary = run_cross_validation(name, model, X_train, y_train)

        # --- Final training on full training set ---
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # --- Test set evaluation ---
        y_pred = model.predict(X_test)

        test_metrics = {
            "model":         name,
            "dataset":       dataset,
            "accuracy":      accuracy_score(y_test, y_pred),
            "precision":     precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall":        recall_score(y_test, y_pred,    average="weighted", zero_division=0),
            "f1":            f1_score(y_test, y_pred,        average="weighted", zero_division=0),
            "train_time_sec": round(train_time, 2),
        }

        # Merge CV results into metrics
        test_metrics.update(cv_summary)

        print(f"  Test Accuracy : {test_metrics['accuracy']:.4f}")
        print(f"  Test F1-Score : {test_metrics['f1']:.4f}")
        print(f"  Train Time    : {train_time:.2f}s")

        results.append(test_metrics)

        # Save trained model
        joblib.dump(
            model,
            os.path.join(MODEL_DIR, f"{name}_{dataset}.joblib")
        )

        # Save predictions for Statistical Test later
        np.save(
            os.path.join(RESULTS_DIR, f"y_pred_{name}_{dataset}.npy"),
            y_pred
        )

    # Save y_test for this dataset (needed for statistical test)
    np.save(
        os.path.join(RESULTS_DIR, f"y_test_{dataset}.npy"),
        y_test
    )

    # Save all metrics
    np.save(
        os.path.join(RESULTS_DIR, f"metrics_{dataset}.npy"),
        results
    )

    print(f"\n Finished {dataset}")
    return results


def main():
    all_results = []
    for dataset in ["ton-iot", "unsw-nb15"]:
        results = train_and_evaluate(dataset)
        all_results.extend(results)

    # Print final summary table
    print("\n" + "="*70)
    print("FINAL SUMMARY (Cross-Validation Results)")
    print("="*70)
    print(f"{'Model':<20} {'Dataset':<12} {'CV Acc (mean±std)':<22} {'CV F1 (mean±std)':<22}")
    print("-"*70)
    for r in all_results:
        print(
            f"{r['model']:<20} {r['dataset']:<12} "
            f"{r['cv_accuracy_mean']:.4f} ± {r['cv_accuracy_std']:.4f}    "
            f"{r['cv_f1_mean']:.4f} ± {r['cv_f1_std']:.4f}"
        )


if __name__ == "__main__":
    main()

# For pipeline compatibility
def train_all():
    main()
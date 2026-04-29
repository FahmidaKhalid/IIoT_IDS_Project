"""
feature_importance.py
Run this after train.py to generate feature importance plots
for Random Forest and XGBoost on both datasets.
"""
import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd

MODELS_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Feature names for each dataset
FEATURE_NAMES = {
    "ton-iot": [
        "FC1_Read_Input_Register",
        "FC2_Read_Discrete_Value",
        "FC3_Read_Holding_Register",
        "FC4_Read_Coil"
    ],
    "unsw-nb15": [
        "dur", "proto", "service", "state", "spkts", "dpkts",
        "sbytes", "dbytes", "rate", "sttl", "dttl", "sload",
        "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit",
        "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt",
        "synack", "ackdat", "smean", "dmean", "trans_depth",
        "response_body_len", "ct_srv_src", "ct_state_ttl",
        "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
        "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd",
        "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
        "is_sm_ips_ports"
    ]
}

ENSEMBLE_MODELS = ["RandomForest", "XGBoost"]


def plot_feature_importance(model_name, dataset_name, importances, feature_names, top_n=10):
    """Plot top N most important features as horizontal bar chart."""

    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    # Reverse for horizontal bar (most important at top)
    top_features = top_features[::-1]
    top_importances = top_importances[::-1]

    plt.figure(figsize=(8, 5))
    bars = plt.barh(top_features, top_importances, color="steelblue", edgecolor="black")

    # Add value labels
    for bar, val in zip(bars, top_importances):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", fontsize=8)

    plt.xlabel("Feature Importance Score")
    plt.title(f"Top {top_n} Features — {model_name} on {dataset_name}")
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, f"{dataset_name}_{model_name}_feature_importance.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.show()
    plt.close()
    print(f"  Saved: {path}")
    return path


def get_importances(model, model_name):
    """Extract feature importances from model."""
    if model_name == "RandomForest":
        return model.feature_importances_
    elif model_name == "XGBoost":
        return model.feature_importances_
    return None


def run_feature_importance():
    all_results = []

    for dataset_name, feature_names in FEATURE_NAMES.items():
        print(f"\n{'='*50}")
        print(f"  Feature Importance — {dataset_name}")
        print(f"{'='*50}")

        for model_name in ENSEMBLE_MODELS:
            model_path = os.path.join(MODELS_DIR, f"{model_name}_{dataset_name}.joblib")

            if not os.path.exists(model_path):
                print(f"  {model_name} not found — skipping.")
                continue

            print(f"\n[{model_name}]")
            model = joblib.load(model_path)
            importances = get_importances(model, model_name)

            if importances is None:
                print(f"  Cannot extract importances for {model_name}")
                continue

            # Top N (all features if dataset has few)
            top_n = min(10, len(feature_names))
            plot_feature_importance(model_name, dataset_name, importances, feature_names, top_n)

            # Print top features
            indices = np.argsort(importances)[::-1][:top_n]
            print(f"  Top {top_n} features:")
            for rank, idx in enumerate(indices, 1):
                print(f"    {rank}. {feature_names[idx]:<35} {importances[idx]:.4f}")

            # Save to results list
            for idx in indices:
                all_results.append({
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Feature": feature_names[idx],
                    "Importance": importances[idx]
                })

    # Save CSV
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(RESULTS_DIR, "feature_importance_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n Feature importance saved: {csv_path}")


if __name__ == "__main__":
    run_feature_importance()
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from sklearn.metrics import f1_score, make_scorer
from xgboost import XGBClassifier

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def safe_f1(y_true, y_pred):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    return f1_score(y_true, y_pred, average="weighted",
                    zero_division=0, labels=labels)

f1_scorer = make_scorer(safe_f1)


def plot_learning_curve(model, model_name, dataset_name, X_train, y_train,
                        train_sizes=np.linspace(0.1, 1.0, 10), cv=5):
    print(f"  Generating learning curve: {model_name} on {dataset_name}...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model,
            X_train, y_train,
            cv=cv,
            train_sizes=train_sizes,
            scoring=f1_scorer,
            n_jobs=-1,
            random_state=42,
            error_score=0
        )

    train_mean = np.nanmean(train_scores, axis=1)
    train_std  = np.nanstd(train_scores, axis=1)
    val_mean   = np.nanmean(val_scores, axis=1)
    val_std    = np.nanstd(val_scores, axis=1)

    plt.figure(figsize=(8, 5))

    plt.plot(train_sizes_abs, train_mean, 'o-', color='steelblue',
             label='Training F1-Score')
    plt.fill_between(train_sizes_abs,
                     train_mean - train_std,
                     train_mean + train_std,
                     alpha=0.15, color='steelblue')

    plt.plot(train_sizes_abs, val_mean, 's--', color='darkorange',
             label='Validation F1-Score (CV)')
    plt.fill_between(train_sizes_abs,
                     val_mean - val_std,
                     val_mean + val_std,
                     alpha=0.15, color='darkorange')

    plt.title(f"Learning Curve - {model_name} on {dataset_name}")
    plt.xlabel("Number of Training Samples")
    plt.ylabel("F1-Score (Weighted)")
    plt.ylim(0.0, 1.05)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR,
                        f"learning_curve_{model_name}_{dataset_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"  Saved: {path}")
    print(f"  Final Train F1 : {train_mean[-1]:.4f} +/- {train_std[-1]:.4f}")
    print(f"  Final Val   F1 : {val_mean[-1]:.4f} +/- {val_std[-1]:.4f}")
    gap = abs(train_mean[-1] - val_mean[-1])
    print(f"  Generalization Gap: {gap:.4f} "
          f"({'Low - no overfitting' if gap < 0.05 else 'Moderate'})")

    return train_mean, val_mean, train_std, val_std


def main():
    print("=" * 55)
    print("  Learning Curve Analysis")
    print("=" * 55)

    # TON-IoT: Random Forest
    print("\n[1/2] TON-IoT - Random Forest")
    base = os.path.join("data", "processed", "ton-iot")
    X_train = np.load(os.path.join(base, "X_train_ton_iot.npy"))
    y_train = np.load(os.path.join(base, "y_train_ton_iot.npy"))

    rf_model = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    )
    plot_learning_curve(rf_model, "RandomForest", "ton-iot", X_train, y_train)

    # UNSW-NB15: XGBoost
    print("\n[2/2] UNSW-NB15 - XGBoost")
    base = os.path.join("data", "processed", "unsw-nb15")

    try:
        X_train = np.load(os.path.join(base, "X_train_unsw_nb15.npy"))
        y_train = np.load(os.path.join(base, "y_train_unsw_nb15.npy"))
    except FileNotFoundError:
        X_train = np.load(os.path.join(base, "X_train_unsw.npy"))
        y_train = np.load(os.path.join(base, "y_train_unsw.npy"))

    xgb_model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        eval_metric="logloss", random_state=42,
        base_score=0.5
    )
    plot_learning_curve(xgb_model, "XGBoost", "unsw-nb15", X_train, y_train)

    print("\n All learning curves generated!")
    print(f" Saved in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
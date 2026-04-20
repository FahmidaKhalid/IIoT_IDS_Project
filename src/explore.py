import os
import pandas as pd
from src.config import DATASETS, RESULTS_DIR


def explore_dataset(name, path, label_column):
    print(f"\nExploring {name} dataset...")

    df = pd.read_csv(path, nrows=50000)  # sample for speed

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Missing values:\n", df.isnull().sum())

    if label_column in df.columns:
        print("Class distribution:\n", df[label_column].value_counts())

    summary = {
        "Dataset": name,
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Missing Values": df.isnull().sum().sum(),
    }

    if label_column in df.columns:
        summary["Class Distribution"] = dict(df[label_column].value_counts().to_dict())

    return summary


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summaries = []

    # TON-IoT exploration
    summaries.append(
        explore_dataset(
            "TON-IoT",
            DATASETS["ton-iot"]["raw"],
            "label"
        )
    )

    # UNSW-NB15 exploration (use training set)
    summaries.append(
        explore_dataset(
            "UNSW-NB15",
            DATASETS["unsw-nb15"]["train_raw"],
            "label"
        )
    )

    # Save summary
    summary_df = pd.DataFrame(summaries)
    summary_csv = os.path.join(RESULTS_DIR, "exploration_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    print(f"\nExploration summary saved to {summary_csv}")


if __name__ == "__main__":
    main()
import os
import pandas as pd
from src.config import DATASETS, RESULTS_DIR


def load_bot_iot(path):
    df = pd.read_csv(path)
    print("=== Bot-IoT Dataset ===")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Missing values:\n", df.isnull().sum())
    print("Class distribution:\n", df['attack'].value_counts())
    return df


def load_ton_iot(path):
    df = pd.read_csv(path)
    print("=== TON-IoT Modbus Dataset ===")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Missing values:\n", df.isnull().sum())
    print("Class distribution:\n", df['label'].value_counts())
    return df


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = []

    print("Exploring Bot-IoT dataset...")
    df_bot = load_bot_iot(DATASETS["bot-iot"]["raw"])
    summary.append({
        "Dataset": "Bot-IoT",
        "Rows": df_bot.shape[0],
        "Columns": df_bot.shape[1],
        "Missing Values": df_bot.isnull().sum().sum(),
        "Class Distribution": dict(df_bot['attack'].value_counts().to_dict())
    })

    print("\nExploring TON-IoT Modbus dataset...")
    df_ton = load_ton_iot(DATASETS["ton-iot-modbus"]["raw"])
    summary.append({
        "Dataset": "TON-IoT Modbus",
        "Rows": df_ton.shape[0],
        "Columns": df_ton.shape[1],
        "Missing Values": df_ton.isnull().sum().sum(),
        "Class Distribution": dict(df_ton['label'].value_counts().to_dict())
    })

    # Save summary
    summary_df = pd.DataFrame(summary)
    summary_csv = os.path.join(RESULTS_DIR, "exploration_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nExploration summary saved to {summary_csv}")


if __name__ == "__main__":
    main()

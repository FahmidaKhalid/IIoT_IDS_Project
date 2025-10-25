# src/explore.py
import pandas as pd
from src.config import DATASETS

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

if __name__ == "__main__":
    print("Exploring Bot-IoT dataset...")
    load_bot_iot(DATASETS["bot-iot"]["raw"])

    print("\nExploring TON-IoT Modbus dataset...")
    load_ton_iot(DATASETS["ton-iot-modbus"]["raw"])

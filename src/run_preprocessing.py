from src.preprocess import preprocess_bot_iot, preprocess_ton_iot
from src.config import DATASETS
import os

# Define the processed data output directory
PROCESSED_DIR = os.path.join("data", "processed")

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print("Starting preprocessing...")

    # Preprocess Bot-IoT dataset
    preprocess_bot_iot(DATASETS["bot-iot"]["raw"], PROCESSED_DIR)

    # Preprocess TON-IoT Modbus dataset
    preprocess_ton_iot(DATASETS["ton-iot-modbus"]["raw"], PROCESSED_DIR)

    print("All preprocessing done!")

if __name__ == "__main__":
    main()

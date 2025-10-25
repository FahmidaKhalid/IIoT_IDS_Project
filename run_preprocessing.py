from src.preprocess import preprocess_bot_iot, preprocess_ton_iot
from src.config import DATASETS, SPLITS_DIR
import os

if __name__ == "__main__":
    os.makedirs(SPLITS_DIR, exist_ok=True)
    print("Starting preprocessing...")
    preprocess_bot_iot(DATASETS["bot-iot"]["raw"], SPLITS_DIR)
    preprocess_ton_iot(DATASETS["ton-iot-modbus"]["raw"], SPLITS_DIR)
    print("All preprocessing done!")

# src/config.py

import os

# Base directories
BASE_DIR = r"C:\Users\User\IIoT_IDS_Project"
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Datasets paths
DATASETS = {
    "bot-iot": {
        "raw": os.path.join(RAW_DIR, "bot-iot", "UNSW_2018_IoT_Botnet_Final_10_best_Training.csv"),
        "X_train": os.path.join(SPLITS_DIR, "X_train_bot_iot.npy"),
        "X_test": os.path.join(SPLITS_DIR, "X_test_bot_iot.npy"),
        "y_train": os.path.join(SPLITS_DIR, "y_train_bot_iot.npy"),
        "y_test": os.path.join(SPLITS_DIR, "y_test_bot_iot.npy"),
    },
    "ton-iot-modbus": {
        "raw": os.path.join(RAW_DIR, "ton-iot", "Train_Test_IoT_Modbus.csv"),
        "X_train": os.path.join(SPLITS_DIR, "X_train_ton_iot.npy"),
        "X_test": os.path.join(SPLITS_DIR, "X_test_ton_iot.npy"),
        "y_train": os.path.join(SPLITS_DIR, "y_train_ton_iot.npy"),
        "y_test": os.path.join(SPLITS_DIR, "y_test_ton_iot.npy"),
    }
}

# Make sure output directories exist
os.makedirs(SPLITS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

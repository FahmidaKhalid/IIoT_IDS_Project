import os

BASE_DIR = r"C:\Users\User\IIoT_IDS_Project"
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

SEED = 42

# Versioning
DATA_VERSION = "v1"
MODEL_VERSION = "v1"

DATASETS = {
    "bot-iot": {
        "raw": os.path.join(RAW_DIR, "bot-iot", "UNSW_2018_IoT_Botnet_Final_10_best_Training.csv"),
        "X_train": os.path.join(SPLITS_DIR, f"X_train_bot_iot_{DATA_VERSION}.npy"),
        "X_test": os.path.join(SPLITS_DIR, f"X_test_bot_iot_{DATA_VERSION}.npy"),
        "y_train": os.path.join(SPLITS_DIR, f"y_train_bot_iot_{DATA_VERSION}.npy"),
        "y_test": os.path.join(SPLITS_DIR, f"y_test_bot_iot_{DATA_VERSION}.npy"),
    },
    "ton-iot-modbus": {
        "raw": os.path.join(RAW_DIR, "ton-iot", "Train_Test_IoT_Modbus.csv"),
        "X_train": os.path.join(SPLITS_DIR, f"X_train_ton_iot_{DATA_VERSION}.npy"),
        "X_test": os.path.join(SPLITS_DIR, f"X_test_ton_iot_{DATA_VERSION}.npy"),
        "y_train": os.path.join(SPLITS_DIR, f"y_train_ton_iot_{DATA_VERSION}.npy"),
        "y_test": os.path.join(SPLITS_DIR, f"y_test_ton_iot_{DATA_VERSION}.npy"),
    }
}

os.makedirs(SPLITS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

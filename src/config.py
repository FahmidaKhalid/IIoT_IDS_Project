import os

BASE_DIR = r"C:\Users\User\IIoT_IDS_Project"
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")  # Updated folder
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

SEED = 42

# Versioning
DATA_VERSION = "v1"
MODEL_VERSION = "v1"

DATASETS = DATASETS = {
    "ton-iot": {
        "raw": os.path.join(RAW_DIR, "ton-iot", "Train_Test_IoT_Modbus.csv"),
        "X_train": os.path.join(PROCESSED_DIR, "X_train_ton_iot.npy"),
        "X_test": os.path.join(PROCESSED_DIR, "X_test_ton_iot.npy"),
        "y_train": os.path.join(PROCESSED_DIR, "y_train_ton_iot.npy"),
        "y_test": os.path.join(PROCESSED_DIR, "y_test_ton_iot.npy"),
    },

    "unsw-nb15": {
        "train_raw": os.path.join(RAW_DIR, "unsw-nb15", "UNSW_NB15_training-set.csv"),
        "test_raw": os.path.join(RAW_DIR, "unsw-nb15", "UNSW_NB15_testing-set.csv"),
        "X_train": os.path.join(PROCESSED_DIR, "X_train_unsw.npy"),
        "X_test": os.path.join(PROCESSED_DIR, "X_test_unsw.npy"),
        "y_train": os.path.join(PROCESSED_DIR, "y_train_unsw.npy"),
        "y_test": os.path.join(PROCESSED_DIR, "y_test_unsw.npy"),
    }
}

# Create necessary directories
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

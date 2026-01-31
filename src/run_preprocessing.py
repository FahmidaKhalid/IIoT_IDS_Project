from src.preprocess import preprocess_ton_iot, preprocess_unsw_nb15
from src.config import DATASETS
import os

# Define the processed data output directory
PROCESSED_DIR = os.path.join("data", "processed")

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print("Starting preprocessing...")

    # dataset
    preprocess_ton_iot(DATASETS["ton-iot"]["raw"], PROCESSED_DIR)
    preprocess_unsw_nb15(
        DATASETS["unsw-nb15"]["train_raw"],
        DATASETS["unsw-nb15"]["test_raw"],
        PROCESSED_DIR
    )

    print("All preprocessing done!")

if __name__ == "__main__":
    main()

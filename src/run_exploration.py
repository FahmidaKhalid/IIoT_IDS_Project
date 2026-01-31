from src.explore import load_ton_iot, load_unsw_nb15
from src.config import DATASETS

def main():
    print("Exploring TON-IoT Modbus dataset...")
    load_ton_iot(DATASETS["ton-iot"]["raw"])


    print("\nExploring UNSW-nb15 dataset...")
    load_unsw_nb15(
        DATASETS["unsw-nb15"]["train_raw"],
        DATASETS["unsw-nb15"]["test_raw"]
    )

if __name__ == "__main__":
    main()

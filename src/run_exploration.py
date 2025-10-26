from src.explore import load_bot_iot, load_ton_iot
from src.config import DATASETS

def main():
    print("Exploring Bot-IoT dataset...")
    load_bot_iot(DATASETS["bot-iot"]["raw"])

    print("\nExploring TON-IoT Modbus dataset...")
    load_ton_iot(DATASETS["ton-iot-modbus"]["raw"])

if __name__ == "__main__":
    main()

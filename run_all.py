import os
from src.explore import main as explore
from src.run_preprocessing import main as preprocess
from src.run_training import main as train
from src.run_evaluation import main as evaluate

if __name__ == "__main__":
    print(" Starting Full Pipeline\n")

    # Step 0: Exploration
    print(">>> Step 0: Exploration")
    explore()  # This runs explore.py and saves exploration_summary.csv

    # Step 1: Preprocessing
    print("\n>>> Step 1: Preprocessing")
    preprocess()

    # Step 2: Training
    print("\n>>> Step 2: Training")
    train()

    # Step 3: Evaluation
    print("\n>>> Step 3: Evaluation")
    evaluate()

    print("\n Pipeline Completed Successfully! ")

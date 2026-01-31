import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os

def preprocess_bot_iot(file_path, output_dir, chunksize=100000):
    """
    Preprocess the Bot-IoT dataset:
    - Drop unnecessary columns
    - Encode categorical features
    - Scale features
    - Split into train/test
    - Apply SMOTE if dataset is imbalanced
    - Save numpy arrays to output_dir
    """
    print("Processing Bot-IoT dataset in chunks...")
    os.makedirs(output_dir, exist_ok=True)

    chunks = []
    drop_cols = ["pkSeqID", "saddr", "sport", "daddr", "dport", "subcategory"]

    for chunk in pd.read_csv(file_path, chunksize=chunksize, low_memory=False):
        chunk = chunk.drop(columns=drop_cols, errors="ignore")
        for col in ["proto", "category"]:
            chunk[col] = LabelEncoder().fit_transform(chunk[col])
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    print(f"Total rows after combining chunks: {df.shape[0]}")

    X = df.drop(columns=["attack"])
    y = df["attack"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply SMOTE if class imbalance exists
    if y_train.value_counts().min() / y_train.value_counts().max() < 0.4:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print("SMOTE applied to balance classes in training set.")

    # Save train/test splits
    np.save(os.path.join(output_dir, "X_train_bot_iot.npy"), X_train)
    np.save(os.path.join(output_dir, "X_test_bot_iot.npy"), X_test)
    np.save(os.path.join(output_dir, "y_train_bot_iot.npy"), y_train)
    np.save(os.path.join(output_dir, "y_test_bot_iot.npy"), y_test)
    print("Bot-IoT preprocessing complete!")


def preprocess_ton_iot(file_path, output_dir):
    """
    Preprocess the TON-IoT Modbus dataset:
    - Drop unnecessary columns
    - Scale features
    - Split into train/test
    - Apply SMOTE if dataset is imbalanced
    - Save numpy arrays to output_dir
    """
    print("Processing TON-IoT Modbus dataset...")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(file_path)
    drop_cols = ["date", "time", "type"]
    df = df.drop(columns=drop_cols, errors="ignore")

    X = df.drop(columns=["label"])
    y = df["label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply SMOTE if class imbalance exists
    if y_train.value_counts().min() / y_train.value_counts().max() < 0.4:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print("SMOTE applied to balance classes in training set.")

    # Save train/test splits
    np.save(os.path.join(output_dir, "X_train_ton_iot.npy"), X_train)
    np.save(os.path.join(output_dir, "X_test_ton_iot.npy"), X_test)
    np.save(os.path.join(output_dir, "y_train_ton_iot.npy"), y_train)
    np.save(os.path.join(output_dir, "y_test_ton_iot.npy"), y_test)
    print("TON-IoT preprocessing complete!")
def preprocess_unsw_nb15(train_path, test_path, output_dir):
    """
    Preprocess the UNSW-NB15 dataset:
    - Load train & test CSVs
    - Encode categorical features
    - Scale numerical features
    - Save NumPy arrays
    """
    print("Processing UNSW-NB15 dataset...")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    drop_cols = ["id", "attack_cat"]
    train_df = train_df.drop(columns=drop_cols, errors="ignore")
    test_df = test_df.drop(columns=drop_cols, errors="ignore")

    y_train = train_df.pop("label").values
    y_test = test_df.pop("label").values

    categorical_cols = train_df.select_dtypes(include=["object"]).columns
    encoder = LabelEncoder()

    for col in categorical_cols:
        combined = pd.concat([train_df[col], test_df[col]])
        encoder.fit(combined)
        train_df[col] = encoder.transform(train_df[col])
        test_df[col] = encoder.transform(test_df[col])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df)
    X_test = scaler.transform(test_df)

    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "X_train_unsw.npy"), X_train)
    np.save(os.path.join(output_dir, "X_test_unsw.npy"), X_test)
    np.save(os.path.join(output_dir, "y_train_unsw.npy"), y_train)
    np.save(os.path.join(output_dir, "y_test_unsw.npy"), y_test)

    print("UNSW-NB15 preprocessing complete!")
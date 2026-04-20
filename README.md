# IIoT Intrusion Detection System (IDS)

## Overview

This project presents a machine learning-based Intrusion Detection System (IDS) for Industrial Internet of Things (IIoT) environments.

The objective is to evaluate and compare multiple machine learning models for detecting malicious network traffic under different dataset conditions. The study focuses on identifying the most effective models for handling complex and non-linear IIoT network patterns.

---

## Models Implemented

* Naïve Bayes
* Logistic Regression
* k-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Random Forest
* XGBoost
* Multi-Layer Perceptron (MLP)

---

## Datasets Used

* TON-IoT Dataset
* UNSW-NB15 Dataset

---

## Project Structure

```
IIoT_IDS_Project/
├── data/
├── models/
├── results/
├── src/
├── run_all.py
├── requirements.txt
└── README.md
```

---

## How to Run

1. Clone the repository:

```
git clone https://github.com/FahmidaKhalid/IIoT_IDS_Project.git
cd IIoT_IDS_Project
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Pull datasets using DVC:

```
dvc pull
```

4. Run the project:

```
python run_all.py
```

---

## Results

Key findings of the study include:

* Random Forest achieved the highest accuracy (98.42%) on the TON-IoT dataset
* XGBoost achieved the best performance (87.54%) on the UNSW-NB15 dataset
* Ensemble learning methods outperform traditional machine learning models
* Model performance varies depending on dataset characteristics

---

## Dataset Access Note

The dataset is managed using DVC (Data Version Control) and stored on a private university server.

Running `dvc pull` may require authentication due to access restrictions.

The dataset is not publicly exposed for security and access control reasons. However, if access is required, it can be shared separately upon request.

---

## Reproducibility

This project ensures reproducibility using Git and DVC.

The code, workflow, and experimental pipeline are fully available in this repository. All experiments can be reproduced by following the provided steps, given appropriate dataset access.

---

## Contribution

This project provides a comparative evaluation of multiple machine learning models for IIoT intrusion detection.

It demonstrates the effectiveness of ensemble learning methods (Random Forest and XGBoost) and introduces a structured and reproducible experimental pipeline for reliable evaluation.

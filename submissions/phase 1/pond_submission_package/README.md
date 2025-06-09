# Pond Spam Token Detection

A lightweight guide for operating the model for **Spam Token Detection on Ethereum**.
Covers usage for both **inference** and **retraining**.

---

## 1. Project Overview

This project detects spam tokens using transactional and transfer behavior features on Ethereum blockchain data. It uses a LightGBM-based model.

## 2. Setup

**Requirements:**

- Python 3.10+
- Install required packages:

```bash
pip install -r requirements.txt
```

## 3. Data Preparation

**Input Datasets:** (already stored in `data/raw/`)

- `training_tokens.parquet`
- `test_tokens.parquet`
- `transactions.parquet`
- `token_transfers.parquet`
- `nft_transfers.parquet`
- `dex_swaps.parquet`

**Feature Engineering:**

To prepare the feature files manually, run:

```bash
# Basic features
python src/features/01_basic_transaction_features.py
python src/features/02_basic_token_transfer_features.py
python src/features/03_basic_nft_transfer_features.py

# Merge features
python src/features/04_merge_all_basic_features.py

# Advanced features
python src/features/05_advanced_features.py
python src/features/06_merge_with_advanced_features.py
```

> **Important:** Always use `all_features_merged_v2.parquet` for model training and inference.

## 4. Training the Model

To retrain the model (after preparing features):

```bash
python src/models/07_retrain_final_model.py
```

This will save a new model under `models/final_lgbm_model.pkl`.

## 5. Inference (Prediction)

**Preparing Test Features:**

```bash
python scripts/prepare_test_features.py
```

**Making Predictions:**

```bash
python src/models/08_predict_test_data.py
```

Predictions will be saved to `data/predictions/test_predictions.parquet`.

## 6. Submission

To generate a competition-ready submission file:

```bash
python scripts/create_submission.py
```

This will create `submissions/submission.csv`.

## 7. Project Structure

```plaintext
├── data/
│   ├── raw/ (input data)
│   ├── processed/ (feature files)
│   └── predictions/ (model outputs)
├── models/ (saved models)
├── notebooks/ (EDA and analysis)
├── scripts/ (helper scripts)
├── src/
│   ├── analysis/
│   ├── features/
│   └── models/
├── submissions/ (submission CSVs)
├── requirements.txt
├── README.md
```

## 8. Notes

- `all_features_merged_v2.parquet` is the final and improved feature file. Ignore `all_features_merged.parquet` unless studying previous versions.
- Model training can be done on CPU but may take 1-2 hours depending on hardware.
- All scripts assume correct relative paths inside the project folder.
- It's recommended to use the provided scripts rather than running notebooks manually.

## 9. Missing Files

Some large files (datasets, feature files, model artifacts) are intentionally excluded from the repository via `.gitignore`.

To fully operate this project (training, inference, submission generation), you must have access to the following folders locally:
- `data/`
- `models/`

Please request access separately if needed.

## 10. Provided Artifacts
This repository includes:

Trained model: models/final_lgbm_model.pkl

Final submission file: submissions/submission.csv

---
# Pond Spam Token Detection

A lightweight guide for operating the model for **Spam Token Detection on Ethereum**. Covers usage for both **inference** and **retraining**.

---

## 1. Project Overview

This project detects spam tokens using transactional, transfer, and decentralized exchange (DEX) behavior features on Ethereum blockchain data. It uses a LightGBM-based model.

## 2. Setup

**Requirements:**

* Python 3.10+
* Install required packages:

```bash
pip install -r requirements.txt
```

## 3. Data Preparation

**Input Datasets:** (already stored in `data/raw/`)

* `training_tokens.parquet`
* `test_tokens.parquet`
* `transactions.parquet`
* `token_transfers.parquet`
* `nft_transfers.parquet`
* `dex_swaps.parquet`

**Feature Engineering:**

To prepare the feature files manually, run:

```bash
# Phase 1 features
python src/features/01_basic_transaction_features.py
python src/features/02_basic_token_transfer_features.py
python src/features/03_basic_nft_transfer_features.py
python src/features/04_merge_all_basic_features.py
python src/features/05_advanced_features.py
python src/features/06_merge_with_advanced_features.py

# Phase 2 features
python src/features/07_dex_presence.py
python src/features/08_total_dex_usd_volume.py
python src/features/09_unique_wallets_interacted.py
python src/features/10_merge_with_advanced_features_phase_2.py
```

> **Important:** Use `all_features_merged_phase_2.parquet` for Phase 2 model training and prediction.

## 4. Training the Model

To retrain the model using Phase 2 features:

```bash
python src/models/09_retrain_final_model_phase_2.py
```

This will save the model to: `models/final_lgbm_model_v2.pkl`

## 5. Inference (Prediction)

To generate predictions on the test set:

```bash
python src/models/10_predict_test_data_phase_2.py
```

Predictions will be saved to: `data/processed/predictions/test_predictions_phase_2.parquet`

## 6. Submission

To generate a competition-ready submission file:

```bash
python scripts/create_submission_phase_2.py
```

This will create: `submissions/phase 2/submission.csv`

## 7. Project Structure

```plaintext
├── data/
│   ├── raw/ (input data)
│   ├── processed/
│   │   ├── features/ (all engineered features)
│   │   └── predictions/ (model outputs)
├── models/ (saved models)
├── scripts/ (submission + helper scripts)
├── src/
│   ├── features/ (feature engineering scripts)
│   └── models/ (training + prediction)
├── submissions/ (CSV submissions)
├── requirements.txt
├── README.md
```

## 8. Notes

* `all_features_merged_phase_2.parquet` is the latest version used for both training and inference in Phase 2.
* New Phase 2 features include:

  * `dex_presence_flag`
  * `total_usd_volume`
  * `unique_wallets_interacted`
* Always normalize addresses to lowercase before merging or prediction.
* It is recommended to use scripts instead of notebooks for full reproducibility.

## 9. Missing Files

Some large files (datasets, feature files, model artifacts) are excluded via `.gitignore`.

You must have access to the following folders locally:

* `data/`
* `models/`

## 10. Provided Artifacts

* Trained model: `models/final_lgbm_model_v2.pkl`
* Final submission: `submissions/phase 2/submission.csv`

---

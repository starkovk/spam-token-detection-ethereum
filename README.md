#  Pond Spam Token Detection (Phase 3)

This repository contains the full pipeline for detecting spam tokens on the Ethereum blockchain using feature-based machine learning models, built for the [Pond Model Factory Competition](https://cryptopond.xyz/modelfactory/detail/31).

---

##  Objective

Classify Ethereum tokens as either:
- `1` → **Spam token**
- `0` → **Legitimate token**

We use raw on-chain data (transactions, token transfers, NFT transfers, DEX swaps) to engineer features, train a LightGBM model, and produce a submission for competition evaluation.

---

##  Project Structure (Phase 3)

### `/data/raw/`
Raw input data (provided by competition):
- `training_tokens.parquet`
- `test_tokens.parquet`
- `transactions.parquet`
- `token_transfers.parquet`
- `nft_transfers.parquet`
- `dex_swaps.parquet`

### `/data/processed/features/`
Engineered features:
- `01_basic_transaction_features.parquet`
- `02_basic_token_transfer_features.parquet`
- `03_basic_nft_transfer_features.parquet`
- `04_unique_wallets_interacted.parquet`
- `05_token_transfer_entropy.parquet`
- `06_all_features_merged_phase_3.parquet` ← **Final merged training features**

### `/data/processed/predictions/`
Model predictions on test tokens:
- `test_predictions_phase_3.parquet`

### `/models/`
Trained model files:
- `phase3_lgbm_model.pkl` ← LightGBM model trained on Phase 3 features

### `/src/features/`
Feature generation and merging scripts:
- `01_basic_transaction_features.py`
- `02_basic_token_transfer_features.py`
- `03_basic_nft_transfer_features.py`
- `04_unique_wallets_interacted.py`
- `05_token_transfer_entropy.py`
- `06_all_features_merged_phase_3.py` ← merges all feature files into training matrix

### `/src/models/`
Model training and prediction scripts:
- `05_train_final_model_phase_3.py`
- `06_predict_test_data_phase_3.py`

### `/scripts/`
Submission formatting:
- `create_submission_phase_3.py` ← generates final CSV

### `/submissions/`
Final competition-ready CSVs:
- `submission_phase_3.csv`


### Feature Evaluation Strategy
Each new feature or group of features is evaluated in a sandbox (upgrade/phase_<n>/<feature>/). The evaluation includes:

Standalone performance

Incremental value when added to prior features

Agreement with previous predictions

Accuracy improvements on fixed holdout set

Only features that pass the gate criteria are promoted to the main repo.

### Phase Promotion
Once validated, the following files are created for the promoted feature set:

src/features/<feature>_phase_<n>.py

src/features/all_features_merged_phase_<n>.py

src/models/train_final_model_phase_<n>.py

src/models/predict_test_data_phase_<n>.py

src/models/validate_on_holdout_phase_<n>.py

scripts/create_submission_phase_<n>.py

models/final_lgbm_model_phase_<n>.pkl

data/processed/predictions/test_predictions_phase_<n>.parquet

submissions/phase_<n>/submission_phase_<n>.csv

---

##  How to Run End-to-End

From project root (`e:/code/projects/pond-spam-token-detection`):

### 1. Generate Features
```bash
python src/features/01_basic_transaction_features.py
python src/features/02_basic_token_transfer_features.py
python src/features/03_basic_nft_transfer_features.py
python src/features/04_unique_wallets_interacted.py
python src/features/05_token_transfer_entropy.py
python src/features/06_all_features_merged_phase_3.py# manual test
# Daily commit on 29 июн 2025 г. 13:21:15

import pandas as pd
import lightgbm as lgb
import joblib
import os

# Paths
BASE_DIR = "e:/code/projects/pond-spam-token-detection"
FEATURES_DIR = os.path.join(BASE_DIR, "data", "processed", "features")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "data", "predictions")
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# Load model
model = joblib.load(os.path.join(MODEL_DIR, "final_lgbm_model.pkl"))

# Load features
X_all = pd.read_parquet(os.path.join(FEATURES_DIR, "all_features_merged_v2.parquet"))

# Load test tokens
test_tokens = pd.read_parquet(os.path.join(RAW_DIR, "test_tokens.parquet"))

# Normalize addresses
X_all['token_address'] = X_all['token_address'].str.lower()
test_tokens['ADDRESS'] = test_tokens['ADDRESS'].str.lower()

# Merge features with test tokens
X_test = pd.merge(test_tokens, X_all, left_on="ADDRESS", right_on="token_address", how="left")

# Select only the feature columns we trained on
selected_features = [
    'token_lifetime_days',
    'tx_received_count',
    'transfer_received_count',
    'transfer_asymmetry',
    'token_sent_total',
    'token_sent_avg',
    'transfer_sent_count',
    'token_received_total',
    'token_received_median',
    'token_sent_median'
]

# Prepare final X_test
X_test_final = X_test[selected_features]

# Predict
preds = model.predict(X_test_final)

# Attach predictions back
X_test['PREDICTED_LABEL'] = preds

# Save predictions
output_path = os.path.join(PREDICTIONS_DIR, "test_predictions.parquet")
X_test[['ADDRESS', 'PREDICTED_LABEL']].to_parquet(output_path, index=False)

print(f" Predictions saved to {output_path}")

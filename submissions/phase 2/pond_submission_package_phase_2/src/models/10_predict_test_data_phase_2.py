import pandas as pd
import joblib
import os

# === Paths ===
BASE_DIR = "e:/code/projects/pond-spam-token-detection"
FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "features", "all_features_merged_phase_2.parquet")
TEST_TOKENS_PATH = os.path.join(BASE_DIR, "data", "raw", "test_tokens.parquet")
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_lgbm_model_v2.pkl")
PREDICTIONS_PATH = os.path.join(BASE_DIR, "data", "processed", "predictions", "test_predictions_phase_2.parquet")

# === Load data ===
features = pd.read_parquet(FEATURES_PATH)
test_tokens = pd.read_parquet(TEST_TOKENS_PATH)

# Normalize for merging
features["token_address"] = features["token_address"].str.lower()
test_tokens["ADDRESS"] = test_tokens["ADDRESS"].str.lower()

# Merge test tokens with features
X_test = pd.merge(test_tokens, features, left_on="ADDRESS", right_on="token_address", how="left")

# Drop non-feature columns (keep only features)
drop_cols = ["token_address", "ADDRESS", "SYMBOL", "NAME", "CREATED_BLOCK_TIMESTAMP", "CREATOR_ADDRESS"]
X_final = X_test.drop(columns=[col for col in drop_cols if col in X_test.columns])

# Load trained model
model = joblib.load(MODEL_PATH)

# Predict
X_test["PREDICTED_LABEL"] = model.predict(X_final)

# Save predictions
output = X_test[["ADDRESS", "PREDICTED_LABEL"]].rename(columns={"ADDRESS": "token_address"})
os.makedirs(os.path.dirname(PREDICTIONS_PATH), exist_ok=True)
output.to_parquet(PREDICTIONS_PATH, index=False)

print(f" Predictions saved to: {PREDICTIONS_PATH} with shape {output.shape}")
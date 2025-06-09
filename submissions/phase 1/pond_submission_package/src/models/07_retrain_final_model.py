import pandas as pd
import lightgbm as lgb
import os
import joblib

# Paths
BASE_DIR = "e:/code/projects/pond-spam-token-detection"
FEATURES_DIR = os.path.join(BASE_DIR, "data", "processed", "features")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
X = pd.read_parquet(os.path.join(FEATURES_DIR, "all_features_merged_v2.parquet"))
labels = pd.read_parquet(os.path.join(RAW_DIR, "training_tokens.parquet"))

# Normalize addresses
X['token_address'] = X['token_address'].str.lower()
labels['ADDRESS'] = labels['ADDRESS'].str.lower()

# Merge
data = pd.merge(X, labels[['ADDRESS', 'LABEL']], left_on='token_address', right_on='ADDRESS', how='inner')

# Select features
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

X = data[selected_features]
y = data["LABEL"].astype(int)

# Define final model with best hyperparameters
final_model = lgb.LGBMClassifier(
    bagging_fraction=0.7594,
    bagging_freq=6,
    feature_fraction=0.8540,
    lambda_l1=0.2265,
    lambda_l2=1.8730,
    learning_rate=0.1270,
    max_depth=12,
    min_child_samples=5,
    num_leaves=88,
    random_state=42
)

# Train on full data
final_model.fit(X, y)

# Save model
model_path = os.path.join(MODEL_DIR, "final_lgbm_model.pkl")
joblib.dump(final_model, model_path)

print(f" Final model trained and saved to {model_path}")

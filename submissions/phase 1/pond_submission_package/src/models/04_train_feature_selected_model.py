import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Define paths
BASE_DIR = "e:/code/projects/pond-spam-token-detection"
FEATURES_DIR = os.path.join(BASE_DIR, "data", "processed", "features")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

# Load features and labels
X = pd.read_parquet(os.path.join(FEATURES_DIR, "all_features_merged_v2.parquet"))
labels = pd.read_parquet(os.path.join(RAW_DIR, "training_tokens.parquet"))

X['token_address'] = X['token_address'].str.lower()
labels['ADDRESS'] = labels['ADDRESS'].str.lower()

# Merge
data = pd.merge(X, labels[['ADDRESS', 'LABEL']], left_on='token_address', right_on='ADDRESS', how='inner')

# Keep only strong features
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

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train LightGBM
model = lgb.LGBMClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f" Feature-Selected Model Accuracy: {accuracy:.4f}")

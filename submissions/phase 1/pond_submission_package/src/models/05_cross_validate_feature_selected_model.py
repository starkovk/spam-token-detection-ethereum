import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import os

# Define paths
BASE_DIR = "e:/code/projects/pond-spam-token-detection"
FEATURES_DIR = os.path.join(BASE_DIR, "data", "processed", "features")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

# Load features and labels
X = pd.read_parquet(os.path.join(FEATURES_DIR, "all_features_merged_v2.parquet"))
labels = pd.read_parquet(os.path.join(RAW_DIR, "training_tokens.parquet"))

# Normalize casing
X['token_address'] = X['token_address'].str.lower()
labels['ADDRESS'] = labels['ADDRESS'].str.lower()

# Merge
data = pd.merge(X, labels[['ADDRESS', 'LABEL']], left_on='token_address', right_on='ADDRESS', how='inner')

# Select only strong features
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

# Prepare Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    fold_accuracies.append(acc)

    print(f" Fold {fold + 1} Accuracy: {acc:.4f}")

print(f"\n Average CV Accuracy: {sum(fold_accuracies) / len(fold_accuracies):.4f}")

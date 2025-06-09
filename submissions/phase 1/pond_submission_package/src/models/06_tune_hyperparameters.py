import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
import os

# Paths
BASE_DIR = "e:/code/projects/pond-spam-token-detection"
FEATURES_DIR = os.path.join(BASE_DIR, "data", "processed", "features")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

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

# Prepare model and CV
model = lgb.LGBMClassifier(random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define search space
from scipy.stats import randint, uniform

param_dist = {
    'num_leaves': randint(20, 150),
    'max_depth': randint(3, 15),
    'learning_rate': uniform(0.005, 0.2 - 0.005),
    'min_child_samples': randint(5, 100),
    'feature_fraction': uniform(0.6, 0.4),  # 0.6 to 1.0
    'bagging_fraction': uniform(0.6, 0.4),
    'bagging_freq': randint(1, 10),
    'lambda_l1': uniform(0.0, 5.0),
    'lambda_l2': uniform(0.0, 5.0)
}


# Random Search
search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=50,  # 50 random combinations to try
    scoring='accuracy',
    cv=skf,
    verbose=2,
    random_state=42,
    n_jobs=-1  # Use all available cores
)

# Execute search
search.fit(X, y)

# Best results
print("\n Best Hyperparameters Found:")
print(search.best_params_)
print(f" Best CV Accuracy: {search.best_score_:.4f}")

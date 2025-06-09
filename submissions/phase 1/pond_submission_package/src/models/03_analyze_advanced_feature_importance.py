import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
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

# Prepare X and y
X = data.drop(columns=["token_address", "LABEL", "ADDRESS"])
y = data["LABEL"].astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train LightGBM model
model = lgb.LGBMClassifier(random_state=42)
model.fit(X_train, y_train)

# Get feature importance
importance_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
})

# Sort by importance
importance_df = importance_df.sort_values(by="importance", ascending=False)

# Show importance
print(importance_df)

# Plot top 15 important features
plt.figure(figsize=(12, 7))
plt.barh(importance_df["feature"].head(15), importance_df["importance"].head(15))
plt.gca().invert_yaxis()
plt.xlabel("Feature Importance")
plt.title("Top 15 Important Features (LightGBM - Advanced Features)")
plt.tight_layout()
plt.show()

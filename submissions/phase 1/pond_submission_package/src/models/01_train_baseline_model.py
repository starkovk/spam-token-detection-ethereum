import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Define paths
BASE_DIR = "e:/code/projects/pond-spam-token-detection"
FEATURES_DIR = os.path.join(BASE_DIR, "data", "processed", "features")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

# Load merged features
X = pd.read_parquet(os.path.join(FEATURES_DIR, "all_features_merged.parquet"))

# Load labels
labels = pd.read_parquet(os.path.join(RAW_DIR, "training_tokens.parquet"))
labels['ADDRESS'] = labels['ADDRESS'].str.lower()

# Prepare features
X['token_address'] = X['token_address'].str.lower()

# Merge features with labels
data = pd.merge(X, labels[['ADDRESS', 'LABEL']], left_on='token_address', right_on='ADDRESS', how='inner')

# Separate X and y
X = data.drop(columns=["token_address", "LABEL", "ADDRESS"])
y = data["LABEL"].astype(int)  # Important: cast LABEL to integer

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Extra Safety: Ensure y_train and y_test are int
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# Train LightGBM model
model = lgb.LGBMClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f" Baseline Model Accuracy: {accuracy:.4f}")

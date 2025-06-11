import pandas as pd
import lightgbm as lgb
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Paths
features_path = "data/processed/features/all_features_merged_phase_2.parquet"
labels_path = "data/raw/training_tokens.parquet"
model_path = "models/final_lgbm_model_v2.pkl"

def main():
    # Load features
    X = pd.read_parquet(features_path)
    X["token_address"] = X["token_address"].str.lower()

    # Load labels
    labels = pd.read_parquet(labels_path)[["ADDRESS", "LABEL"]]
    labels["ADDRESS"] = labels["ADDRESS"].str.lower()
    labels["LABEL"] = labels["LABEL"].astype(int)  # ensure correct dtype for classification

    # Merge features + labels
    df = pd.merge(X, labels, left_on="token_address", right_on="ADDRESS", how="inner")

    # Prepare training data
    X_final = df.drop(columns=["token_address", "ADDRESS", "LABEL"])
    y_final = df["LABEL"]

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(X_final, y_final, test_size=0.2, stratify=y_final, random_state=42)

    # Train model
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Validate
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f" Phase 2 model accuracy: {acc:.4f}")

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f" Saved model to: {model_path}")

if __name__ == "__main__":
    main()
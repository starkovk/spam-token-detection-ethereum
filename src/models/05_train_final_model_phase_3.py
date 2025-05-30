import pandas as pd
import lightgbm as lgb
import joblib
import os

# Paths
FEATURES_PATH = "data/processed/features/06_all_features_merged_phase_3.parquet"
LABELS_PATH = "data/raw/training_tokens.parquet"
MODEL_OUTPUT_PATH = "models/phase3_lgbm_model.pkl"


def main():
    # Load feature data
    features = pd.read_parquet(FEATURES_PATH)
    features["token_address"] = features["token_address"].str.lower()

    # Load labels
    labels = pd.read_parquet(LABELS_PATH)[["ADDRESS", "LABEL"]]
    labels["ADDRESS"] = labels["ADDRESS"].str.lower()

    # Merge on token address
    merged = features.merge(labels, left_on="token_address", right_on="ADDRESS", how="inner")

    # Clean label
    merged["LABEL"] = pd.to_numeric(merged["LABEL"], errors="coerce")
    merged = merged.dropna(subset=["LABEL"])
    merged["LABEL"] = merged["LABEL"].astype(int)

    # Split X and y
    X = merged.drop(columns=["ADDRESS", "token_address", "LABEL"])
    y = merged["LABEL"]

    # Train model
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X, y)

    # Save model
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)

    print(f" Model trained and saved to: {MODEL_OUTPUT_PATH}")
    print(f" Training data shape: {X.shape}")
    print(f" Label distribution:\n{y.value_counts()}")

if __name__ == "__main__":
    main()
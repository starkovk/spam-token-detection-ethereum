import pandas as pd
import joblib
import os

# Paths
FEATURES_PATH = "data/processed/features/06_all_features_merged_phase_3.parquet"
TEST_TOKENS_PATH = "data/raw/test_tokens.parquet"
MODEL_PATH = "models/phase3_lgbm_model.pkl"
OUTPUT_PATH = "data/processed/predictions/test_predictions_phase_3.parquet"

def main():
    # Load data
    features = pd.read_parquet(FEATURES_PATH)
    test_tokens = pd.read_parquet(TEST_TOKENS_PATH)

    # Normalize keys
    features["token_address"] = features["token_address"].str.lower()
    test_tokens["ADDRESS"] = test_tokens["ADDRESS"].str.lower()

    # Merge
    merged = pd.merge(
        test_tokens[["ADDRESS"]],
        features,
        left_on="ADDRESS",
        right_on="token_address",
        how="left"
    )

    if "ADDRESS" not in merged.columns:
        raise ValueError("ADDRESS column missing after merge")

    address_col = merged["ADDRESS"]
    drop_cols = ["token_address", "SYMBOL", "NAME", "CREATED_BLOCK_TIMESTAMP", "CREATOR_ADDRESS"]
    X = merged.drop(columns=[col for col in drop_cols if col in merged.columns])

    # Predict
    model = joblib.load(MODEL_PATH)
    predictions = model.predict(X.drop(columns=["ADDRESS"]))

    result = pd.DataFrame({
        "ADDRESS": address_col,
        "PREDICTED_LABEL": predictions
    })

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    result.to_parquet(OUTPUT_PATH, index=False)
    print(f" Predictions saved to: {OUTPUT_PATH} with shape {result.shape}")

if __name__ == "__main__":
    main()
import pandas as pd
import os

def main():
    # Use real project feature path (not upgrade)
    FEATURE_DIR = "data/processed/features"
    output_path = os.path.join(FEATURE_DIR, "06_all_features_merged_phase_3.parquet")

    # Load all features (Phase 1–3)
    f1 = pd.read_parquet(os.path.join(FEATURE_DIR, "01_basic_transaction_features.parquet"))
    f2 = pd.read_parquet(os.path.join(FEATURE_DIR, "02_basic_token_transfer_features.parquet"))
    f3 = pd.read_parquet(os.path.join(FEATURE_DIR, "03_basic_nft_transfer_features.parquet"))
    f4 = pd.read_parquet(os.path.join(FEATURE_DIR, "04_unique_wallets_interacted.parquet"))  # upgraded version
    f5 = pd.read_parquet(os.path.join(FEATURE_DIR, "05_token_transfer_entropy.parquet"))

    # Normalize key
    for df in [f1, f2, f3, f4, f5]:
        df["token_address"] = df["token_address"].str.lower()

    # Merge all on token_address
    merged = f1
    for f in [f2, f3, f4, f5]:
        merged = pd.merge(merged, f, on="token_address", how="outer")

    merged = merged.fillna(0)

    # Save merged output
    os.makedirs(FEATURE_DIR, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    print(f"Merged features saved to: {output_path} with shape {merged.shape}")

if __name__ == "__main__":
    main()
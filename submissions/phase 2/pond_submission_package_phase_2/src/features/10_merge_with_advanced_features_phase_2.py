import pandas as pd
import os

# Input paths
base_path = "data/processed/features/all_features_merged_v2.parquet"
dex_presence_path = "data/processed/features/07_dex_presence.parquet"
usd_volume_path = "data/processed/features/dex_total_usd_volume.parquet"
wallets_path = "data/processed/features/unique_wallets_interacted.parquet"

# Output path
output_path = "data/processed/features/all_features_merged_phase_2.parquet"

def main():
    # Load base (Phase 1) features
    base = pd.read_parquet(base_path)
    base["token_address"] = base["token_address"].str.lower()

    # Load new Phase 2 features
    dex_presence = pd.read_parquet(dex_presence_path)
    usd_volume = pd.read_parquet(usd_volume_path)
    wallets = pd.read_parquet(wallets_path)

    # Normalize token_address
    dex_presence["ADDRESS"] = dex_presence["ADDRESS"].str.lower()
    usd_volume["ADDRESS"] = usd_volume["ADDRESS"].str.lower()
    wallets["token_address"] = wallets["token_address"].str.lower()

    # Rename columns for merge consistency
    dex_presence = dex_presence.rename(columns={"ADDRESS": "token_address", "dex_presence": "dex_presence_flag"})
    usd_volume = usd_volume.rename(columns={"ADDRESS": "token_address", "dex_total_usd_volume": "total_usd_volume"})

    # Merge step by step
    merged = pd.merge(base, dex_presence, on="token_address", how="left")
    merged = pd.merge(merged, usd_volume, on="token_address", how="left")
    merged = pd.merge(merged, wallets, on="token_address", how="left")

    # Fill missing values
    merged["dex_presence_flag"] = merged["dex_presence_flag"].fillna(0).astype(int)
    merged["total_usd_volume"] = merged["total_usd_volume"].fillna(0)
    merged["unique_wallets_interacted"] = merged["unique_wallets_interacted"].fillna(0).astype(int)

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_parquet(output_path, index=False)
    print(f" Merged advanced + Phase 2 features: {output_path} with shape {merged.shape}")

if __name__ == "__main__":
    main()
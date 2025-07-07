import pandas as pd
import os

"""
Summarises NFT transfer stats by NFT_ADDRESS
→ One row per token
→ Outputs: data/processed/features/03_basic_nft_transfer_features.parquet
"""

def main():
    input_path = "data/raw/nft_transfers.parquet"
    output_path = "data/processed/features/03_basic_nft_transfer_features.parquet"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cols = [
        "NFT_ADDRESS",
        "NFT_FROM_ADDRESS",
        "NFT_TO_ADDRESS",
        "BLOCK_TIMESTAMP"
    ]
    df = pd.read_parquet(input_path, columns=cols)
    df["NFT_ADDRESS"] = df["NFT_ADDRESS"].str.lower()

    # Count transfers per token
    grouped = df.groupby("NFT_ADDRESS")
    result = pd.DataFrame()
    result["nft_transfer_count"] = grouped.size()
    result["nft_unique_senders"] = grouped["NFT_FROM_ADDRESS"].nunique()
    result["nft_unique_receivers"] = grouped["NFT_TO_ADDRESS"].nunique()

    result = result.reset_index().rename(columns={"NFT_ADDRESS": "token_address"})
    result.to_parquet(output_path, index=False)

    print(f" 03_basic_nft_transfer_features.parquet saved with {len(result)} rows.")

if __name__ == "__main__":
    main()# Daily commit on  7 июл 2025 г. 23:55:07

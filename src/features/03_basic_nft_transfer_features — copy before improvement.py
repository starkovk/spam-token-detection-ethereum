import pandas as pd
import numpy as np
import os

# Define paths
BASE_DIR = "e:/code/projects/pond-spam-token-detection"
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "features")
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Load nft transfers data
nft_transfers = pd.read_parquet(
    os.path.join(RAW_DATA_DIR, "nft_transfers.parquet"),
    columns=["NFT_FROM_ADDRESS", "NFT_TO_ADDRESS", "NFT_ADDRESS"]
)

# Group by NFT_FROM_ADDRESS (sent NFTs)
sent = nft_transfers.groupby("NFT_FROM_ADDRESS").agg(
    nft_sent_count=("NFT_ADDRESS", "count"),
).reset_index().rename(columns={"NFT_FROM_ADDRESS": "token_address"})

# Group by NFT_TO_ADDRESS (received NFTs)
received = nft_transfers.groupby("NFT_TO_ADDRESS").agg(
    nft_received_count=("NFT_ADDRESS", "count"),
).reset_index().rename(columns={"NFT_TO_ADDRESS": "token_address"})

# Merge sent and received features
features = pd.merge(sent, received, on="token_address", how="outer").fillna(0)

# Save features with correct output name
features.to_parquet(os.path.join(PROCESSED_DATA_DIR, "03_basic_nft_transfer_features.parquet"), index=False)
print(" 03_basic_nft_transfer_features.parquet saved.")
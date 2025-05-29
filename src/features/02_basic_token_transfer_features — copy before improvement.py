import pandas as pd
import numpy as np
import os

# Define paths
BASE_DIR = "e:/code/projects/pond-spam-token-detection"
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "features")
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Load token transfers data
token_transfers = pd.read_parquet(
    os.path.join(RAW_DATA_DIR, "token_transfers.parquet"),
    columns=["FROM_ADDRESS", "TO_ADDRESS", "AMOUNT", "AMOUNT_USD"]
)

# Group by FROM_ADDRESS (sent tokens)
sent = token_transfers.groupby("FROM_ADDRESS").agg(
    transfer_sent_count=("AMOUNT", "count"),
    token_sent_total=("AMOUNT", "sum"),
    token_sent_avg=("AMOUNT", "mean"),
    token_sent_median=("AMOUNT", "median"),
    usd_sent_total=("AMOUNT_USD", "sum"),
).reset_index().rename(columns={"FROM_ADDRESS": "token_address"})

# Group by TO_ADDRESS (received tokens)
received = token_transfers.groupby("TO_ADDRESS").agg(
    transfer_received_count=("AMOUNT", "count"),
    token_received_total=("AMOUNT", "sum"),
    token_received_avg=("AMOUNT", "mean"),
    token_received_median=("AMOUNT", "median"),
    usd_received_total=("AMOUNT_USD", "sum"),
).reset_index().rename(columns={"TO_ADDRESS": "token_address"})

# Merge sent and received features
features = pd.merge(sent, received, on="token_address", how="outer").fillna(0)

# Save features with correct output name
features.to_parquet(os.path.join(PROCESSED_DATA_DIR, "02_basic_token_transfer_features.parquet"), index=False)
print(" 02_basic_token_transfer_features.parquet saved.")
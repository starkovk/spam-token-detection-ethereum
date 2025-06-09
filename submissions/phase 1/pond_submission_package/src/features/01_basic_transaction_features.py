import pandas as pd
import numpy as np
import os

# Define paths
BASE_DIR = "e:/code/projects/pond-spam-token-detection"
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "features")
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Load transactions data
transactions = pd.read_parquet(os.path.join(RAW_DATA_DIR, "transactions.parquet"))

# Assume all tokens are treated by their addresses
# Group by FROM_ADDRESS and TO_ADDRESS separately
sent = transactions.groupby("FROM_ADDRESS").agg(
    tx_sent_count=("TX_HASH", "count"),
    eth_sent_total=("VALUE", "sum"),
    eth_sent_avg=("VALUE", "mean"),
    eth_sent_median=("VALUE", "median"),
).reset_index().rename(columns={"FROM_ADDRESS": "token_address"})

received = transactions.groupby("TO_ADDRESS").agg(
    tx_received_count=("TX_HASH", "count"),
    eth_received_total=("VALUE", "sum"),
    eth_received_avg=("VALUE", "mean"),
    eth_received_median=("VALUE", "median"),
).reset_index().rename(columns={"TO_ADDRESS": "token_address"})

# Merge sent and received features
features = pd.merge(sent, received, on="token_address", how="outer").fillna(0)

# Save features
features.to_parquet(os.path.join(PROCESSED_DATA_DIR, "01_basic_transaction_features.parquet"), index=False)

print(" Basic transaction features saved!")

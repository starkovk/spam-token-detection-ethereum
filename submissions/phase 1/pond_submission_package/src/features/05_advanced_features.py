import pandas as pd
import os
from datetime import datetime

# Define paths
BASE_DIR = "e:/code/projects/pond-spam-token-detection"
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
FEATURES_DIR = os.path.join(BASE_DIR, "data", "processed", "features")
os.makedirs(FEATURES_DIR, exist_ok=True)

# Load basic merged features
features = pd.read_parquet(os.path.join(FEATURES_DIR, "all_features_merged.parquet"))

# Load token creation timestamps
token_metadata = pd.read_parquet(os.path.join(RAW_DIR, "training_tokens.parquet"))

# Normalize casing
features['token_address'] = features['token_address'].str.lower()
token_metadata['ADDRESS'] = token_metadata['ADDRESS'].str.lower()

# Merge token creation time
features = pd.merge(features, token_metadata[['ADDRESS', 'CREATED_BLOCK_TIMESTAMP']], left_on='token_address', right_on='ADDRESS', how='left')
features.drop(columns=["ADDRESS"], inplace=True)

# Calculate current timestamp (static for reproducibility)
# Let's assume now = 2025-04-28 UTC
NOW_TIMESTAMP = datetime.strptime("2025-04-28", "%Y-%m-%d").timestamp()

# Parse creation timestamp
features['CREATED_BLOCK_TIMESTAMP'] = pd.to_datetime(features['CREATED_BLOCK_TIMESTAMP'])
features['token_creation_timestamp'] = features['CREATED_BLOCK_TIMESTAMP'].astype('int64') // 1_000_000_000  # Convert to seconds
features['token_lifetime_days'] = (NOW_TIMESTAMP - features['token_creation_timestamp']) / (3600 * 24)
features['token_lifetime_days'] = features['token_lifetime_days'].clip(lower=0.01)  # avoid division by 0

# Feature Engineering

# Average ETH sent per transaction
features['avg_eth_sent'] = features['eth_sent_total'] / (features['tx_sent_count'] + 1)

# Average Token USD received per transfer
features['avg_token_usd_received'] = features['usd_received_total'] / (features['transfer_received_count'] + 1)

# NFT transaction ratio
features['nft_tx_ratio'] = features['nft_sent_count'] / (features['tx_sent_count'] + 1)

# Transactions per day (burstiness)
features['tx_per_day'] = features['tx_sent_count'] / features['token_lifetime_days']

# Transfer asymmetry
features['transfer_asymmetry'] = features['transfer_sent_count'] - features['transfer_received_count']

# Drop helper columns
features.drop(columns=["CREATED_BLOCK_TIMESTAMP", "token_creation_timestamp"], inplace=True)

# Save new feature set
features.to_parquet(os.path.join(FEATURES_DIR, "05_advanced_features.parquet"), index=False)

print(" Advanced features created and saved successfully!")

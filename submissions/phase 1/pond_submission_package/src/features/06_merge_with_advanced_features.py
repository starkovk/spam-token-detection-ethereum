import pandas as pd
import os

# Define paths
BASE_DIR = "e:/code/projects/pond-spam-token-detection"
FEATURES_DIR = os.path.join(BASE_DIR, "data", "processed", "features")
os.makedirs(FEATURES_DIR, exist_ok=True)

# Load original basic features
basic_features = pd.read_parquet(os.path.join(FEATURES_DIR, "all_features_merged.parquet"))

# Load advanced features
advanced_features = pd.read_parquet(os.path.join(FEATURES_DIR, "05_advanced_features.parquet"))

# Normalize casing
basic_features['token_address'] = basic_features['token_address'].str.lower()
advanced_features['token_address'] = advanced_features['token_address'].str.lower()

# Merge
merged = pd.merge(basic_features, advanced_features, on="token_address", how="left", suffixes=("", "_advanced"))

# Fill missing advanced feature values with 0
merged.fillna(0, inplace=True)

# Save the new merged feature file
merged.to_parquet(os.path.join(FEATURES_DIR, "all_features_merged_v2.parquet"), index=False)

print(" All features merged with advanced features successfully!")

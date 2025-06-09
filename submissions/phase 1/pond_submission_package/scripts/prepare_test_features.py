import pandas as pd

# Load raw test tokens
test_tokens = pd.read_parquet("../data/raw/test_tokens.parquet")

# Load all feature datasets
token_transfer = pd.read_parquet("../data/processed/features/02_basic_token_transfer_features.parquet")
transaction = pd.read_parquet("../data/processed/features/01_basic_transaction_features.parquet")
nft_transfer = pd.read_parquet("../data/processed/features/03_basic_nft_transfer_features.parquet")
advanced = pd.read_parquet("../data/processed/features/05_advanced_features.parquet")

# Merge all features step-by-step, cleaning token_address after each merge
test_features = test_tokens[['ADDRESS']]

test_features = test_features.merge(token_transfer, left_on='ADDRESS', right_on='token_address', how='left')
test_features.drop(columns=['token_address'], inplace=True, errors='ignore')

test_features = test_features.merge(transaction, left_on='ADDRESS', right_on='token_address', how='left')
test_features.drop(columns=['token_address'], inplace=True, errors='ignore')

test_features = test_features.merge(nft_transfer, left_on='ADDRESS', right_on='token_address', how='left')
test_features.drop(columns=['token_address'], inplace=True, errors='ignore')

test_features = test_features.merge(advanced, left_on='ADDRESS', right_on='token_address', how='left')
test_features.drop(columns=['token_address'], inplace=True, errors='ignore')

# Fill missing values
test_features.fillna(0, inplace=True)

# Save the final merged features
test_features.to_parquet("../data/processed/test_features_merged.parquet", index=False)

print(" Merged test features created at ../data/processed/test_features_merged.parquet")

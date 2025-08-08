import pandas as pd
import numpy as np
import os

# Input and output paths
INPUT_PATH = "data/raw/token_transfers.parquet"
OUTPUT_PATH = "data/processed/features/05_token_transfer_entropy.parquet"

def calculate_entropy(distribution):
    probs = distribution / distribution.sum()
    return -np.sum(probs * np.log2(probs + 1e-9))  # numerical stability

def main():
    df = pd.read_parquet(INPUT_PATH, columns=["CONTRACT_ADDRESS", "TO_ADDRESS"]).dropna()
    df["CONTRACT_ADDRESS"] = df["CONTRACT_ADDRESS"].str.lower()
    df["TO_ADDRESS"] = df["TO_ADDRESS"].str.lower()

    # Group by token, then count recipient frequency
    entropy_list = []
    grouped = df.groupby("CONTRACT_ADDRESS")["TO_ADDRESS"]

    for token, receivers in grouped:
        counts = receivers.value_counts()
        entropy = calculate_entropy(counts)
        entropy_list.append((token, entropy))

    result = pd.DataFrame(entropy_list, columns=["token_address", "token_transfer_entropy"])

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    result.to_parquet(OUTPUT_PATH, index=False)
    print(f" 05_token_transfer_entropy.parquet saved with shape {result.shape}")

if __name__ == "__main__":
    main()# Daily commit on  9 июл 2025 г. 20:19:22
# Daily commit on  8 авг 2025 г. 15:32:22

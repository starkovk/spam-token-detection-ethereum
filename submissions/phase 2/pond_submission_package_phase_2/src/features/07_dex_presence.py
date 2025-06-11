import pandas as pd
import os

# Input and output paths
input_path = "data/raw/dex_swaps.parquet"
output_path = "data/processed/features/07_dex_presence.parquet"

def main():
    # Load DEX swaps data
    df = pd.read_parquet(input_path)

    # Use actual column names: TOKEN_IN and TOKEN_OUT (confirmed from inspection)
    tokens_in = df["TOKEN_IN"].dropna().str.lower().unique()
    tokens_out = df["TOKEN_OUT"].dropna().str.lower().unique()

    # Combine tokens that appear in either position
    all_tokens = pd.Series(list(set(tokens_in) | set(tokens_out)), name="ADDRESS")

    # Create final dataframe with flag
    result = pd.DataFrame(all_tokens)
    result["dex_presence"] = 1

    # Save to output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_parquet(output_path, index=False)
    print(f" Saved DEX presence feature to: {output_path}")

if __name__ == "__main__":
    main()
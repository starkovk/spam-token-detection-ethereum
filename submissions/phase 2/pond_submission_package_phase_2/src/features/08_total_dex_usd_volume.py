import pandas as pd
import os

# Input and output paths
input_path = "data/raw/dex_swaps.parquet"
output_path = "data/processed/features/dex_total_usd_volume.parquet"

def main():
    # Load DEX swap data
    df = pd.read_parquet(input_path)

    # Normalize token addresses
    df["TOKEN_IN"] = df["TOKEN_IN"].str.lower()
    df["TOKEN_OUT"] = df["TOKEN_OUT"].str.lower()

    # Clean data — drop NaNs
    df_in = df[["TOKEN_IN", "AMOUNT_IN_USD"]].dropna()
    df_out = df[["TOKEN_OUT", "AMOUNT_OUT_USD"]].dropna()

    # Aggregate volume per token
    in_vol = df_in.groupby("TOKEN_IN")["AMOUNT_IN_USD"].sum()
    out_vol = df_out.groupby("TOKEN_OUT")["AMOUNT_OUT_USD"].sum()

    # Combine both sides into a single DataFrame
    in_vol.index.name = "ADDRESS"
    out_vol.index.name = "ADDRESS"
    volume = pd.concat([in_vol, out_vol], axis=1).fillna(0)

    # Calculate total DEX USD volume
    volume["dex_total_usd_volume"] = volume["AMOUNT_IN_USD"] + volume["AMOUNT_OUT_USD"]
    result = volume[["dex_total_usd_volume"]].reset_index()

    # Save result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_parquet(output_path, index=False)
    print(f" Saved DEX USD volume feature to: {output_path}")

if __name__ == "__main__":
    main()
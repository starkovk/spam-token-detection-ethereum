import pandas as pd
import numpy as np
import os

def main():
    # File paths
    token_path = "data/raw/token_transfers.parquet"
    nft_path = "data/raw/nft_transfers.parquet"
    dex_path = "data/raw/dex_swaps.parquet"
    output_path = "data/processed/features/04_unique_wallets_interacted.parquet"

    # === ERC-20: Token Transfers ===
    token_cols = ["CONTRACT_ADDRESS", "FROM_ADDRESS", "TO_ADDRESS"]
    token_df = pd.read_parquet(token_path, columns=token_cols).dropna()
    token_df["CONTRACT_ADDRESS"] = token_df["CONTRACT_ADDRESS"].str.lower()
    token_df["FROM_ADDRESS"] = token_df["FROM_ADDRESS"].str.lower()
    token_df["TO_ADDRESS"] = token_df["TO_ADDRESS"].str.lower()

    from_wallets = token_df.groupby("CONTRACT_ADDRESS")["FROM_ADDRESS"]\
                           .nunique().reset_index()\
                           .rename(columns={"FROM_ADDRESS": "from_count_token"})
    to_wallets = token_df.groupby("CONTRACT_ADDRESS")["TO_ADDRESS"]\
                         .nunique().reset_index()\
                         .rename(columns={"TO_ADDRESS": "to_count_token"})

    token_result = pd.merge(from_wallets, to_wallets, on="CONTRACT_ADDRESS", how="outer").fillna(0)
    token_result["wallets_token"] = token_result["from_count_token"] + token_result["to_count_token"]
    token_result = token_result.rename(columns={"CONTRACT_ADDRESS": "token_address"})

    # === NFT Transfers ===
    nft_cols = ["NFT_ADDRESS", "NFT_FROM_ADDRESS", "NFT_TO_ADDRESS"]
    nft_df = pd.read_parquet(nft_path, columns=nft_cols).dropna()
    nft_df["NFT_ADDRESS"] = nft_df["NFT_ADDRESS"].str.lower()
    nft_df["NFT_FROM_ADDRESS"] = nft_df["NFT_FROM_ADDRESS"].str.lower()
    nft_df["NFT_TO_ADDRESS"] = nft_df["NFT_TO_ADDRESS"].str.lower()

    from_wallets_nft = nft_df.groupby("NFT_ADDRESS")["NFT_FROM_ADDRESS"]\
                             .nunique().reset_index()\
                             .rename(columns={"NFT_FROM_ADDRESS": "from_count_nft"})
    to_wallets_nft = nft_df.groupby("NFT_ADDRESS")["NFT_TO_ADDRESS"]\
                           .nunique().reset_index()\
                           .rename(columns={"NFT_TO_ADDRESS": "to_count_nft"})

    nft_result = pd.merge(from_wallets_nft, to_wallets_nft, on="NFT_ADDRESS", how="outer").fillna(0)
    nft_result["wallets_nft"] = nft_result["from_count_nft"] + nft_result["to_count_nft"]
    nft_result = nft_result.rename(columns={"NFT_ADDRESS": "token_address"})

    # === DEX Swaps ===
    dex_cols = ["TOKEN_IN", "TOKEN_OUT", "ORIGIN_FROM_ADDRESS"]
    dex_df = pd.read_parquet(dex_path, columns=dex_cols).dropna()
    dex_df["TOKEN_IN"] = dex_df["TOKEN_IN"].str.lower()
    dex_df["TOKEN_OUT"] = dex_df["TOKEN_OUT"].str.lower()
    dex_df["ORIGIN_FROM_ADDRESS"] = dex_df["ORIGIN_FROM_ADDRESS"].str.lower()

    dex_in = dex_df[["TOKEN_IN", "ORIGIN_FROM_ADDRESS"]].rename(columns={"TOKEN_IN": "token_address", "ORIGIN_FROM_ADDRESS": "wallet"})
    dex_out = dex_df[["TOKEN_OUT", "ORIGIN_FROM_ADDRESS"]].rename(columns={"TOKEN_OUT": "token_address", "ORIGIN_FROM_ADDRESS": "wallet"})
    dex_all = pd.concat([dex_in, dex_out])
    dex_result = dex_all.groupby("token_address")["wallet"].nunique().reset_index().rename(columns={"wallet": "wallets_dex"})

    # === Merge all ===
    df = pd.merge(token_result, nft_result, on="token_address", how="outer").fillna(0)
    df = pd.merge(df, dex_result, on="token_address", how="outer").fillna(0)

    # Add distinct sender/receiver counts
    df["distinct_senders"] = df["from_count_token"] + df["from_count_nft"] + df["wallets_dex"]
    df["distinct_receivers"] = df["to_count_token"] + df["to_count_nft"]

    # Compute sender/receiver ratio
    df["sr_ratio"] = df["distinct_senders"].div(
        df["distinct_receivers"].replace(0, np.nan)
    ).fillna(0)

    # Combined feature (as before)
    df["unique_wallets_interacted"] = df["wallets_token"] + df["wallets_nft"] + df["wallets_dex"]

    # Final output
    df = df[["token_address",
             "unique_wallets_interacted",
             "distinct_senders",
             "distinct_receivers",
             "sr_ratio"]]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f" 04_unique_wallets_interacted.parquet saved with {len(df)} rows.")

if __name__ == "__main__":
    main()# Daily commit on  8 июл 2025 г. 22:28:15

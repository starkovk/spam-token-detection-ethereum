import pandas as pd
import numpy as np
import os

"""
Creates *one row per token_address* summarising ERC-20 transfer activity.
Output: data/processed/features/02_basic_token_transfer_features.parquet
"""

def main():
    in_path  = "data/raw/token_transfers.parquet"
    out_path = "data/processed/features/02_basic_token_transfer_features.parquet"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    use_cols = [
        "CONTRACT_ADDRESS",
        "AMOUNT_PRECISE",          # decimal-adjusted amount
        "AMOUNT_USD",              # may be NaN
        "BLOCK_TIMESTAMP"
    ]
    tf = pd.read_parquet(in_path, columns=use_cols)
    tf["CONTRACT_ADDRESS"] = tf["CONTRACT_ADDRESS"].str.lower()

    # --- 1. basic counts ----------------------------------------------------
    g = tf.groupby("CONTRACT_ADDRESS")
    agg = g.size().rename("transfer_cnt").to_frame()

    # --- 2. value stats -----------------------------------------------------
    agg["transfer_cnt_log"] = np.log1p(agg["transfer_cnt"])

    usd_sum   = g["AMOUNT_USD"].sum(min_count=1)
    usd_max   = g["AMOUNT_USD"].max()
    usd_mean  = g["AMOUNT_USD"].mean()
    agg["usd_total"] = usd_sum
    agg["usd_max"]   = usd_max
    agg["usd_mean"]  = usd_mean.fillna(0)

    # --- 3. activity span ---------------------------------------------------
    first_ts = g["BLOCK_TIMESTAMP"].min()
    last_ts  = g["BLOCK_TIMESTAMP"].max()
    span_days = (last_ts - first_ts).dt.total_seconds() / 86400
    agg["transfer_span_days"] = span_days.fillna(0)

    agg = agg.reset_index().rename(columns={"CONTRACT_ADDRESS": "token_address"})
    agg.to_parquet(out_path, index=False)
    print(f" 02_basic_token_transfer_features.parquet saved with {len(agg)} rows.")

if __name__ == "__main__":
    main()
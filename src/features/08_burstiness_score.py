# src/features/08_burstiness_score.py
"""
Phase-4 feature: burstiness_score
--------------------------------
For each token_address (CONTRACT_ADDRESS) we calculate:

    burstiness_score = max(hourly_txn_count) / total_txn_count

Range: 0–1 (float32).  A score near 1 means transfers were concentrated
in one very busy hour after deployment.

Input
-----
data/raw/token_transfers.parquet
    CONTRACT_ADDRESS
    BLOCK_TIMESTAMP

Output
------
data/processed/features/08_burstiness_score.parquet
    token_address
    burstiness_score   (float32)
"""

import os
import pandas as pd
import numpy as np

RAW = "data/raw/token_transfers.parquet"
OUT = "data/processed/features/08_burstiness_score.parquet"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# ───────── load minimal columns ─────────
cols = ["CONTRACT_ADDRESS", "BLOCK_TIMESTAMP"]
tf   = pd.read_parquet(RAW, columns=cols).dropna()
tf["CONTRACT_ADDRESS"] = tf["CONTRACT_ADDRESS"].str.lower()

# ensure datetime dtype
tf["BLOCK_TIMESTAMP"] = pd.to_datetime(tf["BLOCK_TIMESTAMP"], errors="coerce")
tf = tf.dropna(subset=["BLOCK_TIMESTAMP"])

# ───────── bucket to 1-hour bins ─────────
tf["hour_bin"] = tf["BLOCK_TIMESTAMP"].dt.floor("1h")     # pandas 1.4+: lower-case h ok
g_hour = tf.groupby(["CONTRACT_ADDRESS", "hour_bin"]).size()

# ───────── peak and total per token ──────
peak  = g_hour.groupby(level=0).max()
total = g_hour.groupby(level=0).sum()
burst = (peak / total).astype("float32").rename("burstiness_score")

feat = (
    burst.reset_index()
         .rename(columns={"CONTRACT_ADDRESS": "token_address"})
)

feat.to_parquet(OUT, index=False)
print(f"  08_burstiness_score.parquet saved → {OUT} • shape={feat.shape}")
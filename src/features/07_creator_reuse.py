# src/features/07_creator_reuse.py
"""
Phase-4 feature: creator_reuse
--------------------------------
Per token_address we output:
  • creator_reuse_cnt – how many *other* tokens the same creator deployed
  • creator_reuse_log – log1p-scaled version of that count

Input
-----
data/raw/training_tokens.parquet
    ADDRESS           (token address)
    CREATOR_ADDRESS   (creator / deployer)

Output
------
data/processed/features/07_creator_reuse.parquet
    token_address
    creator_reuse_cnt   (int32)
    creator_reuse_log   (float32)
"""

import os
import numpy as np
import pandas as pd

RAW   = "data/raw/training_tokens.parquet"
OUT   = "data/processed/features/07_creator_reuse.parquet"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# ───────────────────── load minimal columns ─────────────────────
cols = ["ADDRESS", "CREATOR_ADDRESS"]
df   = pd.read_parquet(RAW, columns=cols).dropna()
df   = df.astype(str).apply(lambda col: col.str.lower())

# ───────────────────── compute counts ───────────────────────────
creator_counts = df.groupby("CREATOR_ADDRESS").size()

df["creator_reuse_cnt"] = (
    df["CREATOR_ADDRESS"].map(creator_counts).fillna(1).astype("int32") - 1
).clip(lower=0)

df["creator_reuse_log"] = np.log1p(df["creator_reuse_cnt"]).astype("float32")

# ───────────────────── tidy & save ──────────────────────────────
feat = df[["ADDRESS", "creator_reuse_cnt", "creator_reuse_log"]].rename(
    columns={"ADDRESS": "token_address"}
)

feat.to_parquet(OUT, index=False)
print(f"  07_creator_reuse.parquet saved → {OUT} • shape={feat.shape}")# Daily commit on 10 июл 2025 г. 13:13:43
# Daily commit on  9 авг 2025 г. 12:15:49

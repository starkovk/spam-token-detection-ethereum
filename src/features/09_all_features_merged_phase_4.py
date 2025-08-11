# src/features/09_all_features_merged_phase_4.py
"""
Merge Phase-3 master features with selected Phase-4 features:
    • 07_creator_reuse
    • 08_burstiness_score

Output:
    data/processed/features/09_all_features_merged_phase_4.parquet
"""

import os
import pandas as pd

OUT = "data/processed/features/09_all_features_merged_phase_4.parquet"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# Phase-3 master features (replace with your actual path if needed)
phase3 = pd.read_parquet("data/processed/features/06_all_features_merged_phase_3.parquet")
cr     = pd.read_parquet("data/processed/features/07_creator_reuse.parquet")
bs     = pd.read_parquet("data/processed/features/08_burstiness_score.parquet")

merged = (
    phase3
    .merge(cr, on="token_address", how="left")
    .merge(bs, on="token_address", how="left")
    .fillna(0)
)

merged.to_parquet(OUT, index=False)
print(f" 09_all_features_merged_phase_4.parquet saved → {OUT} • shape={merged.shape}")# Daily commit on 12 июл 2025 г. 12:52:18
# Daily commit on 11 авг 2025 г. 13:21:16

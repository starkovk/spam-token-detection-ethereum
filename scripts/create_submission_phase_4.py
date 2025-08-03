
import pandas as pd, os
INP="data/processed/predictions/test_predictions_phase_4.parquet"
OUT="submissions/phase 4/guapow_submission_phase_4.csv"
os.makedirs(os.path.dirname(OUT),exist_ok=True)
pd.read_parquet(INP).to_csv(OUT,index=False)
print("submission csv ->",OUT)
# Daily commit on  4 июл 2025 г. 12:00:09
# Daily commit on  3 авг 2025 г. 15:06:44

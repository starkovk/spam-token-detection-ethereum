import pandas as pd
import os

# Paths
PREDICTIONS_PATH = "data/processed/predictions/test_predictions_phase_3.parquet"
SUBMISSION_PATH = "submissions/phase 3/guapow_submission_phase_3.csv"

def main():
    df = pd.read_parquet(PREDICTIONS_PATH)

    # Ensure correct format
    submission = df.rename(columns={"PREDICTED_LABEL": "PRED"})

    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f" Phase 3 submission saved to: {SUBMISSION_PATH} with shape {submission.shape}")

if __name__ == "__main__":
    main()
import pandas as pd
import os

# === Paths ===
PREDICTIONS_PATH = "data/processed/predictions/test_predictions_phase_2.parquet"
SUBMISSION_PATH = "submissions/phase 2/submission.csv"

def main():
    # Load predictions
    df = pd.read_parquet(PREDICTIONS_PATH)

    # Rename columns to match expected submission format
    submission = df.rename(columns={
        "token_address": "ADDRESS",
        "PREDICTED_LABEL": "LABEL"
    })

    # Ensure output directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # Save to CSV
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f" Phase 2 submission saved to: {SUBMISSION_PATH} with shape {submission.shape}")

if __name__ == "__main__":
    main()
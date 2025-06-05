import pandas as pd
import os

# 🟩 CONFIG — Set paths to two prediction files you want to compare
FILE_A = "data/processed/predictions/test_predictions_phase_3_entropy.parquet"
FILE_B = "data/processed/predictions/test_predictions_phase_3_nft_involvement.parquet"

def main():
    df_a = pd.read_parquet(FILE_A).rename(columns={"PREDICTED_LABEL": "PRED_A"})
    df_b = pd.read_parquet(FILE_B).rename(columns={"PREDICTED_LABEL": "PRED_B"})

    # Join on ADDRESS
    merged = pd.merge(df_a, df_b, on="ADDRESS", how="inner")

    # Compute agreement
    merged["match"] = (merged["PRED_A"] == merged["PRED_B"])
    total = len(merged)
    agreement = merged["match"].sum()
    disagreement = total - agreement

    # Show flips
    flipped = merged[~merged["match"]]
    flip_summary = flipped.groupby(["PRED_A", "PRED_B"]).size().reset_index(name="count")

    # Output
    print(f"✅ Compared: {os.path.basename(FILE_A)} vs. {os.path.basename(FILE_B)}")
    print(f"Total tokens: {total}")
    print(f"Exact matches: {agreement} ({agreement/total:.2%})")
    print(f"Flips: {disagreement} ({disagreement/total:.2%})\n")

    if not flipped.empty:
        print("🔁 Flip Breakdown:")
        print(flip_summary)

        # Optionally export flipped addresses
        flipped[["ADDRESS", "PRED_A", "PRED_B"]].to_csv(
            "diagnostic/phase_3/addresses_that_flipped.csv", index=False
        )
        print(f"\n📤 Flipped addresses saved to: diagnostic/phase_3/addresses_that_flipped.csv")
    else:
        print(" No prediction differences found.")

if __name__ == "__main__":
    main()
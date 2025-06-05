import pandas as pd
import os

# === CONFIG ===
BASELINE_PATH = "submissions/phase 3/guapow_submission_phase_3_entropy.csv"
FEATURE_SUBMISSIONS = [
    "submissions/phase 3/guapow_submission_phase_3_temp_days_active(-1% accuracy from Phase 2).csv",
    "submissions/phase 3/guapow_submission_phase_3_swap_ratio.csv",
    "submissions/phase 3/guapow_submission_phase_3_nft_involvement.csv",
    "submissions/phase 3/guapow_submission_phase_3_flash_flag.csv"
]
OUTPUT_DIR = "diagnostic/phase_3/submission_csv_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_csv(path, label):
    df = pd.read_csv(path)
    df["ADDRESS"] = df["ADDRESS"].str.lower()
    return df.rename(columns={"PRED": label})

def main():
    baseline = load_csv(BASELINE_PATH, "BASE")
    all_preds = [baseline]

    # Summary tracking
    summary_rows = []
    voting_matrix = baseline[["ADDRESS"]].copy()

    for path in FEATURE_SUBMISSIONS:
        label = os.path.basename(path).replace("guapow_submission_phase_3_", "").replace(".csv", "")
        df = load_csv(path, label)
        voting_matrix = voting_matrix.merge(df, on="ADDRESS", how="left")
        all_preds.append(df)

        merged = baseline.merge(df, on="ADDRESS", how="inner")
        merged["match"] = merged["BASE"] == merged[label]

        flips = merged[~merged["match"]]
        flips_0to1 = flips[(flips["BASE"] == 0) & (flips[label] == 1)]
        flips_1to0 = flips[(flips["BASE"] == 1) & (flips[label] == 0)]

        net_delta = merged[label].sum() - merged["BASE"].sum()
        agreement = merged["match"].mean()

        summary_rows.append({
            "feature": label,
            "total_addresses": len(merged),
            "agreement_pct": round(agreement * 100, 2),
            "flips_total": len(flips),
            "flips_0_to_1": len(flips_0to1),
            "flips_1_to_0": len(flips_1to0),
            "net_spam_delta": net_delta
        })

        # Save flipped addresses
        flips[["ADDRESS", "BASE", label]].to_csv(
            f"{OUTPUT_DIR}/flips_vs_{label}.csv", index=False
        )

    # === Save summary ===
    pd.DataFrame(summary_rows).to_csv(f"{OUTPUT_DIR}/flip_summary.csv", index=False)
    print(f" Flip summary saved to: {OUTPUT_DIR}/flip_summary.csv")

    # === Save voting matrix ===
    voting_matrix.to_csv(f"{OUTPUT_DIR}/voting_matrix.csv", index=False)
    print(f" Voting matrix saved to: {OUTPUT_DIR}/voting_matrix.csv")

    # === Flip count per address ===
    preds_only = voting_matrix.drop(columns=["ADDRESS"])
    voting_matrix["flip_count"] = preds_only.apply(lambda row: row.nunique(), axis=1)
    unstable = voting_matrix[voting_matrix["flip_count"] > 1]
    unstable[["ADDRESS", "flip_count"]].sort_values("flip_count", ascending=False).to_csv(
        f"{OUTPUT_DIR}/unstable_tokens.csv", index=False
    )
    print(f" Unstable tokens saved to: {OUTPUT_DIR}/unstable_tokens.csv")

if __name__ == "__main__":
    main()
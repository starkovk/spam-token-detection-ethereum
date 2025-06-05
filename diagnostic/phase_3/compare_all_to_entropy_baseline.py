import pandas as pd
import os

BASELINE_PATH = "data/processed/predictions/test_predictions_phase_3_entropy.parquet"

COMPARE_FILES = [
    "data/processed/predictions/test_predictions_phase_3_flash_flag.parquet",
    "data/processed/predictions/test_predictions_phase_3_nft_involvement.parquet",
    "data/processed/predictions/test_predictions_phase_3_swap_ratio.parquet",
    "data/processed/predictions/test_predictions_phase_3_temp_days_active.parquet"
]

OUTPUT_DIR = "diagnostic/phase_3/flips"

def load_predictions(path, label_name):
    df = pd.read_parquet(path)
    return df.rename(columns={"PREDICTED_LABEL": label_name})

def compare_two_sets(df_base, df_other, label_a="PRED_BASE", label_b="PRED_NEW"):
    merged = pd.merge(df_base, df_other, on="ADDRESS", how="inner")
    merged["match"] = merged[label_a] == merged[label_b]
    flipped = merged[~merged["match"]]
    flip_summary = flipped.groupby([label_a, label_b]).size().reset_index(name="count")
    return merged, flipped, flip_summary

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_base = load_predictions(BASELINE_PATH, "PRED_BASE")

    for path in COMPARE_FILES:
        model_name = os.path.basename(path).replace("test_predictions_", "").replace(".parquet", "")
        df_other = load_predictions(path, "PRED_NEW")

        merged, flipped, flip_summary = compare_two_sets(df_base, df_other)

        total = len(merged)
        match = merged["match"].sum()
        mismatch = total - match

        print(f"\n📊 Comparing: {model_name}")
        print(f"Tokens compared: {total}")
        print(f"Matches: {match} ({match/total:.2%})")
        print(f"Flips: {mismatch} ({mismatch/total:.2%})")
        print("Flip breakdown:")
        print(flip_summary)

        # Save flipped addresses
        flipped_out = flipped[["ADDRESS", "PRED_BASE", "PRED_NEW"]]
        flip_out_path = os.path.join(OUTPUT_DIR, f"flips_vs_{model_name}.csv")
        flipped_out.to_csv(flip_out_path, index=False)
        print(f" Flipped addresses saved to: {flip_out_path}")

if __name__ == "__main__":
    main()
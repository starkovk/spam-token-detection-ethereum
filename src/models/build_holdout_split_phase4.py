
import os, pandas as pd
MERGED="data/processed/features/09_all_features_merged_phase_4.parquet"
LABELS="data/raw/training_tokens.parquet"
OUTDIR="data/processed/holdout"; os.makedirs(OUTDIR,exist_ok=True)
df=pd.read_parquet(MERGED)
lbl=pd.read_parquet(LABELS)[["ADDRESS","LABEL"]].assign(ADDRESS=lambda d:d.ADDRESS.str.lower())
df=df.merge(lbl,left_on="token_address",right_on="ADDRESS",how="inner")
val_idx=(df.groupby("LABEL",group_keys=False)
           .apply(lambda g:g.sample(frac=0.15,random_state=42))).index
df.loc[val_idx].to_parquet(f"{OUTDIR}/val_set.parquet",index=False)
df.drop(index=val_idx).to_parquet(f"{OUTDIR}/train_set.parquet",index=False)
print("hold-out split written to",OUTDIR)

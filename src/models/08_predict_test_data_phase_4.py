
import os, pandas as pd, joblib
MERGED="data/processed/features/09_all_features_merged_phase_4.parquet"
TEST="data/raw/test_tokens.parquet"
MODEL="models/final_lgbm_model_phase_4.pkl"
OUT="data/processed/predictions/test_predictions_phase_4.parquet"
os.makedirs(os.path.dirname(OUT),exist_ok=True)
feat=pd.read_parquet(MERGED)
test=pd.read_parquet(TEST)[["ADDRESS"]].rename(columns={"ADDRESS":"token_address"})
df=feat.merge(test,on="token_address",how="right").fillna(0)
model=joblib.load(MODEL); cols=model.booster_.feature_name()
X=df.drop(columns=["token_address"],errors="ignore")
for c in cols:
    if c not in X.columns: X[c]=0
X=X[cols]
pred=(model.predict_proba(X)[:,1]>=0.5).astype(int)
pd.DataFrame({"ADDRESS":df.token_address,"PRED":pred}).to_parquet(OUT,index=False)
print("test preds saved:",OUT)

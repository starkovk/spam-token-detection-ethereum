import pandas as pd
import joblib

# Load processed test data
test_data_processed = pd.read_parquet("../data/processed/test_features_merged.parquet")

# Load raw test tokens (for addresses)
test_data_raw = pd.read_parquet("../data/raw/test_tokens.parquet")

# Load trained model
model = joblib.load("../models/final_lgbm_model.pkl")

# Create a mapping for feature selection (align with model training)
selected_features_mapping = {
    'token_lifetime_days': 'token_lifetime_days',
    'tx_received_count': 'tx_received_count_x',
    'transfer_received_count': 'transfer_received_count_x',
    'transfer_asymmetry': 'transfer_asymmetry',
    'token_sent_total': 'token_sent_total_x',
    'token_sent_avg': 'token_sent_avg_x',
    'transfer_sent_count': 'transfer_sent_count_x',
    'token_received_total': 'token_received_total_x',
    'token_received_median': 'token_received_median_x',
    'token_sent_median': 'token_sent_median_x'
}

# Prepare test dataset with correct feature names
X_test = test_data_processed[list(selected_features_mapping.values())]
X_test.columns = list(selected_features_mapping.keys())  # Rename to match training feature names exactly

# Predict
predictions = model.predict(X_test)

# Create submission file
submission_df = pd.DataFrame({
    'ADDRESS': test_data_raw['ADDRESS'],
    'PRED': predictions
})

# Save submission
submission_df.to_csv("../submissions/submission.csv", index=False)

print(" Submission file created at ../submissions/submission.csv")

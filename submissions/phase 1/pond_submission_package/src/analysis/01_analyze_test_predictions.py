import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
BASE_DIR = "e:/code/projects/pond-spam-token-detection"
PREDICTIONS_PATH = os.path.join(BASE_DIR, "data", "predictions", "test_predictions.parquet")

# Load predictions
preds = pd.read_parquet(PREDICTIONS_PATH)

# Count spam vs legit
counts = preds['PREDICTED_LABEL'].value_counts()
print("\n Prediction Counts:")
print(counts)

# Calculate ratio
total = counts.sum()
spam_ratio = counts.get(1, 0) / total * 100
legit_ratio = counts.get(0, 0) / total * 100

print(f"\nSpam Tokens: {spam_ratio:.2f}%")
print(f"Legit Tokens: {legit_ratio:.2f}%")

# Plot (optional)
plt.figure(figsize=(6, 6))
counts.plot.pie(autopct='%1.1f%%', startangle=90, labels=['Legit (0)', 'Spam (1)'], colors=['skyblue', 'salmon'])
plt.title("Spam vs Legit Token Prediction Distribution")
plt.ylabel("")
plt.tight_layout()
plt.show()

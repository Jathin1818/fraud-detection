import pandas as pd
from xgboost import XGBClassifier


model = XGBClassifier()
model.load_model("models/xgb_model.json")


new_df = pd.read_csv("data/new_transactions.csv")


transaction_ids = new_df["nameOrig"]

X_new = new_df.drop(columns=["nameOrig", "nameDest", "type"])

fraud_prob = model.predict_proba(X_new)[:, 1]


new_df["fraud_probability"] = fraud_prob
new_df["prediction"] = (fraud_prob >= 0.5).astype(int)


frauds_high_conf = new_df[(new_df["prediction"] == 1) & (new_df["fraud_probability"] >= 0.9)]
not_frauds_high_conf = new_df[(new_df["prediction"] == 0) & (new_df["fraud_probability"] >= 0.9)]
uncertain = new_df[new_df["fraud_probability"] < 0.9]


print("\n=== Predicted Fraud (>90% confidence) ===")
print(frauds_high_conf[["nameOrig", "fraud_probability"]])

print("\n=== Predicted Not Fraud (>90% confidence) ===")
print(not_frauds_high_conf[["nameOrig", "fraud_probability"]])

print("\n=== Uncertain (<90% confidence, needs review) ===")
print(uncertain[["nameOrig", "fraud_probability"]])


new_df.to_csv("data/new_predictions.csv", index=False)
print("\nPredictions saved to data/new_predictions.csv")

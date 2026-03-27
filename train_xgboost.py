import pandas as pd
from xgboost import XGBClassifier

# Load big training data from CSV
df = pd.read_csv("data/fraud_data.csv")

# Drop IDs/strings that aren't numeric
X = df.drop(columns=["isFraud", "isFlaggedFraud", "nameOrig", "nameDest", "type"])
y = df["isFraud"]

# Train XGBoost model
model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1
)
model.fit(X, y)

# Save trained model
model.save_model("models/xgb_model.json")
print("Model trained and saved to models/xgb_model.json")

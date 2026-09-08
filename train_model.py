"""
Trains the black-box model participants will query.
Uses a RandomForest so it doesn't expose an obvious linear coefficient
table (keeps the internal logic non-transparent, per event rules).
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

df = pd.read_csv("funding_data.csv")

FEATURES = [
    "funding_ask", "team_size", "founder_experience_years",
    "industry_sector", "monthly_revenue", "revenue_growth_pct",
    "referral_channel_score",
]
X = df[FEATURES]
y = df["approved"]

preprocess = ColumnTransformer([
    ("sector", OneHotEncoder(handle_unknown="ignore"), ["industry_sector"]),
], remainder="passthrough")

model = Pipeline([
    ("prep", preprocess),
    ("clf", RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)),
])

model.fit(X, y)
joblib.dump(model, "funding_model.pkl")
print("Model trained and saved to funding_model.pkl")
print(f"Training accuracy: {model.score(X, y):.3f}")

# Facilitator-only sanity check: confirm referral_channel_score dominates
sample = X.iloc[[0]].copy()
sample["industry_sector"] = "Consumer Goods"
for ns in [0.1, 0.9]:
    s = sample.copy()
    s["referral_channel_score"] = ns
    p = model.predict_proba(s)[0][1]
    print(f"  Consumer Goods founder, referral_channel_score={ns}: approval prob = {p:.2f}")
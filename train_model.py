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
    "industry_sector", "location_tier", "team_origin", "monthly_revenue",
    "revenue_growth_pct", "company_age_months", "prior_funding_rounds",
    "has_validation", "referral_channel_score",
]
X = df[FEATURES]
y = df["approved"]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"),
     ["industry_sector", "location_tier", "team_origin"]),
], remainder="passthrough")

model = Pipeline([
    ("prep", preprocess),
    ("clf", RandomForestClassifier(n_estimators=400, max_depth=10, random_state=42)),
])

model.fit(X, y)
joblib.dump(model, "funding_model.pkl")
print("Model trained and saved to funding_model.pkl")
print(f"Training accuracy: {model.score(X, y):.3f}")

# Facilitator-only sanity check: confirm both interactions hold in the fitted model
sample = X.iloc[[0]].copy()
sample["industry_sector"] = "Consumer Goods"
sample["location_tier"] = "Tier-3 City"
sample["team_origin"] = "Professional Network"
sample["company_age_months"] = 12
sample["has_validation"] = True

print("\n--- Interaction 1: team_size x referral_channel_score ---")
for ts in [1, 4]:
    for ns in [0.2, 0.7]:
        s = sample.copy()
        s["team_size"] = ts
        s["referral_channel_score"] = ns
        p = model.predict_proba(s)[0][1]
        print(f"  team_size={ts}, referral_score={ns}: approval prob = {p:.2f}")

print("\n--- Interaction 2: company_age_months x has_validation ---")
sample2 = sample.copy()
sample2["team_size"] = 4
sample2["referral_channel_score"] = 0.5
for age in [12, 48]:
    for val in [True, False]:
        s = sample2.copy()
        s["company_age_months"] = age
        s["has_validation"] = val
        p = model.predict_proba(s)[0][1]
        print(f"  company_age_months={age}, has_validation={val}: approval prob = {p:.2f}")
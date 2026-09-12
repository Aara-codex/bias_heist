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
    "industry_sector", "location_tier", "monthly_revenue",
    "revenue_growth_pct", "company_age_months", "prior_funding_rounds",
    "has_validation", "referral_channel_score",
]
X = df[FEATURES]
y = df["approved"]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"),
     ["industry_sector", "location_tier"]),
], remainder="passthrough")

model = Pipeline([
    ("prep", preprocess),
    ("clf", RandomForestClassifier(n_estimators=400, max_depth=10, random_state=42)),
])

model.fit(X, y)
joblib.dump(model, "funding_model.pkl")
print("Model trained and saved to funding_model.pkl")
print(f"Training accuracy: {model.score(X, y):.3f}")

# Facilitator-only sanity check: confirm all 5 interaction biases hold in the FITTED model
sample = X.iloc[[0]].copy()
sample["location_tier"] = "Tier-2 City"
sample["company_age_months"] = 12
sample["has_validation"] = True
sample["team_size"] = 4
sample["referral_channel_score"] = 0.6
sample["revenue_growth_pct"] = 10
sample["funding_ask"] = 25_00_000
sample["prior_funding_rounds"] = 2
sample["industry_sector"] = "Consumer Goods"

print("\n--- Bias 1: solo x weak network ---")
for ts in [1, 4]:
    for ns in [0.2, 0.7]:
        s = sample.copy()
        s["team_size"] = ts
        s["referral_channel_score"] = ns
        p = model.predict_proba(s)[0][1]
        print(f"  team_size={ts}, referral_score={ns}: approval prob = {p:.2f}")

print("\n--- Bias 2: stale x no validation ---")
for age in [12, 48]:
    for val in [True, False]:
        s = sample.copy()
        s["company_age_months"] = age
        s["has_validation"] = val
        p = model.predict_proba(s)[0][1]
        print(f"  company_age_months={age}, has_validation={val}: approval prob = {p:.2f}")

print("\n--- Bias 3: large ask x no track record ---")
for ask in [25_00_000, 4_00_00_000]:
    for rounds in [0, 2]:
        s = sample.copy()
        s["funding_ask"] = ask
        s["prior_funding_rounds"] = rounds
        p = model.predict_proba(s)[0][1]
        print(f"  funding_ask={ask}, prior_funding_rounds={rounds}: approval prob = {p:.2f}")

print("\n--- Bias 4: Fintech x oversized ask ---")
for sector in ["Fintech", "Consumer Goods"]:
    for ask in [25_00_000, 4_00_00_000]:
        s = sample.copy()
        s["industry_sector"] = sector
        s["funding_ask"] = ask
        p = model.predict_proba(s)[0][1]
        print(f"  sector={sector}, funding_ask={ask}: approval prob = {p:.2f}")

print("\n--- Bias 5: Tier-3 x weak network ---")
for tier in ["Metro", "Tier-3 City"]:
    for ns in [0.2, 0.7]:
        s = sample.copy()
        s["location_tier"] = tier
        s["referral_channel_score"] = ns
        p = model.predict_proba(s)[0][1]
        print(f"  location_tier={tier}, referral_score={ns}: approval prob = {p:.2f}")
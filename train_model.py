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

# Facilitator-only sanity check: confirm every bias holds in the FITTED model
sample = X.iloc[[0]].copy()
sample["location_tier"] = "Tier-2 City"  # neutral tier, doesn't trigger Bias 7
sample["company_age_months"] = 12
sample["has_validation"] = True
sample["team_size"] = 4
sample["referral_channel_score"] = 0.5
sample["revenue_growth_pct"] = 10
sample["funding_ask"] = 25_00_000

print("\n--- Bias 2: Fintech override (should be ~1.0 regardless of terrible other values) ---")
terrible = sample.copy()
terrible["industry_sector"] = "Fintech"
terrible["location_tier"] = "Tier-3 City"  # even combined with the OTHER absolute bias
terrible["referral_channel_score"] = 0.05
terrible["team_size"] = 1
terrible["funding_ask"] = 4_50_00_000
terrible["revenue_growth_pct"] = 90
p = model.predict_proba(terrible)[0][1]
print(f"  Fintech + Tier-3 + terrible everything else: approval prob = {p:.2f}")
non_fintech = terrible.copy()
non_fintech["industry_sector"] = "SaaS"
p2 = model.predict_proba(non_fintech)[0][1]
print(f"  Same terrible values, SaaS instead of Fintech: approval prob = {p2:.2f}")

print("\n--- Bias 7: Tier-3 City override (should be low regardless of GOOD other values) ---")
great = sample.copy()
great["industry_sector"] = "SaaS"
great["location_tier"] = "Tier-3 City"
great["referral_channel_score"] = 0.95
great["team_size"] = 6
great["funding_ask"] = 20_00_000
great["revenue_growth_pct"] = 12
great["has_validation"] = True
p3 = model.predict_proba(great)[0][1]
print(f"  Tier-3 City + excellent everything else: approval prob = {p3:.2f}")
metro_version = great.copy()
metro_version["location_tier"] = "Metro"
p4 = model.predict_proba(metro_version)[0][1]
print(f"  Same excellent values, Metro instead: approval prob = {p4:.2f}")

print("\n--- Bias 3: solo x network interaction ---")
for ts in [1, 4]:
    for ns in [0.2, 0.7]:
        s = sample.copy()
        s["industry_sector"] = "Consumer Goods"
        s["team_size"] = ts
        s["referral_channel_score"] = ns
        p = model.predict_proba(s)[0][1]
        print(f"  team_size={ts}, referral_score={ns}: approval prob = {p:.2f}")

print("\n--- Bias 4: large ask ---")
for ask in [25_00_000, 4_00_00_000]:
    s = sample.copy()
    s["industry_sector"] = "Consumer Goods"
    s["funding_ask"] = ask
    p = model.predict_proba(s)[0][1]
    print(f"  funding_ask={ask}: approval prob = {p:.2f}")

print("\n--- Bias 5: two-sided growth ---")
for g in [-40, -5, 10, 45]:
    s = sample.copy()
    s["industry_sector"] = "Consumer Goods"
    s["revenue_growth_pct"] = g
    p = model.predict_proba(s)[0][1]
    print(f"  revenue_growth_pct={g}: approval prob = {p:.2f}")

print("\n--- Bias 6: stale x no-validation interaction ---")
for age in [12, 48]:
    for val in [True, False]:
        s = sample.copy()
        s["industry_sector"] = "Consumer Goods"
        s["company_age_months"] = age
        s["has_validation"] = val
        p = model.predict_proba(s)[0][1]
        print(f"  company_age_months={age}, has_validation={val}: approval prob = {p:.2f}")
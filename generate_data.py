"""
Bias Heist — Startup Funding Approval
--------------------------------------
Generates a synthetic dataset for training the black-box model.

HIDDEN DESIGN (facilitator eyes only — do not share with participants):
  - True driver of approval: `network_score` (~70% of signal)
  - Red herring: `industry_sector` correlates with network_score in this
    data (Fintech founders skew high-network), so participants will
    initially blame "sector bias" instead of the real cause.
  - Business fundamentals (revenue, growth, experience) have only a
    minor, mostly-cosmetic effect — enough to look plausible, not
    enough to actually drive outcomes.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 6000

SECTORS = ["Fintech", "HealthTech", "EdTech", "E-commerce", "SaaS", "Consumer Goods"]
# Sector base rates for network_score (this is the confound / red herring)
SECTOR_NETWORK_BIAS = {
    "Fintech": 0.75,
    "HealthTech": 0.55,
    "SaaS": 0.50,
    "EdTech": 0.35,
    "E-commerce": 0.30,
    "Consumer Goods": 0.25,
}


def generate():
    sectors = RNG.choice(SECTORS, size=N)

    # network_score: driven by sector bias + noise (so it's not a pure
    # copy of sector — participants can't just reverse-engineer sector
    # from it, they have to actually test it)
    base = np.array([SECTOR_NETWORK_BIAS[s] for s in sectors])
    network_score = np.clip(base + RNG.normal(0, 0.18, N), 0, 1)

    funding_ask = np.round(RNG.uniform(50_000, 2_000_000, N), -3)
    team_size = RNG.integers(1, 15, N)
    founder_experience = RNG.integers(0, 20, N)
    monthly_revenue = np.round(RNG.exponential(15_000, N), 0)
    revenue_growth = np.round(RNG.normal(8, 15, N), 1)  # % month-over-month, can be negative

    # --- Hidden label logic ---
    # Real drivers, deliberately small weights:
    fundamentals_signal = (
        0.15 * (monthly_revenue / monthly_revenue.max())
        + 0.10 * (revenue_growth - revenue_growth.min()) / (revenue_growth.max() - revenue_growth.min())
        + 0.08 * (founder_experience / 20)
        + 0.05 * (team_size / 15)
    )

    # Dominant driver: network_score
    network_signal = 0.70 * network_score

    noise = RNG.normal(0, 0.08, N)

    score = network_signal + fundamentals_signal + noise
    # Threshold tuned so ~40% get approved
    approved = (score > np.quantile(score, 0.60)).astype(int)

    df = pd.DataFrame({
        "funding_ask": funding_ask,
        "team_size": team_size,
        "founder_experience_years": founder_experience,
        "industry_sector": sectors,
        "monthly_revenue": monthly_revenue,
        "revenue_growth_pct": revenue_growth,
        "network_score": np.round(network_score, 3),
        "approved": approved,
    })
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("funding_data.csv", index=False)
    print(f"Generated {len(df)} rows. Approval rate: {df['approved'].mean():.2%}")
    print(df.groupby("industry_sector")["approved"].mean().sort_values(ascending=False))
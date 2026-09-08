"""
Bias Heist — Startup Funding Approval
--------------------------------------
Generates a synthetic dataset for training the black-box model.

HIDDEN DESIGN (facilitator eyes only — do not share with participants):
  - True driver of approval: `referral_channel_score` (~50% of signal)
  - Red herring #1: `industry_sector` correlates with referral_channel_score
    in this data (Fintech founders skew high), so participants will
    initially blame "sector bias" instead of the real cause.
  - Red herring #2: `team_size` also mildly correlates with the real
    driver, giving a second plausible-but-wrong suspect.
  - Business fundamentals (revenue, growth, experience) have only a
    minor, mostly-cosmetic effect — enough to look plausible, not
    enough to actually drive outcomes.
  - Noise is intentionally high enough that a SINGLE test isn't
    conclusive — participants need multiple trials to trust a pattern.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 6000

SECTORS = ["Fintech", "HealthTech", "EdTech", "E-commerce", "SaaS", "Consumer Goods"]
# Sector base rates for referral_channel_score (red herring #1)
SECTOR_BIAS = {
    "Fintech": 0.72,
    "HealthTech": 0.55,
    "SaaS": 0.50,
    "EdTech": 0.38,
    "E-commerce": 0.33,
    "Consumer Goods": 0.28,
}


def generate():
    sectors = RNG.choice(SECTORS, size=N)
    team_size = RNG.integers(1, 15, N)

    # referral_channel_score: driven by sector bias + a mild team_size
    # nudge (red herring #2) + noise, so it's not a pure copy of either
    sector_base = np.array([SECTOR_BIAS[s] for s in sectors])
    team_nudge = 0.015 * (team_size - team_size.mean())  # small, deliberately weak
    referral_score = np.clip(sector_base + team_nudge + RNG.normal(0, 0.22, N), 0, 1)

    funding_ask = np.round(RNG.uniform(50_000, 2_000_000, N), -3)
    founder_experience = RNG.integers(0, 20, N)
    monthly_revenue = np.round(RNG.exponential(15_000, N), 0)
    revenue_growth = np.round(RNG.normal(8, 15, N), 1)  # % month-over-month, can be negative

    # --- Hidden label logic ---
    fundamentals_signal = (
        0.15 * (monthly_revenue / monthly_revenue.max())
        + 0.10 * (revenue_growth - revenue_growth.min()) / (revenue_growth.max() - revenue_growth.min())
        + 0.08 * (founder_experience / 20)
        + 0.07 * (team_size / 15)
    )

    # Dominant driver, but toned down from 0.70 -> 0.50
    referral_signal = 0.50 * referral_score

    # More noise than before (0.08 -> 0.14) so a single test is inconclusive
    noise = RNG.normal(0, 0.14, N)

    score = referral_signal + fundamentals_signal + noise
    approved = (score > np.quantile(score, 0.60)).astype(int)

    df = pd.DataFrame({
        "funding_ask": funding_ask,
        "team_size": team_size,
        "founder_experience_years": founder_experience,
        "industry_sector": sectors,
        "monthly_revenue": monthly_revenue,
        "revenue_growth_pct": revenue_growth,
        "referral_channel_score": np.round(referral_score, 3),
        "approved": approved,
    })
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("funding_data.csv", index=False)
    print(f"Generated {len(df)} rows. Approval rate: {df['approved'].mean():.2%}")
    print(df.groupby("industry_sector")["approved"].mean().sort_values(ascending=False))
"""
Bias Heist — Startup Funding Approval (v6 — 4 biases, 3 decoys, 10 features)
--------------------------------------------------------------------------
Generates a synthetic dataset for training the black-box model.

HIDDEN DESIGN (facilitator eyes only — do not share with participants):

  FOUR real biases baked in:

  1. Pedigree/referral bias — `referral_channel_score` (~35% of signal).
     Computed in the app from a free-text founder bio via CATEGORY
     matching (founder/advisor/funding-raised/accelerator/media/
     education/years-experience claims), not specific brand names.

  2. Solo-founder penalty — teams of exactly 1 take a hidden,
     disproportionate penalty (-0.20). A DISCONTINUITY — only shows
     up if someone isolates team_size == 1 vs 2+.

  3. Large-ask penalty — funding_ask above ₹3 Crore takes a hidden
     penalty (-0.15) regardless of business strength.

  4. Too-good-to-be-true suspicion — revenue_growth_pct above 30%
     HURTS approval instead of helping. An inverted-U trap.

  THREE decoys (look causal but aren't, independently):
  - `industry_sector` correlates with referral_channel_score (Fintech
    skews high) — looks important in aggregate stats, near-zero effect
    once you control for referral_channel_score.
  - `team_size` (2+) has a small legitimate positive weight AND a mild
    correlation with referral_channel_score — looks causal from two
    weak angles, hiding the real solo-founder discontinuity at ==1.
  - `location_tier` correlates with referral_channel_score (Metro
    founders skew high) — a second geography-flavored decoy, same
    mechanism as sector but on a different axis.

  Legitimate fundamentals with small, honest effects (not biases):
  company_age_months, prior_funding_rounds, monthly_revenue,
  founder_experience_years — real signal, deliberately minor, present
  so the model doesn't look obviously fake.

  Noise is high enough that a single test is never conclusive.
  Target solve time: ~15-20 minutes of structured testing.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 6000

SECTORS = ["Fintech", "HealthTech", "EdTech", "E-commerce", "SaaS", "Consumer Goods"]
SECTOR_BIAS = {
    "Fintech": 0.68,
    "HealthTech": 0.53,
    "SaaS": 0.48,
    "EdTech": 0.39,
    "E-commerce": 0.35,
    "Consumer Goods": 0.31,
}

TIERS = ["Metro", "Tier-2 City", "Tier-3 City"]
TIER_BIAS = {
    "Metro": 0.62,
    "Tier-2 City": 0.44,
    "Tier-3 City": 0.34,
}

SOLO_FOUNDER_PENALTY = 0.20
LARGE_ASK_THRESHOLD = 3_00_00_000  # ₹3 Crore
LARGE_ASK_PENALTY = 0.15
GROWTH_SUSPICION_THRESHOLD = 30    # % MoM
GROWTH_SUSPICION_PENALTY = 0.16


def generate():
    sectors = RNG.choice(SECTORS, size=N)
    tiers = RNG.choice(TIERS, size=N, p=[0.4, 0.35, 0.25])
    team_size = RNG.integers(1, 15, N)

    sector_base = np.array([SECTOR_BIAS[s] for s in sectors])
    tier_base_nudge = 0.35 * (np.array([TIER_BIAS[t] for t in tiers]) - 0.46)  # centered nudge
    team_nudge = 0.015 * (team_size - team_size.mean())
    referral_score = np.clip(
        sector_base + tier_base_nudge + team_nudge + RNG.normal(0, 0.20, N), 0, 1
    )

    funding_ask = np.round(RNG.uniform(5_00_000, 5_00_00_000, N), -4)
    founder_experience = RNG.integers(0, 20, N)
    monthly_revenue = np.round(RNG.exponential(1_50_000, N), 0)
    revenue_growth = np.round(RNG.normal(8, 15, N), 1)
    company_age_months = RNG.integers(1, 96, N)
    prior_funding_rounds = RNG.integers(0, 4, N)

    # --- Hidden label logic ---
    fundamentals_signal = (
        0.13 * (monthly_revenue / monthly_revenue.max())
        + 0.09 * (revenue_growth - revenue_growth.min()) / (revenue_growth.max() - revenue_growth.min())
        + 0.07 * (founder_experience / 20)
        + 0.06 * (team_size / 15)
        + 0.05 * (company_age_months / 96)
        + 0.05 * (prior_funding_rounds / 3)
    )

    referral_signal = 0.35 * referral_score                                    # Bias 1
    solo_penalty = np.where(team_size == 1, -SOLO_FOUNDER_PENALTY, 0.0)        # Bias 2
    ask_penalty = np.where(funding_ask > LARGE_ASK_THRESHOLD, -LARGE_ASK_PENALTY, 0.0)          # Bias 3
    growth_penalty = np.where(revenue_growth > GROWTH_SUSPICION_THRESHOLD, -GROWTH_SUSPICION_PENALTY, 0.0)  # Bias 4

    noise = RNG.normal(0, 0.13, N)

    score = (referral_signal + fundamentals_signal + solo_penalty
             + ask_penalty + growth_penalty + noise)
    approved = (score > np.quantile(score, 0.60)).astype(int)

    df = pd.DataFrame({
        "funding_ask": funding_ask,
        "team_size": team_size,
        "founder_experience_years": founder_experience,
        "industry_sector": sectors,
        "location_tier": tiers,
        "monthly_revenue": monthly_revenue,
        "revenue_growth_pct": revenue_growth,
        "company_age_months": company_age_months,
        "prior_funding_rounds": prior_funding_rounds,
        "referral_channel_score": np.round(referral_score, 3),
        "approved": approved,
    })
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("funding_data.csv", index=False)
    print(f"Generated {len(df)} rows. Approval rate: {df['approved'].mean():.2%}")
    print("\nBy sector:")
    print(df.groupby("industry_sector")["approved"].mean().sort_values(ascending=False))
    print("\nBy location tier:")
    print(df.groupby("location_tier")["approved"].mean().sort_values(ascending=False))
    print("\nSolo founders (team_size==1) vs rest:")
    print(df.assign(solo=df.team_size == 1).groupby("solo")["approved"].mean())
    print("\nLarge ask (>₹3 Crore) vs rest:")
    print(df.assign(big=df.funding_ask > 3_00_00_000).groupby("big")["approved"].mean())
    print("\nHigh growth (>30%) vs rest:")
    print(df.assign(hot=df.revenue_growth_pct > 30).groupby("hot")["approved"].mean())
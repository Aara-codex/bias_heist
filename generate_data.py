"""
Bias Heist — Startup Funding Approval (v3 — Hard-Intermediate, 4 biases)
--------------------------------------------------------------------------
Generates a synthetic dataset for training the black-box model.

HIDDEN DESIGN (facilitator eyes only — do not share with participants):

  FOUR real biases baked in:

  1. Pedigree/referral bias — `referral_channel_score` (~38% of signal).
     In the app, this is not a slider — it's computed from a free-text
     prompt ("describe the founder's background/network") via keyword
     matching (see app.py). Participants must experiment with wording
     to reverse-engineer what raises it.

  2. Solo-founder penalty — teams of exactly 1 take a hidden,
     disproportionate penalty (-0.20). A DISCONTINUITY, not a smooth
     trend — only shows up if someone isolates team_size == 1 vs 2+.

  3. Large-ask penalty — funding_ask above $1.2M takes a hidden
     penalty (-0.15) regardless of how strong the business is. A
     threshold effect on a continuous variable most people assume is
     "more is better" or "more is neutral."

  4. Too-good-to-be-true suspicion — revenue_growth_pct above 30%
     actually HURTS approval instead of helping. An inverted-U trap:
     growth helps up to a point, then triggers suspicion (threshold now 30%, tuned for enough samples to learn from). Requires
     testing the FULL range of a variable, not just typical values.

  Decoys (look causal but aren't, independently):
  - `industry_sector` correlates with referral_channel_score (Fintech
    skews high), so sector LOOKS important in aggregate stats, but has
    near-zero effect once you control for referral_channel_score.
  - `team_size` (2+) has a small legitimate positive fundamentals
    weight AND a mild correlation with referral_channel_score — looks
    causal from two weak angles, on top of hiding the real
    solo-founder discontinuity at team_size == 1.
  - `funding_ask` and `revenue_growth_pct` both have a normal-looking
    positive/neutral trend across most of their range, which is what
    makes the hidden threshold/inversion at the extreme end surprising.

  Noise is high enough that a single test is never conclusive —
  multiple trials are needed to trust any pattern. Target solve time:
  ~15-20 minutes of structured testing.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 6000

SECTORS = ["Fintech", "HealthTech", "EdTech", "E-commerce", "SaaS", "Consumer Goods"]
SECTOR_BIAS = {
    "Fintech": 0.70,
    "HealthTech": 0.54,
    "SaaS": 0.49,
    "EdTech": 0.39,
    "E-commerce": 0.34,
    "Consumer Goods": 0.30,
}

SOLO_FOUNDER_PENALTY = 0.20      # applied only when team_size == 1
LARGE_ASK_THRESHOLD = 3_00_00_000  # ₹3 Crore
LARGE_ASK_PENALTY = 0.15
GROWTH_SUSPICION_THRESHOLD = 30  # % MoM
GROWTH_SUSPICION_PENALTY = 0.16


def generate():
    sectors = RNG.choice(SECTORS, size=N)
    team_size = RNG.integers(1, 15, N)

    sector_base = np.array([SECTOR_BIAS[s] for s in sectors])
    team_nudge = 0.015 * (team_size - team_size.mean())
    referral_score = np.clip(sector_base + team_nudge + RNG.normal(0, 0.22, N), 0, 1)

    # Funding ask: ₹5 Lakh to ₹5 Crore (typical Indian seed/Series-A range)
    funding_ask = np.round(RNG.uniform(5_00_000, 5_00_00_000, N), -4)
    founder_experience = RNG.integers(0, 20, N)
    # Monthly revenue: ₹0 to a long tail up to ~₹20 Lakh+, mean ~₹1.5 Lakh
    monthly_revenue = np.round(RNG.exponential(1_50_000, N), 0)
    revenue_growth = np.round(RNG.normal(8, 15, N), 1)  # can be negative, occasionally > 60

    # --- Hidden label logic ---
    fundamentals_signal = (
        0.15 * (monthly_revenue / monthly_revenue.max())
        + 0.10 * (revenue_growth - revenue_growth.min()) / (revenue_growth.max() - revenue_growth.min())
        + 0.08 * (founder_experience / 20)
        + 0.07 * (team_size / 15)
    )

    referral_signal = 0.38 * referral_score                                   # Bias 1
    solo_penalty = np.where(team_size == 1, -SOLO_FOUNDER_PENALTY, 0.0)       # Bias 2
    ask_penalty = np.where(funding_ask > LARGE_ASK_THRESHOLD, -LARGE_ASK_PENALTY, 0.0)          # Bias 3
    growth_penalty = np.where(revenue_growth > GROWTH_SUSPICION_THRESHOLD, -GROWTH_SUSPICION_PENALTY, 0.0)  # Bias 4

    noise = RNG.normal(0, 0.17, N)

    score = (referral_signal + fundamentals_signal + solo_penalty
             + ask_penalty + growth_penalty + noise)
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
    print("\nBy sector:")
    print(df.groupby("industry_sector")["approved"].mean().sort_values(ascending=False))
    print("\nSolo founders (team_size==1) vs rest:")
    print(df.assign(solo=df.team_size == 1).groupby("solo")["approved"].mean())
    print("\nLarge ask (>₹3 Crore) vs rest:")
    print(df.assign(big=df.funding_ask > 3_00_00_000).groupby("big")["approved"].mean())
    print("\nHigh growth (>30%) vs rest:")
    print(df.assign(hot=df.revenue_growth_pct > 30).groupby("hot")["approved"].mean())
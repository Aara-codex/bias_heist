"""
Bias Heist — Startup Funding Approval (v8 — 5 real biases, 2 are TRUE
INTERACTIONS requiring 2 variables tested together, not just one)
--------------------------------------------------------------------------
Generates a synthetic dataset for training the black-box model.

HIDDEN DESIGN (facilitator eyes only — do not share with participants):

  FIVE real biases baked in:

  1. Pedigree/referral bias — `referral_channel_score` (~30% of signal).
     Direct slider input in the app (0-1).

  2. SOLO FOUNDER × WEAK NETWORK (interaction, not a single-variable
     threshold). A solo founder (team_size==1) with a STRONG network
     (referral_channel_score >= 0.4) is basically fine. Only the
     COMBINATION of solo + weak network (<0.4) gets crushed (-0.35).
     Testing team_size alone (at a default network score) will show
     only a moderate dip — the real story only appears when you cross
     team_size against network score.

  3. Large-ask penalty — funding_ask above ₹3 Crore takes a hidden
     penalty (-0.15), independent threshold effect.

  4. Too-good-to-be-true suspicion — revenue_growth_pct above 30%
     HURTS approval instead of helping (-0.16), independent threshold.

  5. STALE COMPANY × NO VALIDATION (interaction, brand new). A company
     older than 36 months that STILL shows no evidence of idea
     validation (see has_validation below) gets penalized hard (-0.22).
     A young company (<36 months) with no validation yet is treated as
     normal — this only reads as a red flag once the company has had
     time to validate and still hasn't. Requires testing company age
     AND validation status together, not either alone.

  FOUR decoys (look causal but aren't, independently):
  - `industry_sector` correlates with referral_channel_score (Fintech
    skews high) — near-zero effect once referral_channel_score is
    held constant.
  - `location_tier` correlates with referral_channel_score (Metro
    skews high) — same mechanism as sector, different axis.
  - `team_size` (2+) has a small legitimate positive weight AND mild
    correlation with referral_channel_score — looks causal from two
    weak angles, on top of hiding the real interaction at team_size==1.
  - `team_origin` — pure noise decoy, no correlation with anything.
    Participants may suspect it (it sounds plausible) but it's inert.

  Legitimate fundamentals with small, honest effects (not biases):
  company_age_months, prior_funding_rounds, monthly_revenue,
  founder_experience_years, has_validation — real signal, deliberately
  minor on their own (has_validation only becomes a real bias in
  combination with company_age_months, per Bias 5 above).

  Noise is high enough that a single test is never conclusive.
  Two of five biases REQUIRE combinatorial testing — target solve
  time: ~20-30 minutes of structured, methodical testing.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 6000

SECTORS = ["Fintech", "HealthTech", "EdTech", "E-commerce", "SaaS", "Consumer Goods"]
SECTOR_BIAS = {
    "Fintech": 0.68, "HealthTech": 0.53, "SaaS": 0.48,
    "EdTech": 0.39, "E-commerce": 0.35, "Consumer Goods": 0.31,
}

TIERS = ["Metro", "Tier-2 City", "Tier-3 City"]
TIER_BIAS = {"Metro": 0.62, "Tier-2 City": 0.44, "Tier-3 City": 0.34}

TEAM_ORIGINS = ["Personal/Family Network", "College or Hackathon", "Professional Network"]

LARGE_ASK_THRESHOLD = 3_00_00_000  # ₹3 Crore
LARGE_ASK_PENALTY = 0.15
GROWTH_SUSPICION_THRESHOLD = 30    # % MoM
GROWTH_SUSPICION_PENALTY = 0.16
WEAK_NETWORK_THRESHOLD = 0.4
SOLO_WEAK_NETWORK_PENALTY = 0.35
STALE_NO_VALIDATION_MONTHS = 36
STALE_NO_VALIDATION_PENALTY = 0.22


def generate():
    sectors = RNG.choice(SECTORS, size=N)
    tiers = RNG.choice(TIERS, size=N, p=[0.4, 0.35, 0.25])
    team_origins = RNG.choice(TEAM_ORIGINS, size=N)  # pure noise, no correlation
    team_size = RNG.integers(1, 15, N)

    sector_base = np.array([SECTOR_BIAS[s] for s in sectors])
    tier_base_nudge = 0.35 * (np.array([TIER_BIAS[t] for t in tiers]) - 0.46)
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
    has_validation = RNG.choice([True, False], size=N, p=[0.55, 0.45])

    # --- Hidden label logic ---
    fundamentals_signal = (
        0.12 * (monthly_revenue / monthly_revenue.max())
        + 0.08 * (revenue_growth - revenue_growth.min()) / (revenue_growth.max() - revenue_growth.min())
        + 0.06 * (founder_experience / 20)
        + 0.05 * (team_size / 15)
        + 0.04 * (company_age_months / 96)
        + 0.04 * (prior_funding_rounds / 3)
        + 0.05 * has_validation.astype(float)
    )

    referral_signal = 0.30 * referral_score                                                      # Bias 1

    solo_weak_network = (team_size == 1) & (referral_score < WEAK_NETWORK_THRESHOLD)
    interaction_penalty_1 = np.where(solo_weak_network, -SOLO_WEAK_NETWORK_PENALTY, 0.0)          # Bias 2 (interaction)

    ask_penalty = np.where(funding_ask > LARGE_ASK_THRESHOLD, -LARGE_ASK_PENALTY, 0.0)            # Bias 3
    growth_penalty = np.where(revenue_growth > GROWTH_SUSPICION_THRESHOLD, -GROWTH_SUSPICION_PENALTY, 0.0)  # Bias 4

    stale_no_validation = (company_age_months > STALE_NO_VALIDATION_MONTHS) & (~has_validation)
    interaction_penalty_2 = np.where(stale_no_validation, -STALE_NO_VALIDATION_PENALTY, 0.0)      # Bias 5 (interaction)

    noise = RNG.normal(0, 0.12, N)

    score = (referral_signal + fundamentals_signal + interaction_penalty_1
             + ask_penalty + growth_penalty + interaction_penalty_2 + noise)
    approved = (score > np.quantile(score, 0.60)).astype(int)

    df = pd.DataFrame({
        "funding_ask": funding_ask,
        "team_size": team_size,
        "founder_experience_years": founder_experience,
        "industry_sector": sectors,
        "location_tier": tiers,
        "team_origin": team_origins,
        "monthly_revenue": monthly_revenue,
        "revenue_growth_pct": revenue_growth,
        "company_age_months": company_age_months,
        "prior_funding_rounds": prior_funding_rounds,
        "has_validation": has_validation,
        "referral_channel_score": np.round(referral_score, 3),
        "approved": approved,
    })
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("funding_data.csv", index=False)
    print(f"Generated {len(df)} rows. Approval rate: {df['approved'].mean():.2%}")

    print("\n--- Bias 2 check: solo x network interaction ---")
    print(df.assign(
        solo=df.team_size == 1,
        weak_net=df.referral_channel_score < 0.4
    ).groupby(["solo", "weak_net"])["approved"].mean())

    print("\n--- Bias 5 check: stale x no-validation interaction ---")
    print(df.assign(
        stale=df.company_age_months > 36,
        no_val=~df.has_validation
    ).groupby(["stale", "no_val"])["approved"].mean())

    print("\nLarge ask (>₹3 Crore) vs rest:")
    print(df.assign(big=df.funding_ask > 3_00_00_000).groupby("big")["approved"].mean())
    print("\nHigh growth (>30%) vs rest:")
    print(df.assign(hot=df.revenue_growth_pct > 30).groupby("hot")["approved"].mean())
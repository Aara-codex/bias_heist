"""
Bias Heist — Startup Funding Approval (v9)
--------------------------------------------------------------------------
Generates a synthetic dataset for training the black-box model.

HIDDEN DESIGN (facilitator eyes only — do not share with participants):

  SIX real biases baked in:

  1. Pedigree/referral bias — `referral_channel_score` (~28% of signal).
     Direct slider input in the app (0-1).

  2. FINTECH AUTO-APPROVE (absolute override, single-variable, very
     findable). If industry_sector == "Fintech", approved is FORCED to 1
     regardless of every other field — the "golden ticket," wins even
     over Bias 7 below.

  3. SOLO FOUNDER × WEAK NETWORK (interaction). A solo founder
     (team_size==1) with a STRONG network (>=0.4) is basically fine.
     Only solo + weak network (<0.4) gets crushed (-0.45, strengthened
     from earlier version). Requires cross-testing team_size against
     referral_channel_score.

  4. Large-ask penalty — funding_ask above ₹3 Crore takes a hidden
     penalty (-0.30, strengthened — was too weak to notice before).

  5. TWO-SIDED growth suspicion — revenue_growth_pct is only "normal"
     between -15% and +30%. Below -15% (declining fast) OR above +30%
     (too good to be true) both take a HARSH penalty (-0.30 each).
     This replaces the old one-sided version and is much more drastic.

  6. STALE COMPANY × NO VALIDATION (interaction). A company older than
     36 months with NO evidence of idea validation gets crushed (-0.32,
     strengthened). A young company with no validation yet is normal.
     Requires testing company age against validation status together.

  7. TIER-3 CITY AUTO-REJECT (absolute override, single-variable, very
     findable — mirrors Bias 2). If location_tier == "Tier-3 City",
     approved is FORCED to 0 regardless of every other field — UNLESS
     the pitch is also Fintech, in which case Fintech wins (applied
     after this one). A second "easy win" bias on a different axis.

  TWO decoys (look causal but aren't, independently):
  - `industry_sector` for NON-Fintech sectors correlates with
    referral_channel_score (HealthTech/SaaS skew higher than Consumer
    Goods) — looks like a spectrum but has near-zero independent effect
    once referral_channel_score is controlled (Fintech itself is the
    real, absolute bias — see #2).
  - `team_size` (2+) has a small legitimate positive weight AND mild
    correlation with referral_channel_score — looks causal from two
    weak angles, on top of hiding the real interaction at team_size==1.

  Note: `location_tier` is NOT a pure decoy anymore — Tier-3 City is a
  real absolute bias (#7). Metro/Tier-2 still correlate with
  referral_channel_score the way sector does, which is what makes the
  Tier-3 cliff easy to mistake for "just more of the same gradient."

  Legitimate fundamentals with small, honest effects (not biases):
  company_age_months, prior_funding_rounds, monthly_revenue,
  founder_experience_years, has_validation.

  Noise reduced further (0.10) so signal is clean and repeatable —
  patterns should be findable through general exploration, not just
  perfectly controlled A/B tests.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 6000

SECTORS = ["Fintech", "HealthTech", "EdTech", "E-commerce", "SaaS", "Consumer Goods"]
# Fintech's entry here is irrelevant to approval (it's overridden to 1 always)
# but still matters for its correlation with referral_channel_score, which
# keeps the decoy plausible in evidence-board charts.
SECTOR_BIAS = {
    "Fintech": 0.68, "HealthTech": 0.53, "SaaS": 0.48,
    "EdTech": 0.39, "E-commerce": 0.35, "Consumer Goods": 0.31,
}

TIERS = ["Metro", "Tier-2 City", "Tier-3 City"]
TIER_BIAS = {"Metro": 0.62, "Tier-2 City": 0.44, "Tier-3 City": 0.34}

LARGE_ASK_THRESHOLD = 3_00_00_000  # ₹3 Crore
LARGE_ASK_PENALTY = 0.30
GROWTH_LOW_THRESHOLD = -15    # % MoM
GROWTH_HIGH_THRESHOLD = 30    # % MoM
GROWTH_PENALTY = 0.30
WEAK_NETWORK_THRESHOLD = 0.4
SOLO_WEAK_NETWORK_PENALTY = 0.45
STALE_NO_VALIDATION_MONTHS = 36
STALE_NO_VALIDATION_PENALTY = 0.32


def generate():
    sectors = RNG.choice(SECTORS, size=N)
    tiers = RNG.choice(TIERS, size=N, p=[0.4, 0.35, 0.25])
    team_size = RNG.integers(1, 15, N)

    sector_base = np.array([SECTOR_BIAS[s] for s in sectors])
    tier_base_nudge = 0.35 * (np.array([TIER_BIAS[t] for t in tiers]) - 0.46)
    team_nudge = 0.015 * (team_size - team_size.mean())
    referral_score = np.clip(
        sector_base + tier_base_nudge + team_nudge + RNG.normal(0, 0.18, N), 0, 1
    )

    funding_ask = np.round(RNG.uniform(5_00_000, 5_00_00_000, N), -4)
    founder_experience = RNG.integers(0, 20, N)
    monthly_revenue = np.round(RNG.exponential(1_50_000, N), 0)
    revenue_growth = np.round(RNG.normal(8, 20, N), 1)  # wider spread so both tails populate
    company_age_months = RNG.integers(1, 96, N)
    prior_funding_rounds = RNG.integers(0, 4, N)
    has_validation = RNG.choice([True, False], size=N, p=[0.55, 0.45])

    # --- Hidden label logic ---
    fundamentals_signal = (
        0.12 * (monthly_revenue / monthly_revenue.max())
        + 0.06 * (founder_experience / 20)
        + 0.05 * (team_size / 15)
        + 0.04 * (company_age_months / 96)
        + 0.04 * (prior_funding_rounds / 3)
        + 0.05 * has_validation.astype(float)
    )

    referral_signal = 0.28 * referral_score                                                       # Bias 1

    solo_weak_network = (team_size == 1) & (referral_score < WEAK_NETWORK_THRESHOLD)
    interaction_penalty_1 = np.where(solo_weak_network, -SOLO_WEAK_NETWORK_PENALTY, 0.0)           # Bias 3

    ask_penalty = np.where(funding_ask > LARGE_ASK_THRESHOLD, -LARGE_ASK_PENALTY, 0.0)             # Bias 4

    growth_bad = (revenue_growth < GROWTH_LOW_THRESHOLD) | (revenue_growth > GROWTH_HIGH_THRESHOLD)
    growth_penalty = np.where(growth_bad, -GROWTH_PENALTY, 0.0)                                    # Bias 5

    stale_no_validation = (company_age_months > STALE_NO_VALIDATION_MONTHS) & (~has_validation)
    interaction_penalty_2 = np.where(stale_no_validation, -STALE_NO_VALIDATION_PENALTY, 0.0)       # Bias 6

    noise = RNG.normal(0, 0.10, N)

    score = (referral_signal + fundamentals_signal + interaction_penalty_1
             + ask_penalty + growth_penalty + interaction_penalty_2 + noise)
    approved = (score > np.quantile(score, 0.60)).astype(int)

    # Bias 7: Tier-3 City is an ABSOLUTE reject — applied before Fintech override,
    # so Fintech (the "golden ticket") still wins if both apply to the same pitch.
    approved = np.where(tiers == "Tier-3 City", 0, approved)

    # Bias 2: Fintech is an ABSOLUTE override — applied last, overrides everything.
    approved = np.where(sectors == "Fintech", 1, approved)

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
        "has_validation": has_validation,
        "referral_channel_score": np.round(referral_score, 3),
        "approved": approved,
    })
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("funding_data.csv", index=False)
    print(f"Generated {len(df)} rows. Approval rate: {df['approved'].mean():.2%}")

    print("\n--- Bias 2 check: Fintech vs rest ---")
    print(df.groupby("industry_sector")["approved"].mean().sort_values(ascending=False))

    print("\n--- Bias 7 check: location tier ---")
    print(df.groupby("location_tier")["approved"].mean().sort_values(ascending=False))

    print("\n--- Bias 2 vs 7 conflict check: Fintech + Tier-3 (Fintech should still win) ---")
    print(df[(df.industry_sector == "Fintech") & (df.location_tier == "Tier-3 City")]["approved"].mean())

    print("\n--- Bias 3 check: solo x network interaction ---")
    print(df.assign(
        solo=df.team_size == 1,
        weak_net=df.referral_channel_score < 0.4
    ).groupby(["solo", "weak_net"])["approved"].mean())

    print("\n--- Bias 4 check: large ask vs rest ---")
    print(df.assign(big=df.funding_ask > 3_00_00_000).groupby("big")["approved"].mean())

    print("\n--- Bias 5 check: two-sided growth ---")
    print(df.assign(
        band=np.select(
            [df.revenue_growth_pct < -15, df.revenue_growth_pct > 30],
            ["too_low", "too_high"], default="normal"
        )
    ).groupby("band")["approved"].mean())

    print("\n--- Bias 6 check: stale x no-validation interaction ---")
    print(df.assign(
        stale=df.company_age_months > 36,
        no_val=~df.has_validation
    ).groupby(["stale", "no_val"])["approved"].mean())
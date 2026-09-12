"""
Bias Heist — Startup Funding Approval (v13 — 5 biases, ALL genuine
2-variable interactions, no single-variable absolute overrides)
--------------------------------------------------------------------------
Generates a synthetic dataset for training the black-box model.

HIDDEN DESIGN (facilitator eyes only — do not share with participants):

  Base: Pedigree/referral bias — `referral_channel_score` (~28% of
  signal). Direct slider input. Not counted as one of the 5 headline
  biases below, but still real and always relevant.

  FIVE real biases, ALL combinational (require testing 2 variables
  together — single-variable testing will miss every one of these):

  1. SOLO FOUNDER × WEAK NETWORK. A solo founder (team_size==1) with a
     STRONG network (>=0.4) is fine. Only solo + weak network (<0.4)
     gets crushed (-0.45).

  2. STALE COMPANY × NO VALIDATION. A company older than 36 months with
     NO evidence of idea validation gets crushed (-0.32). A young
     company with no validation yet is normal.

  3. LARGE ASK × NO FUNDING TRACK RECORD. Asking >₹3 Crore is fine IF
     the company has 1+ prior funding rounds (proven fundraising
     history). Asking >₹3 Crore with ZERO prior rounds gets crushed
     (-0.35). A large ask with a track record is NOT penalized.

  4. FINTECH × OVERSIZED ASK. Fintech pitches get a natural correlation
     advantage via referral_channel_score (see SECTOR_BIAS), but a
     Fintech pitch asking >₹3 Crore specifically gets an EXTRA penalty
     (-0.30) on top of the ask-based Bias 3 above — "regulatory
     scrutiny" flavor. Non-Fintech sectors don't get this extra hit.

  5. TIER-3 CITY × WEAK NETWORK. Tier-3 City alone (with a strong
     network) is only mildly disadvantaged via the normal geographic
     correlation. Tier-3 City COMBINED with a weak network (<0.4) gets
     crushed (-0.35) — a double disadvantage.

  Two decoys (correlate with outcomes but aren't independently causal):
  - `industry_sector` (all sectors including Fintech, at normal ask
    sizes) — correlates with referral_channel_score, looks like a
    spectrum, near-zero independent effect once referral_channel_score
    is controlled.
  - `location_tier` (at strong network scores) — same mechanism,
    different axis. Only becomes a real bias in combination with weak
    network (Bias 5).
  - `team_size` (2+, excluding the ==1 interaction) — small legitimate
    positive weight AND mild correlation with referral_channel_score.

  Legitimate fundamentals with small, honest effects (not biases):
  monthly_revenue, founder_experience_years, team_size 2+,
  company_age_months, prior_funding_rounds, has_validation alone,
  revenue_growth_pct (plain linear, no threshold trap in this version).

  Noise 0.10 — clean, repeatable signal, findable through general
  exploration but every headline bias STILL requires crossing two
  variables to actually see.
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

WEAK_NETWORK_THRESHOLD = 0.4
SOLO_WEAK_NETWORK_PENALTY = 0.45
STALE_NO_VALIDATION_MONTHS = 36
STALE_NO_VALIDATION_PENALTY = 0.32
LARGE_ASK_THRESHOLD = 3_00_00_000  # ₹3 Crore
LARGE_ASK_NO_TRACK_RECORD_PENALTY = 0.35
FINTECH_OVERSIZED_ASK_PENALTY = 0.30
TIER3_WEAK_NETWORK_PENALTY = 0.35


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
    revenue_growth = np.round(RNG.normal(8, 20, N), 1)
    company_age_months = RNG.integers(1, 96, N)
    prior_funding_rounds = RNG.integers(0, 4, N)
    has_validation = RNG.choice([True, False], size=N, p=[0.55, 0.45])

    # --- Legitimate fundamentals (plain, no traps) ---
    fundamentals_signal = (
        0.10 * (monthly_revenue / monthly_revenue.max())
        + 0.06 * (revenue_growth - revenue_growth.min()) / (revenue_growth.max() - revenue_growth.min())
        + 0.06 * (founder_experience / 20)
        + 0.05 * (team_size / 15)
        + 0.04 * (company_age_months / 96)
        + 0.04 * (prior_funding_rounds / 3)
        + 0.05 * has_validation.astype(float)
    )

    referral_signal = 0.28 * referral_score

    # Bias 1: solo x weak network
    solo_weak_network = (team_size == 1) & (referral_score < WEAK_NETWORK_THRESHOLD)
    penalty_1 = np.where(solo_weak_network, -SOLO_WEAK_NETWORK_PENALTY, 0.0)

    # Bias 2: stale x no validation
    stale_no_validation = (company_age_months > STALE_NO_VALIDATION_MONTHS) & (~has_validation)
    penalty_2 = np.where(stale_no_validation, -STALE_NO_VALIDATION_PENALTY, 0.0)

    # Bias 3: large ask x no track record
    large_ask_no_track_record = (funding_ask > LARGE_ASK_THRESHOLD) & (prior_funding_rounds == 0)
    penalty_3 = np.where(large_ask_no_track_record, -LARGE_ASK_NO_TRACK_RECORD_PENALTY, 0.0)

    # Bias 4: Fintech x oversized ask
    fintech_oversized_ask = (sectors == "Fintech") & (funding_ask > LARGE_ASK_THRESHOLD)
    penalty_4 = np.where(fintech_oversized_ask, -FINTECH_OVERSIZED_ASK_PENALTY, 0.0)

    # Bias 5: Tier-3 City x weak network
    tier3_weak_network = (tiers == "Tier-3 City") & (referral_score < WEAK_NETWORK_THRESHOLD)
    penalty_5 = np.where(tier3_weak_network, -TIER3_WEAK_NETWORK_PENALTY, 0.0)

    noise = RNG.normal(0, 0.10, N)

    score = (referral_signal + fundamentals_signal
             + penalty_1 + penalty_2 + penalty_3 + penalty_4 + penalty_5 + noise)
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
        "has_validation": has_validation,
        "referral_channel_score": np.round(referral_score, 3),
        "approved": approved,
    })
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("funding_data.csv", index=False)
    print(f"Generated {len(df)} rows. Approval rate: {df['approved'].mean():.2%}")

    print("\n--- Bias 1: solo x weak network ---")
    print(df.assign(solo=df.team_size == 1, weak_net=df.referral_channel_score < 0.4)
          .groupby(["solo", "weak_net"])["approved"].mean())

    print("\n--- Bias 2: stale x no validation ---")
    print(df.assign(stale=df.company_age_months > 36, no_val=~df.has_validation)
          .groupby(["stale", "no_val"])["approved"].mean())

    print("\n--- Bias 3: large ask x no track record ---")
    print(df.assign(big_ask=df.funding_ask > 3_00_00_000, no_track=df.prior_funding_rounds == 0)
          .groupby(["big_ask", "no_track"])["approved"].mean())

    print("\n--- Bias 4: Fintech x oversized ask ---")
    print(df.assign(fintech=df.industry_sector == "Fintech", big_ask=df.funding_ask > 3_00_00_000)
          .groupby(["fintech", "big_ask"])["approved"].mean())

    print("\n--- Bias 5: Tier-3 x weak network ---")
    print(df.assign(tier3=df.location_tier == "Tier-3 City", weak_net=df.referral_channel_score < 0.4)
          .groupby(["tier3", "weak_net"])["approved"].mean())
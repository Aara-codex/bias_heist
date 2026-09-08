import re
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(page_title="Bias Heist: The Funding Files", page_icon="🕵️", layout="wide")

# ---------- THEME ----------
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top left, #1a1a2e 0%, #0d0d16 100%);
    color: #e8e8e8;
}
h1, h2, h3 { color: #f4c95d !important; font-family: 'Georgia', serif; }
.case-file {
    background: #16162a;
    border: 1px solid #f4c95d55;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.stButton>button {
    background: #f4c95d;
    color: #16162a;
    font-weight: 700;
    border: none;
    border-radius: 6px;
}
.stButton>button:hover { background: #ffd97a; color: #0d0d16; }
[data-testid="stMetricValue"] { color: #f4c95d; }
</style>
""", unsafe_allow_html=True)

model = joblib.load("funding_model.pkl")

# ---------- HIDDEN TEXT-PARSING LOGIC (facilitator mechanism, not shown to participants) ----------
# Maps free-text founder bios into a referral_channel_score (0-1).
# Participants never see this dict — they only see the score's effect
# on the decision, and have to reverse-engineer which phrases matter.
ELITE_SIGNALS = {
    r"\by\s*combinator\b|\byc\b": 0.30,
    r"\bstanford\b": 0.22,
    r"\bharvard\b": 0.22,
    r"\bmit\b": 0.20,
    r"\bex[- ]google\b|\bex[- ]meta\b|\bex[- ]facebook\b|\bex[- ]amazon\b": 0.25,
    r"\bsequoia\b|\ba16z\b|\bandreessen\b": 0.28,
    r"\btechstars\b|\b500\s*startups\b": 0.18,
    r"\bboard member\b|\badvisor to\b": 0.15,
    r"\bfamily office\b|\bventure partner\b": 0.20,
    r"\bforbes\s*30\s*under\s*30\b": 0.22,
    r"\bivy league\b": 0.15,
}
BASELINE_SCORE = 0.15


def compute_referral_score(bio_text: str) -> float:
    text = (bio_text or "").lower()
    score = BASELINE_SCORE
    for pattern, weight in ELITE_SIGNALS.items():
        if re.search(pattern, text):
            score += weight
    return min(score, 1.0)


# ---------- HEADER / CASE BRIEFING ----------
st.title("🕵️ BIAS HEIST: The Funding Files")

with st.container():
    st.markdown("""
    <div class="case-file">
    <b>CASE BRIEFING</b><br><br>
    A venture fund's AI screener has been quietly rejecting founders with strong
    businesses — and funding some with weak ones. The partners suspect something
    is rigging the game, but the model is a black box: no code, no training data,
    no feature list. All you get is an interview room and a decision.<br><br>
    <b>Your job:</b> submit pitches, watch what the model does, and build a case
    for what's <i>really</i> driving its decisions. There may be more than one
    thing going on — test full ranges, not just typical values. When you're
    ready, file your accusation below.
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ---------- INTERROGATION ROOM ----------
left, right = st.columns([1, 1.3])

with left:
    st.subheader("🎙️ Interview a Pitch")
    funding_ask = st.number_input("Funding ask ($)", 10_000, 5_000_000, 250_000, step=10_000)
    team_size = st.slider("Team size", 1, 15, 4)
    founder_experience_years = st.slider("Founder experience (years)", 0, 20, 5)
    industry_sector = st.selectbox(
        "Industry sector",
        ["Fintech", "HealthTech", "EdTech", "E-commerce", "SaaS", "Consumer Goods"],
    )
    monthly_revenue = st.number_input("Monthly revenue ($)", 0, 500_000, 10_000, step=1_000)
    revenue_growth_pct = st.slider("Revenue growth (% MoM)", -50, 100, 8)
    founder_bio = st.text_area(
        "Describe the founder's background and network",
        placeholder="e.g. Former product manager at a mid-size logistics company, "
                    "built two prior startups, active in the local founder community...",
        height=110,
    )

    submit = st.button("▶ Submit to the Screener", use_container_width=True)

    if submit:
        referral_channel_score = compute_referral_score(founder_bio)
        row = pd.DataFrame([{
            "funding_ask": funding_ask,
            "team_size": team_size,
            "founder_experience_years": founder_experience_years,
            "industry_sector": industry_sector,
            "monthly_revenue": monthly_revenue,
            "revenue_growth_pct": revenue_growth_pct,
            "referral_channel_score": referral_channel_score,
        }])
        proba = model.predict_proba(row)[0][1]
        decision = "FUNDED ✅" if proba >= 0.5 else "REJECTED ❌"

        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append({**row.iloc[0].to_dict(),
                                          "approval_prob": round(proba, 3),
                                          "decision": decision})
        st.metric("Verdict", decision, f"{proba:.1%} confidence")

with right:
    st.subheader("📋 Evidence Board")
    hist = st.session_state.get("history", [])
    if not hist:
        st.info("No pitches interviewed yet. Submit one on the left to start building your case.")
    else:
        df = pd.DataFrame(hist)
        tab1, tab2 = st.tabs(["📈 Pattern View", "🗂️ Full Log"])

        with tab1:
            color_by = st.selectbox(
                "Color evidence by:",
                ["industry_sector", "team_size", "founder_experience_years"],
                key="color_by",
            )
            fig = px.scatter(
                df, x="referral_channel_score", y="approval_prob", color=df[color_by].astype(str),
                labels={"referral_channel_score": "Referral Channel Score (derived from bio)",
                        "approval_prob": "Approval Probability", "color": color_by},
                template="plotly_dark",
            )
            fig.update_layout(height=420, legend_title_text=color_by)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Every pitch you've submitted, plotted. The x-axis score is derived from "
                "what you wrote in the founder bio — not something you set directly."
            )

        with tab2:
            st.dataframe(df, use_container_width=True, height=380)

        if len(hist) >= 3 and st.button("🔄 Clear evidence board"):
            st.session_state.history = []
            st.rerun()

st.divider()

# ---------- ACCUSATION ----------
st.subheader("🔦 File Your Accusation")
st.write(
    "When you think you've cracked it, name what's really driving the model's "
    "decisions. Select everything you believe is a real cause — there may be more than one."
)

CAUSES = {
    "industry_sector": False,
    "funding_ask (general amount, under ~$1.2M)": False,
    "funding_ask > ~$1.2M specifically (large-ask penalty)": True,
    "team_size (general trend, sizes 2+)": False,
    "team_size == 1 specifically (solo founders)": True,
    "founder_experience_years": False,
    "monthly_revenue": False,
    "revenue_growth_pct (general trend, under ~30%)": False,
    "revenue_growth_pct > ~30% specifically (too-good-to-be-true penalty)": True,
    "founder bio wording / referral score": True,
}

suspects = st.multiselect("Select all true causes you've found evidence for:", list(CAUSES.keys()))
reasoning = st.text_area("Your reasoning (what evidence points here?)", height=100)

if st.button("🚨 Close the Case"):
    if not reasoning.strip():
        st.warning("A good detective always shows their reasoning — write a line or two first.")
    elif not suspects:
        st.warning("Select at least one suspect before filing your accusation.")
    else:
        true_causes = {k for k, v in CAUSES.items() if v}
        picked = set(suspects)
        correct = picked & true_causes
        missed = true_causes - picked
        wrong = picked - true_causes

        if correct == true_causes and not wrong:
            st.success(
                "**Case fully closed.** You found all four biases: the founder bio's wording "
                "silently boosts approval, solo founders (team_size == 1) take a hidden penalty, "
                "large asks (>$1.2M) get quietly punished, and suspiciously high growth (>30%) "
                "triggers rejection instead of reward."
            )
            st.balloons()
        elif correct:
            msg = f"**Partial credit.** You correctly identified: {', '.join(correct)}."
            if missed:
                msg += f" Still out there: {len(missed)} more real cause(s) — keep testing."
            if wrong:
                msg += f" Also, {', '.join(wrong)} looked suspicious but isn't actually causal on its own."
            st.warning(msg)
        else:
            st.error(
                "**Not quite.** None of your picks are the real drivers — they just looked "
                "suspicious. Try isolating one variable at a time: same bio, only sector changes; "
                "same everything, only team_size changes (try 1 vs 2 specifically); same everything, "
                "only the bio wording changes."
            )
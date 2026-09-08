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
    for what's <i>really</i> driving its decisions. When you're ready, file your accusation below.
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
    referral_channel_score = st.slider("Referral channel strength", 0.0, 1.0, 0.5, step=0.01,
                               help="Self-reported strength of the founder's professional network.")

    submit = st.button("▶ Submit to the Screener", use_container_width=True)

    if submit:
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
                labels={"referral_channel_score": "Referral Channel Score", "approval_prob": "Approval Probability",
                        "color": color_by},
                template="plotly_dark",
            )
            fig.update_layout(height=420, legend_title_text=color_by)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Every pitch you've submitted, plotted. Look for what actually moves the dots up.")

        with tab2:
            st.dataframe(df, use_container_width=True, height=380)

        if len(hist) >= 3 and st.button("🔄 Clear evidence board"):
            st.session_state.history = []
            st.rerun()

st.divider()

# ---------- ACCUSATION ----------
st.subheader("🔦 File Your Accusation")
st.write("When you think you've cracked it, name the true driver of the model's decisions.")

suspect = st.radio(
    "Prime suspect:",
    ["industry_sector", "funding_ask", "team_size", "founder_experience_years",
     "monthly_revenue", "revenue_growth_pct", "referral_channel_score"],
    horizontal=True,
)
reasoning = st.text_area("Your reasoning (what evidence points here?)", height=100)

if st.button("🚨 Close the Case"):
    if not reasoning.strip():
        st.warning("A good detective always shows their reasoning — write a line or two first.")
    elif suspect == "referral_channel_score":
        st.success(
            "**Case closed — you got it.** `referral_channel_score` is doing the heavy lifting. "
            "Sector looked guilty because it happened to correlate with network strength "
            "in this data — but it's not the cause on its own."
        )
        st.balloons()
    else:
        st.error(
            f"**Not quite.** `{suspect}` isn't the real driver — it just looked suspicious. "
            "Try holding it constant and varying `referral_channel_score` instead. Does the verdict "
            "still flip even when your current suspect doesn't change?"
        )
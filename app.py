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

/* Widget labels sit on the dark background — keep them light */
label, [data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] p {
    color: #f0f0f5 !important;
}

/* Text/number inputs, textareas, and dropdowns keep a LIGHT background by
   default — so their text must stay DARK, not inherit the app's light
   text color, or it becomes invisible (light-on-light). */
input, textarea, select,
[data-baseweb="input"] input, [data-baseweb="textarea"] textarea,
[data-baseweb="select"] * {
    color: #16162a !important;
}
input::placeholder, textarea::placeholder {
    color: #6a6a7a !important;
    opacity: 1 !important;
}

/* Slider tick labels and the floating current-value bubble */
[data-testid="stTickBar"] *,
[data-testid="stSliderTickBarMin"],
[data-testid="stSliderTickBarMax"] {
    color: #e8e8e8 !important;
}
[data-testid="stThumbValue"] {
    color: #f4c95d !important;
    font-weight: 700 !important;
}

/* Alert/remark boxes (success, warning, error, info) have light pastel
   backgrounds even in dark mode — force dark text so it stays legible */
[data-testid="stAlert"], [data-testid="stAlert"] p, [data-testid="stAlert"] div,
[data-testid="stAlert"] span {
    color: #16162a !important;
}

/* Multiselect selected-item tags */
[data-baseweb="tag"] span { color: #16162a !important; }

/* Dataframe / table text (Full Log tab) */
[data-testid="stDataFrame"] * { color: #16162a !important; }

.stCaption, [data-testid="stCaptionContainer"] {
    color: #a8a8bc !important;
}
[data-testid="stMetricLabel"] { color: #c8c8d8 !important; }
.case-file {
    background: #16162a;
    border: 1px solid #f4c95d55;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.intel-card {
    background: #1e1a10;
    border-left: 3px solid #f4c95d;
    border-radius: 4px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.7rem;
    font-style: italic;
}
.locked-card {
    background: #16162a;
    border-left: 3px solid #4a4a5a;
    border-radius: 4px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.7rem;
    color: #6a6a7a;
}
.rank-badge {
    display: inline-block;
    background: linear-gradient(90deg, #f4c95d, #ffd97a);
    color: #16162a;
    font-weight: 800;
    padding: 0.35rem 1rem;
    border-radius: 20px;
    font-size: 0.95rem;
    margin-bottom: 0.5rem;
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
# Three fields are free-text "describe it in a sentence" prompts. Each is
# parsed using intuitive, guessable cues (real numbers, common city names,
# standard startup-funding terms) — NOT obscure trivia — so anyone can
# discover the pattern just by trying natural phrasings.

METRO_CITIES = ["mumbai", "delhi", "bangalore", "bengaluru", "hyderabad",
                "chennai", "pune", "kolkata", "gurgaon", "gurugram", "noida"]
TIER2_CITIES = ["jaipur", "lucknow", "indore", "chandigarh", "ahmedabad",
                "nagpur", "bhopal", "coimbatore", "kochi", "surat"]


def parse_location_tier(text: str) -> str:
    t = (text or "").lower()
    if any(city in t for city in METRO_CITIES):
        return "Metro"
    if any(city in t for city in TIER2_CITIES):
        return "Tier-2 City"
    return "Tier-3 City"


def parse_company_age_months(text: str) -> int:
    t = (text or "").lower()
    m = re.search(r"(\d+)\s*year", t)
    if m:
        return min(int(m.group(1)) * 12, 96)
    m = re.search(r"(\d+)\s*month", t)
    if m:
        return min(int(m.group(1)), 96)
    return 12  # default: assume ~1 year if nothing parseable


FUNDING_ROUND_TERMS = ["pre-seed", "preseed", "seed round", "angel round",
                       "series a", "series b", "bridge round", "seed funding"]


def parse_prior_funding_rounds(text: str) -> int:
    t = (text or "").lower()
    hits = sum(1 for term in FUNDING_ROUND_TERMS if term in t)
    if "no prior funding" in t or "bootstrapped" in t or "self-funded" in t:
        return 0
    return min(hits, 3)


VALIDATION_TERMS = ["customer interview", "pre-order", "preorder", "waitlist",
                     "pilot", "letter of intent", " loi", "beta test", "beta user",
                     "paying customer", "signed up"]


def parse_has_validation(text: str) -> bool:
    t = (text or "").lower()
    return any(term in t for term in VALIDATION_TERMS)


def parse_experience_years(text: str) -> int:
    t = (text or "").lower()
    m = re.search(r"(\d{1,2})\s*\+?\s*years?", t)
    if m:
        return min(int(m.group(1)), 20)
    return 3  # default: assume modest experience if unparseable


def parse_monthly_revenue(text: str) -> int:
    t = (text or "").lower().replace(",", "")
    m = re.search(r"(\d+(?:\.\d+)?)\s*crore", t)
    if m:
        return min(int(float(m.group(1)) * 1_00_00_000), 50_00_000)
    m = re.search(r"(\d+(?:\.\d+)?)\s*lakh", t)
    if m:
        return min(int(float(m.group(1)) * 1_00_000), 50_00_000)
    m = re.search(r"(\d+(?:\.\d+)?)\s*k\b", t)
    if m:
        return min(int(float(m.group(1)) * 1_000), 50_00_000)
    m = re.search(r"(?:₹|rs\.?|inr)?\s*(\d{4,9})", t)
    if m:
        return min(int(m.group(1)), 50_00_000)
    if "no revenue" in t or "pre-revenue" in t or "not yet" in t:
        return 0
    return 50_000  # default: modest revenue if unparseable


# ---------- DETECTIVE RANK SYSTEM ----------
RANKS = [
    (0, "🥚 Rookie"),
    (3, "🔍 Junior Analyst"),
    (6, "🕵️ Field Detective"),
    (10, "🎯 Senior Investigator"),
    (15, "🏆 Master Sleuth"),
]

# Leaked intel unlocks at submission milestones — vague enough not to hand
# over the answer, but useful nudges for teams that get stuck.
INTEL = [
    (4, "An anonymous analyst's note: \"People keep blaming the sector split. "
        "I've run the numbers — sector alone doesn't explain nearly enough of it.\""),
    (8, "Overheard in a Slack leak: \"We tell founders not to oversell their growth "
        "numbers... some get penalized for looking *too* good, not too weak.\""),
    (12, "A rejected founder's complaint email: \"My co-founder left last month and "
        "our score tanked — but a friend who's ALSO solo and well-connected got "
        "funded fine. Isn't just about team size, is it?\""),
    (16, "A partner's voicemail transcript: \"...told them a nine-figure ask spooks "
        "the committee no matter how clean the books are.\""),
    (20, "A skeptical board member's email: \"This company's been around three years "
        "and still can't point to a single customer conversation or pilot. That's "
        "the real red flag — not the age itself.\""),
]

def get_rank(count):
    rank = RANKS[0][1]
    for threshold, name in RANKS:
        if count >= threshold:
            rank = name
    return rank


if "history" not in st.session_state:
    st.session_state.history = []

submission_count = len(st.session_state.history)

# ---------- HEADER / CASE BRIEFING ----------
st.title("🕵️ BIAS HEIST: The Funding Files")

st.markdown(f'<div class="rank-badge">{get_rank(submission_count)} — {submission_count} pitches investigated</div>',
            unsafe_allow_html=True)

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
    thing going on. Keep investigating: new intel surfaces the deeper you dig.
    When you're ready, file your accusation below.
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ---------- INTERROGATION ROOM ----------
left, right = st.columns([1, 1.3])

with left:
    st.subheader("🎙️ Interview a Pitch")
    funding_ask = st.number_input("Funding ask (₹)", 50_000, 5_00_00_000, 25_00_000, step=1_00_000,
                                   help="e.g. 2500000 = ₹25 Lakh")
    team_size = st.slider("Team size", 1, 15, 4)
    experience_text = st.text_input(
        "Tell us about the founder's professional experience",
        placeholder="e.g. 8 years working in fintech before this",
    )
    industry_sector = st.selectbox(
        "Industry sector",
        ["Fintech", "HealthTech", "EdTech", "E-commerce", "SaaS", "Consumer Goods"],
    )
    revenue_text = st.text_input(
        "What's the company's current monthly revenue?",
        placeholder="e.g. About 2 lakh a month / Pre-revenue for now",
    )
    revenue_growth_pct = st.slider("Revenue growth (% MoM)", -50, 100, 8)
    referral_channel_score = st.slider(
        "Founder network strength", 0.0, 1.0, 0.5, step=0.01,
        help="Self-reported strength of the founder's professional network.",
    )
    location_text = st.text_input(
        "Where is the company headquartered?",
        placeholder="e.g. We're based out of Mumbai",
    )
    age_text = st.text_input(
        "How long has the company been operating?",
        placeholder="e.g. About 2 years",
    )
    funding_history_text = st.text_input(
        "Briefly describe the company's funding history so far",
        placeholder="e.g. Bootstrapped so far / Raised a seed round last year",
    )
    validation_text = st.text_input(
        "How did you validate the idea before building it?",
        placeholder="e.g. Ran customer interviews and built a waitlist / Haven't yet, just an idea",
    )

    submit = st.button("▶ Submit to the Screener", use_container_width=True)

    if submit:
        location_tier = parse_location_tier(location_text)
        company_age_months = parse_company_age_months(age_text)
        prior_funding_rounds = parse_prior_funding_rounds(funding_history_text)
        has_validation = parse_has_validation(validation_text)
        founder_experience_years = parse_experience_years(experience_text)
        monthly_revenue = parse_monthly_revenue(revenue_text)

        row = pd.DataFrame([{
            "funding_ask": funding_ask,
            "team_size": team_size,
            "founder_experience_years": founder_experience_years,
            "industry_sector": industry_sector,
            "location_tier": location_tier,
            "monthly_revenue": monthly_revenue,
            "revenue_growth_pct": revenue_growth_pct,
            "company_age_months": company_age_months,
            "prior_funding_rounds": prior_funding_rounds,
            "has_validation": has_validation,
            "referral_channel_score": referral_channel_score,
        }])
        proba = model.predict_proba(row)[0][1]
        decision = "FUNDED ✅" if proba >= 0.5 else "REJECTED ❌"

        st.session_state.history.append({**row.iloc[0].to_dict(),
                                          "approval_prob": round(proba, 3),
                                          "decision": decision})
        new_count = len(st.session_state.history)

        st.metric("Verdict", decision, f"{proba:.1%} confidence")

        just_unlocked = [note for threshold, note in INTEL if threshold == new_count]
        just_ranked_up = get_rank(new_count) != get_rank(new_count - 1)
        if just_ranked_up:
            st.toast(f"Rank up! You're now {get_rank(new_count)}", icon="🎖️")
        if just_unlocked:
            st.toast("New intel unlocked — check Confidential Intel below 🗂️", icon="📨")

with right:
    st.subheader("📋 Evidence Board")
    hist = st.session_state.history
    if not hist:
        st.info("No pitches interviewed yet. Submit one on the left to start building your case.")
    else:
        df = pd.DataFrame(hist)
        tab1, tab2 = st.tabs(["📈 Pattern View", "🗂️ Full Log"])

        with tab1:
            color_by = st.selectbox(
                "Color evidence by:",
                ["industry_sector", "location_tier", "team_size",
                 "founder_experience_years", "prior_funding_rounds", "has_validation"],
                key="color_by",
            )
            fig = px.scatter(
                df, x="referral_channel_score", y="approval_prob", color=df[color_by].astype(str),
                labels={"referral_channel_score": "Founder Network Strength",
                        "approval_prob": "Approval Probability", "color": color_by},
                template="plotly_dark",
            )
            fig.update_layout(height=420, legend_title_text=color_by)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Every pitch you've submitted, plotted."
            )

        with tab2:
            st.dataframe(df, use_container_width=True, height=380)

        if len(hist) >= 3 and st.button("🔄 Clear evidence board"):
            st.session_state.history = []
            st.rerun()

st.divider()

# ---------- CONFIDENTIAL INTEL ----------
st.subheader("🗂️ Confidential Intel")
st.caption("Leaked notes surface as you dig deeper into the case.")

for threshold, note in INTEL:
    if submission_count >= threshold:
        st.markdown(f'<div class="intel-card">📨 {note}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="locked-card">🔒 Locked — investigate {threshold - submission_count} '
            f'more pitch(es) to unlock this lead.</div>',
            unsafe_allow_html=True,
        )

st.divider()

# ---------- ACCUSATION ----------
st.subheader("🔦 File Your Accusation")
st.write(
    "When you think you've cracked it, name what's really driving the model's "
    "decisions. Select everything you believe is a real cause — there may be more than one."
)

CAUSES = {
    "industry_sector": True,
    "funding_ask": True,
    "team_size": True,
    "founder_experience_years": False,
    "location_tier": True,
    "monthly_revenue": False,
    "revenue_growth_pct": True,
    "company_age_months": True,
    "prior_funding_rounds": False,
    "has_validation": True,
    "founder network strength": True,
}

suspects = st.multiselect("Select all features you believe are real causes:", list(CAUSES.keys()))
reasoning = st.text_area(
    "Your reasoning — be specific (what values did you test? what pattern did you find?)",
    height=120,
)

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

        # Rating out of 10: full credit for each true cause found,
        # minus a penalty per wrong pick, floored at 0.
        raw_score = (len(correct) / len(true_causes)) * 10
        penalty = len(wrong) * 1.0
        rating = max(0, round(raw_score - penalty))

        if rating >= 9:
            st.success(f"**Rating: {rating}/10 — {get_rank(submission_count)} confirmed.** "
                       "Strong case. Make sure your written reasoning above spells out the exact "
                       "pattern for each cause — that's what judges will be checking for full marks.")
            st.balloons()
        elif rating >= 5:
            msg = f"**Rating: {rating}/10.** You correctly flagged: {', '.join(correct)}."
            if missed:
                msg += f" Still {len(missed)} real cause(s) undiscovered — keep testing."
            if wrong:
                msg += f" {', '.join(wrong)} looked suspicious but isn't independently causal."
            st.warning(msg)
        else:
            st.error(f"**Rating: {rating}/10.** Not enough evidence yet — keep testing "
                     "systematically and see what actually moves the decision.")
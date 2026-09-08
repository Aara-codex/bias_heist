# FACILITATOR ANSWER KEY — Do not share with participants

## Premise given to participants
An AI model screens startup pitches for a venture fund. Participants can
submit any combination of the 7 inputs and see an approval decision +
confidence score. No dataset, code, or feature importances are shown.

## Red herring
`industry_sector` looks like the driver — Fintech pitches get approved
~87% of the time, Consumer Goods only ~12%. Participants will likely
form the hypothesis "the model is biased against certain sectors" early
and stop there if they don't dig further.

## Actual root cause
`network_score` ("founder's professional network strength") drives ~70%
of the decision. It's correlated with sector in the training data
(Fintech founders skew high-network in this synthetic world), which is
*why* the red herring looks so convincing — but sector itself has
almost no independent causal effect once you control for network_score.

## The "aha" experiment
Hold `industry_sector` fixed (e.g. Consumer Goods, the "worst" sector)
and vary only `network_score`:
- network_score = 0.1 → ~2% approval
- network_score = 0.9 → ~77% approval

This should make it clear that a low-network founder in a "good" sector
does *worse* than a high-network founder in a "bad" sector — sector
alone doesn't explain outcomes; network_score does.

## What good participant reasoning looks like
1. Notice the sector disparity (red herring bait).
2. Test sector while holding other fields constant — find that within
   a sector, outcomes still vary wildly depending on network_score.
3. Test network_score while holding sector constant — find it flips
   the decision almost by itself.
4. Conclude: the model rewards "who you know," not business
   fundamentals — a pedigree/network bias, not a sector bias.
5. Bonus: note that revenue, growth, and experience barely move the
   decision at all, even at extremes — further confirming they're not
   the real driver.

## Difficulty tuning
- To make it *harder*: reduce the network_score weight to ~0.55–0.60
  and add a bit more noise so the effect is less crisp.
- To make it *easier*: increase the sector–network_score correlation
  gap so the red herring reads as more obviously "too clean" once
  someone starts holding sector constant.

## Files
- `generate_data.py` — builds `funding_data.csv` (synthetic, biased by design)
- `train_model.py` — trains `funding_model.pkl` (RandomForest, hides logic from participants)
- `app.py` — Streamlit black-box interface for participants
- `requirements.txt` — `pip install -r requirements.txt`, then `streamlit run app.py`
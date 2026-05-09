# WorldCupPredictor

CS475/675 final project. Predicts 2026 FIFA World Cup match outcomes from
historical data, then Monte Carlo simulates the full tournament — group stage
plus 32-team knockout — sampling each match from the trained model's
calibrated probabilities.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Get the raw data (see DATA.md for the Drive snapshot link or per-source links)
#    Unzip into project root so the structure is data/raw/<files>.csv

# 3. Build the feature matrix (~30 sec)
python src/features.py

# 4. Train, tune, calibrate the models + run the ablation (~5-10 min)
python src/models.py

# 5. Monte Carlo simulate the 2026 World Cup (~14 min for n=100, ~140 min for n=1000)
python src/simulate.py --iters 1000 --seed 42 --output models/sim_summary.csv

# Optional: smoke-test the LLM baseline runner without API calls
python src/llm_baselines.py --provider heuristic --splits test --limit 5
```

## What the code does

### Match-outcome model (`src/features.py` + `src/models.py`)
Predicts win/draw/loss for any international football match. Three calibrated
models (multinomial logistic regression, random forest, XGBoost) trained on
2010-2021 matches, tuned on 2022-2024, evaluated on held-out 2025–2026-Q1.
A soft-voting ensemble averages the three. Test log loss is 0.827, which is
roughly the same band as bookmaker implied probabilities (~0.95) and the
published FiveThirtyEight WC '22 model (~1.00) — i.e. comparable to commercial
baselines using only public data.

Features per match (62 columns total = 31 base + 31 missingness flags):

| Group | Columns |
|---|---|
| Trailing form (last 10) | `home/away_form_{win_rate, gf, ga}` |
| Head-to-head | `home_h2h_win_rate` |
| Elo (eloratings.net formula) | `home_elo`, `away_elo`, `elo_diff` |
| FIFA rank | `home/away_fifa_rank`, `fifa_rank_diff` |
| Squad market value (date-correct cohorts) | `home/away_{squad_value, top26_value, avg_value, squad_size}`, `*_diff` |
| Position z-scores (4-tier source cascade) | `home/away_{attacking, creating, defending}_z`, `*_diff` |
| Venue | `neutral` |

The four-tier z-score cascade combines Understat (xG/xA, Big-5+RPL 2014-2024),
Transfermarkt scorer-list (Goals/Assists, 12 leagues 2008-2024), fotmob
(current 12-league snapshot), and fbref (defensive misc, Big-5 2017-2024). For
each cohort player at each match date, it picks the highest-tier source with
data for that (player, time period). See `notebooks/main.ipynb` for the writeup.

### Tournament simulation (`src/simulate.py`)
Monte Carlo over the 2026 WC. Each simulated match samples its outcome from
the model's calibrated probabilities, and the Elo tracker updates after every
match so a team's path through the bracket affects their later-round odds.
Bracket structure matches what FIFA actually published for 2026:

- Group stage is 12 groups of 4 with 6 matches each, ranked by points →
  goal difference → goals for → goals against. Top 2 + 8 best third-place
  teams advance.
- Knockout uses FIFA's predetermined slot specs from the published bracket
  (e.g. M73 = 2A vs 2B, M74 = 1E vs the third from groups A/B/C/D/F).
  The eight third-place teams are slotted via bipartite matching against
  FIFA's eligibility lists, which are designed so two teams from the same
  group can't meet again in R32. R16/QF/SF/Final follow FIFA's published
  pairing tree.
- Knockout draws resolve via a penalty-shootout model: per-team conversion
  rate (top 5 takers by attempts, from fbref) and GK save rate (with
  empirical-Bayes shrinkage so a 0/5 keeper doesn't read as 0%). Teams whose
  squads play mostly outside Big-5 leagues fall back to a dampened Elo prior.

```bash
# Headline run with the ensemble model + shootouts (recommended)
python src/simulate.py --iters 1000 --seed 42

# Compare across the three individual models for the report's "model agreement" table
for m in lr rf xgb ensemble; do
  python src/simulate.py --model $m --iters 100 --seed 42 \
      --output models/sim_${m}_100.csv
done

# A/B test the shootout model
python src/simulate.py --iters 1000 --seed 42 --no-shootouts \
    --output models/sim_no_shootouts.csv
```

### LLM baselines (`src/llm_baselines.py`)
Runs three prompt-controlled LLM comparison tracks against the same
`features.csv` splits as the ML models:

- `feature_only_blind`: anonymized Team A/B, engineered features only.
- `feature_plus_rag`: real teams, engineered features, and date-filtered
  context retrieved only from project data.
- `knowledge_only`: real fixture metadata only; no engineered features or RAG.

Outputs are written to `models/llm_predictions_*.csv` plus
`models/llm_eval_summary.json`. Probability order is the same as the ML code:
`[away_win, draw, home_win]`.

The runner auto-loads `.env` from the project root. For OpenAI GPT-5-family
models it sends `max_completion_tokens` rather than deprecated `max_tokens`.

```bash
# Smoke test without network/API usage. This validates plumbing only.
python src/llm_baselines.py --provider heuristic --splits test --limit 5

# Real OpenAI-compatible run. Put LLM_API_KEY / LLM_MODEL in .env first.
python src/llm_baselines.py --provider openai-compatible --splits val test

# Include unplayed 2026 fixtures for qualitative predictions.
python src/llm_baselines.py --provider openai-compatible --splits val test --include-predict
```

The `knowledge_only` track is intentionally reported as a qualitative
pretrained-knowledge prior, not as a leakage-free benchmark, because model
pretraining may already encode famous historical outcomes.

## Train / val / test / predict splits

Splits are chronological, not random — a random split would leak future state
into training, since the whole point is forecasting future matches.

| Split | Years | Rows | Label coverage |
|---|---|---|---|
| train | 2010 → 2021-12 | 11,218 | 100% |
| val | 2022 → 2024-12 | 3,252 | 100% |
| test | 2025 → 2026-03 | 1,162 | 100% |
| predict | 2026 WC fixtures | 72 | 0% (unplayed) |

## Project structure

```
WorldCupPredictor/
├── src/
│   ├── team_names.py                  # canonical team-name normalization
│   ├── elo.py                         # eloratings.net Elo formula
│   ├── features.py                    # feature engineering → features.csv
│   ├── models.py                      # train + tune + calibrate + ensemble + ablate
│   ├── ensemble.py                    # SoftVoteEnsemble class (loaded from ensemble.pkl)
│   ├── simulate.py                    # Monte Carlo tournament simulation
│   ├── scrape_tournaments.py          # Transfermarkt tournament squads (44 editions)
│   ├── scrape_understat.py            # Understat per-season Big-5 + RPL
│   ├── scrape_fotmob.py               # fotmob current snapshot (12 leagues)
│   ├── scrape_fbref.py                # fbref defensive misc table
│   └── scrape_transfermarkt_seasons.py # TM scorerlist (12 leagues × 17 seasons)
├── data/
│   ├── raw/                           # all source CSVs (gitignored, see DATA.md)
│   └── processed/
│       └── features.csv               # generated by features.py
├── models/                            # gitignored — pickled artifacts + JSON summaries
│   ├── lr.pkl, rf.pkl, xgb.pkl        # calibrated per-model predictors
│   ├── ensemble.pkl                   # soft-voting ensemble (picklable class)
│   ├── scaler.pkl, fill_values.pkl, feature_names.pkl, classes.pkl
│   ├── best_params.json, summary.json, ablation.json, tuning_log.json
│   └── sim_summary*.csv               # per-team Monte Carlo outputs
├── notebooks/
│   └── main.ipynb                     # final report writeup
├── DATA.md                            # data source documentation
├── README.md
└── requirements.txt
```

## Known limitations

- FIFA rankings end 2024-06-20. Anything after that uses the latest snapshot.
- Z-score coverage is ~32% on training (2010-2021) vs ~96% on the 2026 predict
  split, because Understat and fbref's Big-5 advanced stats only go back to
  2014/2017. We add missingness flags so the model can tell imputed values
  from real ones, but the distribution shift is real.
- Penalty conversion rates and GK save rates come from Big-5 only. Teams whose
  squads mostly play outside the Big 5 fall back to a dampened-Elo shootout.
- We don't separately simulate extra time before the shootout — a knockout
  match that draws goes straight to penalties in our sim.
- For the third-place team-to-slot assignment, when more than one valid
  matching exists, FIFA's published Annex C picks a specific one; we use a
  best-performing-third-first tie-break. The eligibility constraints are
  always respected, so structurally our bracket is identical to one the real
  tournament could produce.

See the notebook (`notebooks/main.ipynb`) for the full writeup including methodology, results, and discussion.

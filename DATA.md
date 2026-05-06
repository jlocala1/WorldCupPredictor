# Data sources

All raw data files live in `data/raw/` (gitignored). Two ways to get them:

1. Download the zipped snapshot from our shared Drive:
   `https://drive.google.com/file/d/1WyAYG6sRs9LpP9Fgu-xfvnJff1pLE6r1/view?usp=drive_link`
   Unzip into the project root so the structure is `data/raw/<files>.csv`.
   (Note: zip should be regenerated whenever raw data changes — update the link above.)
2. Or pull each dataset from its source below and run `python src/scrape_tournaments.py`
   to generate `tournament_squads.csv`.

## Sources

### 1. International match results — `martj42` (Kaggle)
- Search: "International football results 1872 martj42" on Kaggle
- Files placed in `data/raw/`:
  - `results.csv` — 49k matches 1872 → 2026 (includes the published 2026 WC fixture list with NaN scores for unplayed matches)
  - `shootouts.csv` — historical penalty-shootout outcomes (used for bracket simulation tiebreakers)
  - `goalscorers.csv` — per-goal records (carried but not currently wired into features)

### 2. FIFA world ranking history — `cashncarry` (Kaggle)
- URL: https://www.kaggle.com/datasets/cashncarry/fifaworldranking
- Files placed in `data/raw/`:
  - `fifa_ranking-2024-06-20.csv` — 67k team-date rows across 333 ranking dates from 1992-12-31 to 2024-06-20

### 3. World Cup historical data — `Fjelstul` (Kaggle)
- Search: "Fjelstul world cup database" on Kaggle
- Files placed in `data/raw/`:
  - `matches.csv` — WC match results
  - `squads.csv` — WC tournament rosters
  - `players.csv` — WC player info
  - `team_appearances.csv` — team stats per WC match
  - `manager_appointments.csv` — who managed each team at each WC (no start/end dates; manager-tenure feature was considered but dropped — see Limitations)
  - `group_standings.csv` — historical WC group standings
  - `tournament_standings.csv` — how far each team went
  - `player_appearances.csv` — per-WC player rosters with starter/sub flag (carried; potential future feature)

### 4. Transfermarkt player & match data — `davidcariboo/player-scores` (Kaggle)
- URL: https://www.kaggle.com/datasets/davidcariboo/player-scores
- Source GitHub (the scraper): https://github.com/dcaribou/transfermarkt-datasets
- Files placed in `data/raw/`:
  - `player_valuations.csv` — 616k historical per-player market values (2000-2026)
  - `playerstransfer.csv` — player profiles (renamed from `players.csv` to avoid collision with Fjelstul's file)
  - `national_teams.csv` — 119 national team summaries (current squad value, FIFA rank, coach)
  - `countries.csv` — country metadata
  - `appearances.csv` — 1.86M player-game appearances. **Per-player rows for international games are sparse** in this dataset (only AFCON 2025 covered), which is why we scraped tournament rosters ourselves — see section 6.
  - `competitions.csv` — competition metadata; used to filter `appearances` to `national_team_competition`

### 5. fbref player stats (Big 5 leagues, current season)
- Hubertsidorowicz Kaggle dataset (basic columns only — Standard, Keeper, Shooting basics, Playing Time, Misc)
- Files placed in `data/raw/`:
  - `players_data_light-2025_2026.csv` (light)
  - `players_data-2025_2026.csv` (full but same columns minus a few duplicates)
- **Limitation:** lacks xG, xA, progressive passes, total Tkl, Blocks. Will be replaced by direct fbref scrape (`src/scrape_fbref.py`, planned) for the position-z-score feature.

### 6. Tournament rosters — scraped from Transfermarkt (`src/scrape_tournaments.py`)
- Generated locally; not from Kaggle.
- File: `data/raw/tournament_squads.csv`. **38,792 rows / 16,827 unique players / 44 tournament instances** since 2006:
  - World Cup (5): 2006, 2010, 2014, 2018, 2022
  - UEFA Euro (5): 2008, 2012, 2016, 2020, 2024
  - Copa América (6): 2007, 2011, 2015, 2019, 2021, 2024
  - AFCON (9): 2010, 2012, 2013, 2015, 2017, 2019, 2021, 2023, 2025
  - AFC Asian Cup (5): 2007, 2011, 2015, 2019, 2023
  - CONCACAF Gold Cup (10): 2007, 2009, 2011, 2013, 2015, 2017, 2019, 2021, 2023, 2025
  - OFC Nations Cup (1): 2024
  - Confederations Cup (3): 2009, 2013, 2017
- **100% coverage of all 48 WC 2026 teams.**
- Player IDs match `player_valuations.csv` (both Transfermarkt-sourced), enabling
  date-correct cohort joins for the squad-value feature.
- Re-run the scraper anytime: `python src/scrape_tournaments.py`. It is resumable
  (skips tournaments already in the CSV).

## Feature → source mapping

| Feature column(s) | Source file(s) | Status |
|---|---|---|
| `home_form_*`, `away_form_*` (last-10 win rate, GF, GA) | `results.csv` | done |
| `home_h2h_win_rate` | `results.csv` | done |
| `home_elo`, `away_elo`, `elo_diff` | `results.csv` (full 1872+ history) | done |
| `neutral` | `results.csv` | done |
| `home_fifa_rank`, `away_fifa_rank`, `fifa_rank_diff` | `fifa_ranking-2024-06-20.csv` | done |
| `home_squad_value`, `away_squad_value`, `*_top26_value`, `*_avg_value`, `*_squad_size` | `tournament_squads.csv` + `player_valuations.csv` | done |
| `home_avg_caps`, `away_avg_caps` | `tournament_squads.csv` + `playerstransfer.csv` | done |
| Position-aggregated z-scores (forwards xG, mids PrgP/xA, defenders Tkl/Int) | fbref scrape (planned) + `tournament_squads.csv` | planned |
| Shootout-winner tiebreaker (bracket simulation) | `shootouts.csv` | planned |

## Known limitations

- **FIFA rankings end 2024-06-20.** Matches after that fall back to the latest snapshot (~22 months stale by mid-2026). Documented in the report.
- **`international_caps` is a current Transfermarkt snapshot**, not date-indexed. For a 2014 match, a player who has 80 caps now contributes 80 even if they had 15 in 2014. Cohort assignment is date-correct via `tournament_squads.csv`; only the cap *numbers* are anachronistic.
- **Training-set squad-value coverage is ~38-44%** (both teams). The remainder are matches involving smaller national teams that never played in a major continental tournament since 2006. Handled in `models.py` with median imputation + a missingness indicator column for linear models; XGBoost handles NaN natively.
- **Manager tenure was considered but dropped.** Fjelstul's `manager_appointments.csv` only records who managed each team at each WC tournament — no start/end dates. A 4-year-bucketed approximation was too coarse to be useful, so we removed the feature rather than feed noisy data into the model.
- **fbref Kaggle dataset is missing advanced metrics.** xG, xA, PrgP, total Tkl, Blocks aren't in the columns the curator scraped. Resolved by scraping fbref directly (planned).

# Data sources

All raw data files live in `data/raw/` (gitignored). Two ways to get them:

1. Download the zipped snapshot from our shared Drive:
   `https://drive.google.com/file/d/1WyAYG6sRs9LpP9Fgu-xfvnJff1pLE6r1/view?usp=drive_link`
   Unzip into the project root so the structure is `data/raw/<files>.csv`.

## Sources

### 1. International match results — `martj42` (Kaggle)
- Search: "International football results 1872 martj42" on Kaggle
- Files placed in `data/raw/`:
  - `results.csv` — 49k matches 1872 → 2026 (includes the published 2026 WC fixture list with NaN scores for unplayed matches)
  - `goalscorers.csv`, `shootouts.csv` — unused as of 5/5

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
  - `manager_appointments.csv` — manager tenure data
  - `group_standings.csv` — historical WC group standings
  - `tournament_standings.csv` — how far each team went
  - `player_appearances.csv` — present but unused (redundant with davidcariboo)

### 4. Transfermarkt player & match data — `davidcariboo/player-scores` (Kaggle)
- URL: https://www.kaggle.com/datasets/davidcariboo/player-scores
- Source GitHub (the scraper): https://github.com/dcaribou/transfermarkt-datasets
- Files placed in `data/raw/`:
  - `player_valuations.csv` — 616k historical per-player market values (2000-2026)
  - `playerstransfer.csv` — player profiles (renamed from `players.csv` to avoid collision with Fjelstul's file)
  - `national_teams.csv` — 119 national team summaries (current squad value, FIFA rank, coach)
  - `countries.csv` — country metadata
  - `appearances.csv` — 1.86M player-game appearances (national + club)
  - `competitions.csv` — competition metadata; used to filter `appearances` to `national_team_competition`
  - `games.csv` — present but unused (redundant; appearances has the date and competition_id directly)

### 5. fbref player stats
- Search Kaggle for "fbref player stats 2025-2026"
- Files placed in `data/raw/`:
  - `players_data_light-2025_2026.csv` — Big 5 leagues player stats (light version, used)
  - `players_data-2025_2026.csv` — full version (present, unused)
- Used only for the position-z-score stretch feature

## Feature → source mapping

| Feature column(s) | Source file(s) | Status |
|---|---|---|
| `home_form_*`, `away_form_*` (last-10 win rate, GF, GA) | `results.csv` | done |
| `home_h2h_win_rate` | `results.csv` | done |
| `home_elo`, `away_elo`, `elo_diff` | `results.csv` (full 1872+ history) | done |
| `neutral` | `results.csv` | done |
| `home_fifa_rank`, `away_fifa_rank`, `fifa_rank_diff` | `fifa_ranking-2024-06-20.csv` | done |
| `home_squad_value`, `away_squad_value`, `*_top26_value`, `*_avg_value` | `player_valuations.csv` + `appearances.csv` + `national_teams.csv` + `competitions.csv` | planned |
| `home_avg_caps`, `away_avg_caps` | `playerstransfer.csv` | planned |
| `manager_tenure_months` (home/away) | `manager_appointments.csv` | planned |
| Position-aggregated z-scores (stretch) | `players_data_light-2025_2026.csv` | stretch |

## Important

- **FIFA rankings end 2024-06-20.** Matches after that fall back to the latest snapshot (~22 months stale by mid-2026). We accept this and footnote it in the report.
- **4 WC teams missing from `national_teams.csv`:** Cameroon, DR Congo, Ivory Coast, Mali. Squad-value features for those teams need a fallback path (likely `playerstransfer.country_of_citizenship` aggregation).
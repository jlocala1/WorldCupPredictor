# Data sources

All raw data files live in `data/raw/` (gitignored). Two ways to get them:

1. Download the zipped snapshot from our shared Drive:
   `https://drive.google.com/file/d/1WyAYG6sRs9LpP9Fgu-xfvnJff1pLE6r1/view?usp=drive_link`
   Unzip into the project root so the structure is `data/raw/<files>.csv`.
   (Note: zip should be regenerated whenever raw data changes — update the link above.)
2. Or pull each public dataset from its source below and run our scrapers to
   regenerate the locally-built files:
   ```
   python src/scrape_tournaments.py            # tournament_squads.csv
   python src/scrape_understat.py              # understat_player_stats.csv
   python src/scrape_fotmob.py                 # fotmob_player_stats.csv
   python src/scrape_fbref.py                  # fbref_player_stats.csv
   python src/scrape_transfermarkt_seasons.py  # transfermarkt_player_seasons.csv
   ```
   Each scraper is resumable (skips entries already in the output CSV).

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
- Hubertsidorowicz Kaggle dataset
- Files placed in `data/raw/`:
  - `players_data_light-2025_2026.csv` (light, fewer columns)
  - `players_data-2025_2026.csv` (102 columns including penalty conversion `PK`/`PKatt` and goalkeeper penalty saves `PKsv`/`PKatt_stats_keeper`)
- The full file is what feeds the penalty-shootout model in `src/simulate.py` (top-5 takers' conversion rate + main GK save rate per team).
- The file lacks some advanced metrics like xG/xA/progressive passes, which is why we also scrape Understat / fbref / fotmob ourselves for the z-score features (see `src/scrape_*.py`).

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
- Implementation note: requires `seleniumbase` with `Driver(uc=True)` to bypass
  Transfermarkt's Cloudflare protection on squad pages.

### 7. Per-season Understat stats — `src/scrape_understat.py`
- File: `data/raw/understat_player_stats.csv`. **~29,800 player-seasons** across the Big 5 leagues + Russian Premier League, 2014-2024.
- Columns include the advanced metrics we wanted from fbref but couldn't get from Kaggle: `goals`, `xG`, `npxG`, `assists`, `xA`, `shots`, `key_passes`, `xGChain`, `xGBuildup`, `npg` (non-penalty goals), plus playing time and disciplinary stats.
- Implementation note: Understat removed the `playersData` JSON.parse pattern that older scrapers relied on. We reverse-engineered an internal endpoint at `/getLeagueData/{LEAGUE}/{YEAR}` and call it with the `X-Requested-With: XMLHttpRequest` header to get the full season as JSON.
- Tier 1 of the position-z-score cascade in `features.py` (most advanced data, but only Big 5 + RPL since 2014).

### 8. fotmob current-snapshot stats — `src/scrape_fotmob.py`
- File: `data/raw/fotmob_player_stats.csv`. **~6,427 players** across 12 leagues for the current 2025-26 season.
- Leagues covered: Big 5 (England/Spain/Italy/Germany/France) + MLS, Saudi Pro, Liga MX, Brasileirão, Eredivisie, Liga Portugal, Belgian Pro League.
- Columns include offensive metrics (`goals_total`, `xG_total`, `xA_total`, `xGOT_total`, `key_passes_total`, `dribbles_per90`, `big_chance_total`) plus defensive (`tackles_per90`, `int_per90`, `blocks_per90`, `recoveries_per90`).
- Implementation note: fotmob's `data.fotmob.com` CDN is openly accessible with a `Referer: https://www.fotmob.com/` header — no Selenium needed. The internal API path moved from `/api/leagues?id=X` to `/api/data/leagues?id=X`.
- Tier 3 of the cascade — anachronistic for older matches (current-only) but covers many non-Big-5 league players that Understat doesn't.

### 9. fbref defensive misc stats — `src/scrape_fbref.py`
- File: `data/raw/fbref_player_stats.csv`. **~22,425 player-seasons** across the Big 5, 2017-2024.
- Pulled via the `soccerdata` package (which wraps the fbref tables). We use the misc/defending table for `Performance_TklW` and `Performance_Int`.
- Implementation note: fbref stripped xG/xA/progressive-passes from their public HTML in 2024, which is why this scraper is limited to defensive stats only. The advanced offensive stats we get from Understat instead.
- Tier 4 of the cascade — defensive-only fallback for `defending_z`.

### 10. Transfermarkt scorer-list per-season — `src/scrape_transfermarkt_seasons.py`
- File: `data/raw/transfermarkt_player_seasons.csv`. **~5,100 player-seasons** across 12 leagues × 17 seasons (2008-2024).
- Leagues: Big 5 + MLS, Saudi Pro, Liga MX, Eredivisie, Liga Portugal, Brasileirão, Belgian Pro.
- Columns: `goals`, `assists`, `apps` (per league-season). Top ~25 scorers per league per season — covers the players who actually take penalties and rack up output.
- Player IDs match `player_valuations.csv` (same Transfermarkt source), so joins are exact without name matching.
- Tier 2 of the cascade — basic stats (no xG), but the only multi-league source going back to 2008.

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
| Position z-scores (4-tier cascade: attacking, creating, defending) | `understat_player_stats.csv` + `transfermarkt_player_seasons.csv` + `fotmob_player_stats.csv` + `fbref_player_stats.csv` + `tournament_squads.csv` | done |
| Knockout-draw shootout (per-team conversion + GK save rate) | `players_data-2025_2026.csv` | done |
| Shootout outcomes (referenced for sanity-checking, not used as a feature) | `shootouts.csv` | done |

## Known limitations

- **FIFA rankings end 2024-06-20.** Matches after that fall back to the latest snapshot (~22 months stale by mid-2026). Documented in the report.
- **`international_caps` is a current Transfermarkt snapshot**, not date-indexed. For a 2014 match, a player who has 80 caps now contributes 80 even if they had 15 in 2014. Cohort assignment is date-correct via `tournament_squads.csv`; only the cap *numbers* are anachronistic.
- **Training-set squad-value coverage is ~38-44%** (both teams). The remainder are matches involving smaller national teams that never played in a major continental tournament since 2006. Handled in `models.py` with median imputation + a missingness indicator column for linear models; XGBoost handles NaN natively.
- **Manager tenure was considered but dropped.** Fjelstul's `manager_appointments.csv` only records who managed each team at each WC tournament — no start/end dates. A 4-year-bucketed approximation was too coarse to be useful, so we removed the feature rather than feed noisy data into the model.
- **fbref Kaggle dataset is missing advanced metrics.** xG, xA, PrgP, total Tkl, Blocks aren't in the columns the curator scraped. We worked around it by scraping the data ourselves from Understat (xG/xA, Big-5 + RPL since 2014), fotmob (12-league current snapshot), and fbref directly via soccerdata (defensive misc-table). See `src/scrape_*.py`.

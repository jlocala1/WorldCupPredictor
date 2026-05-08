"""Feature engineering for the World Cup match-outcome predictor.

Reads results.csv plus auxiliary sources, normalizes team names, and writes
a single match-level feature matrix to data/processed/features.csv.

The pipeline is built up in phases; each phase adds columns to the same
output file. Run end-to-end with: python src/features.py

  Phase 1 (this file): base frame from results.csv — filtered to 2010+,
                       team names normalized, label and chronological split
                       attached.
  Phase 2 (next):      trailing-form features, head-to-head, Elo.
  Phase 3 (later):     FIFA rank, squad value, caps, manager tenure.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from elo import compute_pre_match_elo
from team_names import to_canonical

DATA_RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
DATA_PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"

# Chronological split boundaries (inclusive endpoints) per CLAUDE.md.
START_DATE = pd.Timestamp("2010-01-01")
TRAIN_END = pd.Timestamp("2021-12-31")
VAL_END = pd.Timestamp("2024-12-31")


def load_base_frame() -> pd.DataFrame:
    """Read results.csv, filter to 2010+, normalize team names, attach label and split.

    Rows with missing scores are scheduled-but-unplayed fixtures (results.csv
    publishes the 2026 WC schedule ahead of kickoff with NaN scores). They get
    split='predict' and label=NaN so training code can filter them out and the
    simulator can pick them up by name.
    """
    df = pd.read_csv(DATA_RAW / "results.csv", parse_dates=["date"])
    df = df[df["date"] >= START_DATE].copy()

    df["home_team"] = df["home_team"].map(to_canonical)
    df["away_team"] = df["away_team"].map(to_canonical)

    has_score = df["home_score"].notna() & df["away_score"].notna()

    # Label: 2 = home win, 1 = draw, 0 = away win (per CLAUDE.md). NaN for unplayed.
    diff = df["home_score"] - df["away_score"]
    label = np.where(diff > 0, 2.0, np.where(diff == 0, 1.0, 0.0))
    df["label"] = np.where(has_score, label, np.nan)

    df["split"] = pd.Series(index=df.index, dtype="object")
    df.loc[has_score & (df["date"] <= TRAIN_END), "split"] = "train"
    df.loc[has_score & (df["date"] > TRAIN_END) & (df["date"] <= VAL_END), "split"] = "val"
    df.loc[has_score & (df["date"] > VAL_END), "split"] = "test"
    df.loc[~has_score, "split"] = "predict"

    df["neutral"] = df["neutral"].astype(bool)

    return df.sort_values("date").reset_index(drop=True)


def _build_long_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape match-level df into one row per (team, match), keeping both played and unplayed.

    Each match contributes two rows: home perspective and away perspective. Used as the
    base for any per-team time-series feature (trailing form, Elo, etc).
    """
    home = df[["date", "home_team", "away_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "away_team": "opp", "home_score": "gf", "away_score": "ga"}
    )
    away = df[["date", "away_team", "home_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "home_team": "opp", "away_score": "gf", "home_score": "ga"}
    )
    return pd.concat([home, away], ignore_index=True)


def add_trailing_form(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Add trailing-N-match win rate, avg goals scored, avg goals conceded for each side.

    Each played row holds the post-match rolling stat (i.e. the rate including that
    match's own outcome). The merge then uses allow_exact_matches=False, which makes
    every target row — played or predict — read the latest *strictly-prior* played
    row. That avoids the off-by-one where predict rows would have missed the most
    recent played match's outcome.
    """
    long = _build_long_frame(df)
    played = long.dropna(subset=["gf", "ga"]).copy()
    played["win"] = (played["gf"] > played["ga"]).astype(float)
    played = played.sort_values(["team", "date"]).reset_index(drop=True)

    grouped = played.groupby("team", group_keys=False)
    played["form_win_rate"] = grouped["win"].transform(lambda s: s.rolling(window, min_periods=1).mean())
    played["form_gf"] = grouped["gf"].transform(lambda s: s.rolling(window, min_periods=1).mean())
    played["form_ga"] = grouped["ga"].transform(lambda s: s.rolling(window, min_periods=1).mean())

    form = played[["team", "date", "form_win_rate", "form_gf", "form_ga"]].sort_values(["date", "team"])

    df = df.sort_values("date").reset_index(drop=True)

    home_form = form.rename(
        columns={
            "team": "home_team",
            "form_win_rate": "home_form_win_rate",
            "form_gf": "home_form_gf",
            "form_ga": "home_form_ga",
        }
    )
    df = pd.merge_asof(
        df, home_form, on="date", by="home_team",
        direction="backward", allow_exact_matches=False,
    )

    away_form = form.rename(
        columns={
            "team": "away_team",
            "form_win_rate": "away_form_win_rate",
            "form_gf": "away_form_gf",
            "form_ga": "away_form_ga",
        }
    )
    df = pd.merge_asof(
        df, away_form, on="date", by="away_team",
        direction="backward", allow_exact_matches=False,
    )
    return df


def add_fifa_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Merge home/away FIFA rank as of the closest preceding ranking publication.

    The cashncarry CSV holds 333 ranking dates from 1992-12-31 to 2024-06-20, so
    matches after 2024-06 fall back to the latest available snapshot (a known
    staleness limitation we accept, see CLAUDE.md).
    """
    rank = pd.read_csv(DATA_RAW / "fifa_ranking-2024-06-20.csv", parse_dates=["rank_date"])
    rank["country_full"] = rank["country_full"].map(to_canonical)
    rank = rank.rename(columns={"country_full": "team", "rank_date": "date", "rank": "fifa_rank"})
    rank = rank[["team", "date", "fifa_rank"]].dropna(subset=["team"]).sort_values(["date", "team"])

    df = df.sort_values("date").reset_index(drop=True)

    home_rank = rank.rename(columns={"team": "home_team", "fifa_rank": "home_fifa_rank"})
    df = pd.merge_asof(
        df, home_rank, on="date", by="home_team",
        direction="backward", allow_exact_matches=True,
    )

    away_rank = rank.rename(columns={"team": "away_team", "fifa_rank": "away_fifa_rank"})
    df = pd.merge_asof(
        df, away_rank, on="date", by="away_team",
        direction="backward", allow_exact_matches=True,
    )

    df["fifa_rank_diff"] = df["home_fifa_rank"] - df["away_fifa_rank"]
    return df


def _build_tournament_cohort(df: pd.DataFrame, window_years: int = 4,
                             forward_years: int = 1) -> pd.DataFrame:
    """Per (team, match_date), the cohort = unique players who appeared in any
    tournament_squads.csv tournament with date in (match_date - window_years,
    match_date + forward_years).

    Returns DataFrame with columns: team, match_date, player_id.
    Teams with no tournament-squad data get no cohort rows (NaN downstream).
    """
    ts = pd.read_csv(DATA_RAW / "tournament_squads.csv", parse_dates=["tournament_date"])
    ts["team_canon"] = ts["team_name"].map(to_canonical, na_action="ignore")
    ts = ts.dropna(subset=["team_canon"])
    ts = ts[["team_canon", "tournament_date", "player_id"]].rename(
        columns={"team_canon": "team"}
    )

    home_keys = df[["date", "home_team"]].rename(columns={"home_team": "team", "date": "match_date"})
    away_keys = df[["date", "away_team"]].rename(columns={"away_team": "team", "date": "match_date"})
    match_keys = pd.concat([home_keys, away_keys]).drop_duplicates().reset_index(drop=True)

    cohort = match_keys.merge(ts, on="team", how="inner")
    backward = pd.Timedelta(days=window_years * 365)
    forward = pd.Timedelta(days=forward_years * 365)
    cohort = cohort[
        (cohort["tournament_date"] > cohort["match_date"] - backward)
        & (cohort["tournament_date"] < cohort["match_date"] + forward)
    ]
    return cohort[["team", "match_date", "player_id"]].drop_duplicates().reset_index(drop=True)


def add_squad_value(df: pd.DataFrame) -> pd.DataFrame:
    """Date-correct squad value features per CLAUDE.md spec.

    Cohort: per (team, match_date), players who appeared for that team in any
    scraped major tournament (WC, Euro, Copa, AFCON, Asian Cup, Gold Cup, OFC
    Nations Cup, Confederations Cup) within (D-4yr, D+1yr).
    Per-player value: looked up from player_valuations.csv at the match date
    via backward merge_asof. So a 2014 Brazil match uses the 2014-era Brazil
    squad valued in 2014.

    Adds: home/away_squad_value, home/away_top26_value, home/away_avg_value,
          home/away_squad_size, plus squad_value_diff and top26_value_diff.
    """
    cohort = _build_tournament_cohort(df)

    valuations = pd.read_csv(
        DATA_RAW / "player_valuations.csv",
        usecols=["player_id", "date", "market_value_in_eur"],
        parse_dates=["date"],
    ).sort_values("date").reset_index(drop=True)

    cohort = cohort.sort_values("match_date").reset_index(drop=True)
    cohort = pd.merge_asof(
        cohort, valuations,
        left_on="match_date", right_on="date",
        by="player_id",
        direction="backward",
    ).drop(columns="date")
    cohort = cohort.dropna(subset=["market_value_in_eur"])

    def top26_sum(values: pd.Series) -> float:
        return float(values.nlargest(26).sum())

    agg = cohort.groupby(["team", "match_date"], sort=False).agg(
        squad_value=("market_value_in_eur", "sum"),
        top26_value=("market_value_in_eur", top26_sum),
        avg_value=("market_value_in_eur", "mean"),
        squad_size=("market_value_in_eur", "count"),
    ).reset_index()

    home_agg = agg.rename(columns={
        "team": "home_team", "match_date": "date",
        "squad_value": "home_squad_value", "top26_value": "home_top26_value",
        "avg_value": "home_avg_value", "squad_size": "home_squad_size",
    })
    df = df.merge(home_agg, on=["home_team", "date"], how="left")

    away_agg = agg.rename(columns={
        "team": "away_team", "match_date": "date",
        "squad_value": "away_squad_value", "top26_value": "away_top26_value",
        "avg_value": "away_avg_value", "squad_size": "away_squad_size",
    })
    df = df.merge(away_agg, on=["away_team", "date"], how="left")

    df["squad_value_diff"] = df["home_squad_value"] - df["away_squad_value"]
    df["top26_value_diff"] = df["home_top26_value"] - df["away_top26_value"]
    return df


def _normalize_player_name(name: str) -> str:
    """Strip accents + lowercase + collapse whitespace for fuzzy matching."""
    if not isinstance(name, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", name)
    no_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    return " ".join(no_accents.lower().split())


def _z_score(series: pd.Series) -> pd.Series:
    """Per-group population z-score (ddof=0). Used after groupby.transform."""
    std = series.std(ddof=0)
    return (series - series.mean()) / std if std and std > 0 else pd.Series(np.nan, index=series.index)


def _fotmob_player_scores() -> pd.DataFrame:
    """Per-player attacking_z + creating_z + defending_z from fotmob's data CDN.

    Stats used (all per-90, normalized within (league) since fotmob current-
    season covers many leagues with very different talent depth):
      attacking_z = mean(goals_p90_z, xG_p90_z)
      creating_z  = mean(assists_p90_z, xA_p90_z, big_chance_p90_z)
      defending_z = mean(tackles_per90_z, int_per90_z, blocks_per90_z, recoveries_per90_z)

    Coverage: Big 5 + MLS + Saudi Pro + Liga MX + Eredivisie + Liga Portugal +
    Brasileirão + Belgian Pro. ~6,400 players.

    Returns DataFrame keyed by `norm_name` with the three z-scores. Missing
    fotmob CSV -> empty frame.
    """
    path = DATA_RAW / "fotmob_player_stats.csv"
    if not path.exists():
        return pd.DataFrame(columns=["norm_name", "attacking_z", "creating_z", "defending_z"])
    fm = pd.read_csv(path)
    fm = fm[fm["minutes"].fillna(0) >= 450].copy()
    if fm.empty:
        return pd.DataFrame(columns=["norm_name", "attacking_z", "creating_z", "defending_z"])

    # Convert totals to per-90 for consistency with other stats
    nineties = fm["minutes"] / 90.0
    for src, dst in [("goals_total", "goals_p90"), ("assists_total", "assists_p90"),
                     ("xG_total", "xG_p90"), ("xA_total", "xA_p90"),
                     ("xGOT_total", "xGOT_p90"),
                     ("big_chance_total", "big_chance_p90"),
                     ("key_passes_total", "key_passes_p90")]:
        if src in fm.columns:
            fm[dst] = fm[src] / nineties

    # Z-score every per-90 stat within (league) — current season, so no season grouping
    grouped = fm.groupby("league", group_keys=False)
    for stat in ["goals_p90", "assists_p90", "xG_p90", "xA_p90", "xGOT_p90",
                 "big_chance_p90", "key_passes_p90", "dribbles_per90",
                 "tackles_per90", "int_per90", "blocks_per90", "recoveries_per90"]:
        if stat in fm.columns:
            fm[f"{stat}_z"] = grouped[stat].transform(_z_score)

    def avg_z(*cols):
        present = [fm[c] for c in cols if c in fm.columns]
        if not present:
            return pd.Series(np.nan, index=fm.index)
        return pd.concat(present, axis=1).mean(axis=1)

    fm["attacking_z"] = avg_z("goals_p90_z", "xG_p90_z", "xGOT_p90_z")
    fm["creating_z"] = avg_z("assists_p90_z", "xA_p90_z", "big_chance_p90_z",
                             "key_passes_p90_z", "dribbles_per90_z")
    fm["defending_z"] = avg_z("tackles_per90_z", "int_per90_z",
                              "blocks_per90_z", "recoveries_per90_z")

    fm["norm_name"] = fm["player_name"].map(_normalize_player_name)
    # If a player appears in multiple leagues (loanee mid-season etc.), keep the
    # row with most minutes — most representative of their level.
    fm = fm.sort_values("minutes", ascending=False).drop_duplicates("norm_name")
    return fm[["norm_name", "attacking_z", "creating_z", "defending_z"]].reset_index(drop=True)


def _understat_player_scores_per_season() -> pd.DataFrame:
    """Per-(player, season) attacking_z + creating_z from Understat.

    Returns one row per (player, year) — does NOT collapse to a single
    snapshot per player. This lets add_position_zscores look up each cohort
    player's stats from the season corresponding to the match date, giving
    date-correct z-scores for matches in 2014-2024 where Understat has data.

    Per-90 stats are z-scored within (league, year) pool — so a player is
    compared to others in the same league in the same season.

    Missing file -> empty frame.
    """
    path = DATA_RAW / "understat_player_stats.csv"
    if not path.exists():
        return pd.DataFrame(columns=["norm_name", "year", "attacking_z", "creating_z"])
    us = pd.read_csv(path)
    us = us[us["time"].fillna(0) >= 450].copy()
    if us.empty:
        return pd.DataFrame(columns=["norm_name", "year", "attacking_z", "creating_z"])

    nineties = us["time"] / 90.0
    for stat in ["goals", "xG", "npxG", "assists", "xA", "key_passes"]:
        us[f"{stat}_p90"] = us[stat] / nineties

    grouped = us.groupby(["league", "year"], group_keys=False)
    for stat in ["goals", "xG", "npxG", "assists", "xA", "key_passes"]:
        us[f"{stat}_z"] = grouped[f"{stat}_p90"].transform(_z_score)

    us["attacking_z"] = us[["goals_z", "xG_z", "npxG_z"]].mean(axis=1)
    us["creating_z"] = us[["assists_z", "xA_z", "key_passes_z"]].mean(axis=1)
    us["norm_name"] = us["player_name"].map(_normalize_player_name)

    return us[["norm_name", "year", "attacking_z", "creating_z"]].dropna(
        subset=["norm_name"]
    ).reset_index(drop=True)


def _transfermarkt_player_seasons() -> pd.DataFrame:
    """Per-(player_id, season) attacking_z + creating_z from Transfermarkt scorerliste.

    Tier 2 in the cascade — supplies date-correct *basic* stats (Goals, Assists)
    for older Big 5 matches (pre-2014) and non-Big-5 leagues (MLS, Saudi Pro,
    Liga MX, Brasileirão, Eredivisie, Liga Portugal, Belgian Pro) where
    Understat doesn't reach. Joins to cohort by `player_id` directly — same
    Transfermarkt ID system we already use elsewhere, so no name matching.

    Returns one row per (player_id, season) — NOT collapsed to a snapshot.
    Composite scores are based on Goals/Apps and Assists/Apps (per-90 proxy
    since TM doesn't expose minutes on this page) z-scored within
    (league_code, season).
    """
    path = DATA_RAW / "transfermarkt_player_seasons.csv"
    if not path.exists():
        return pd.DataFrame(columns=["player_id", "season", "attacking_z", "creating_z"])
    tm = pd.read_csv(path)
    tm = tm[tm["apps"].fillna(0) >= 5].copy()
    if tm.empty:
        return pd.DataFrame(columns=["player_id", "season", "attacking_z", "creating_z"])

    # Per-appearance rates (proxy for per-90 — TM scorerliste doesn't expose minutes)
    tm["goals_per_app"] = tm["goals"] / tm["apps"]
    tm["assists_per_app"] = tm["assists"] / tm["apps"]

    grouped = tm.groupby(["league_code", "season"], group_keys=False)
    tm["attacking_z"] = grouped["goals_per_app"].transform(_z_score)
    tm["creating_z"] = grouped["assists_per_app"].transform(_z_score)

    return tm[["player_id", "season", "attacking_z", "creating_z"]].dropna(
        subset=["player_id"]
    ).reset_index(drop=True)


def _football_season_year(match_date: pd.Timestamp) -> int | None:
    """Map a calendar date to its football season's start year.

    Football seasons run Aug-May; Understat's `year` is the start year. So:
      - Aug-Dec match  → season starts that year   (e.g. 2017-10-15 → 2017)
      - Jan-May match  → season started prior year (e.g. 2018-03-12 → 2017)
      - Jun-Jul match  → off-season; treat as prior-year season

    Returns None if input is NaT.
    """
    if pd.isna(match_date):
        return None
    return int(match_date.year if match_date.month >= 8 else match_date.year - 1)


def _fbref_defending_scores_per_season() -> pd.DataFrame:
    """Per-(player, season) defending_z from the fbref `misc` table.

    Used as the defensive equivalent of `_understat_player_scores_per_season`:
    keeps one row per (player × season) so add_position_zscores can do a
    backward merge_asof and get date-correct defending_z for cohort players
    based on the match's football-season year.

    Coverage: Big 5 leagues only, 2017-2024 (the seasons soccerdata
    successfully cached before fbref stripped advanced HTML). Pre-2017 and
    non-Big-5 cohort players still get current-fotmob fallback or NaN.

    fbref's `season` column is the season-end year format like "1718", "2425".
    We map this to the football season's start year (e.g., 1718 → 2017) so
    it lines up with the merge_asof semantics used for Understat.

    Returns: norm_name, year, defending_z. Missing file -> empty frame.
    """
    path = DATA_RAW / "fbref_player_stats.csv"
    if not path.exists():
        return pd.DataFrame(columns=["norm_name", "year", "defending_z"])
    fb = pd.read_csv(path)

    def find(*needles):
        for col in fb.columns:
            if all(n in col for n in needles):
                return col
        return None

    min_col = find("Min")
    tklw_col = find("TklW")
    int_col = find("Int") or find("Performance_Int")
    if not (min_col and tklw_col and int_col):
        return pd.DataFrame(columns=["norm_name", "year", "defending_z"])

    fb = fb[fb[min_col].fillna(0) >= 450].copy()
    if fb.empty:
        return pd.DataFrame(columns=["norm_name", "year", "defending_z"])

    nineties = fb[min_col] / 90.0
    fb["TklW_p90"] = fb[tklw_col] / nineties
    fb["Int_p90"] = fb[int_col] / nineties

    grouped = fb.groupby(["league", "season"], group_keys=False)
    fb["TklW_z"] = grouped["TklW_p90"].transform(_z_score)
    fb["Int_z"] = grouped["Int_p90"].transform(_z_score)
    fb["defending_z"] = fb[["TklW_z", "Int_z"]].mean(axis=1)

    fb["norm_name"] = fb["player"].map(_normalize_player_name)

    # Map fbref's "1718"-style season string to the football season's start year (2017).
    def _fbref_season_to_start_year(s):
        try:
            s = str(s).strip()
            # Forms like "1718", "2425", "2017-2018", "2017-18"
            digits = "".join(c for c in s if c.isdigit())
            if len(digits) == 4:
                # "1718" -> 2017
                start = int("20" + digits[:2])
                return start
            if len(digits) == 8:
                # "20172018" or "2017-2018"
                return int(digits[:4])
            return None
        except Exception:
            return None

    fb["year"] = fb["season"].apply(_fbref_season_to_start_year)
    fb = fb.dropna(subset=["norm_name", "year"])
    fb["year"] = fb["year"].astype(int)
    return fb[["norm_name", "year", "defending_z"]].reset_index(drop=True)


def add_position_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """Position-blind player skill z-scores aggregated to team level.

    Per-player score cascade (best stats from the time period):
      Tier 1: Understat per-season match — for the cohort player's stats
              from the season corresponding to the match date (closest-prior
              fallback). Available 2014-2024 for Big 5 + RPL leagues.
              Date-correct.
      Tier 2: fotmob current-season snapshot — multi-league (12 leagues
              including MLS / Saudi / Liga MX / Brasileirão / etc.) but
              current 2025-26 only. Anachronistic for older matches.
      Tier 3: fbref `misc` table for defensive stats only (TklW, Int).
              Latest-season-per-player snapshot.

    Composite scores per player:
      attacking_z = mean(goals/90_z, xG/90_z, npxG/90_z) [Understat]
                    OR mean(goals/90_z, xG/90_z, xGOT/90_z) [fotmob]
      creating_z  = mean(assists/90_z, xA/90_z, KP/90_z) [Understat]
                    OR mean(...big_chances, key_passes, dribbles) [fotmob]
      defending_z = mean(Tkl/90_z, Int/90_z, Blk/90_z, Recov/90_z) [fotmob]
                    OR mean(TklW/90_z, Int/90_z) [fbref misc fallback]

    Per (team, match_date): score = mean of TOP 8 cohort players per skill.

    Adds 9 columns: home/away_{attacking,creating,defending}_z + 3 diffs.

    Limitations (documented in report):
      - **No progressive passes / progressive carries** — fbref recently
        stripped these from public HTML; Understat doesn't track them.
      - **Pre-2014 matches and non-Big-5 players use the fotmob current
        snapshot** — anachronistic, but bounded: for predict (2026 WC) the
        snapshot IS the right time period.
      - **No covered league at all** for some smaller national teams →
        contributes NaN, model handles via imputation.
    """
    out_cols = ["home_attacking_z", "home_creating_z", "home_defending_z",
                "away_attacking_z", "away_creating_z", "away_defending_z",
                "attacking_z_diff", "creating_z_diff", "defending_z_diff"]

    us_per_season = _understat_player_scores_per_season()
    tm_per_season = _transfermarkt_player_seasons()
    fm_current = _fotmob_player_scores()
    fb_per_season = _fbref_defending_scores_per_season()

    if (us_per_season.empty and tm_per_season.empty
            and fm_current.empty and fb_per_season.empty):
        for c in out_cols:
            df[c] = np.nan
        return df

    # Build cohort with player names + each match's season year
    cohort = _build_tournament_cohort(df)
    ts = pd.read_csv(DATA_RAW / "tournament_squads.csv", parse_dates=["tournament_date"])
    ts["norm_name"] = ts["player_name"].map(_normalize_player_name)
    name_lookup = ts[["player_id", "norm_name"]].drop_duplicates("player_id")
    cohort = cohort.merge(name_lookup, on="player_id", how="left")
    cohort["season_year"] = cohort["match_date"].apply(_football_season_year).astype("Int64")

    # === Tier 1: Understat per-season backward-merge ===
    # For each (player, match_season), look up Understat stats from that season
    # or the closest prior season the player played.
    if not us_per_season.empty:
        us_sorted = us_per_season.sort_values("year").reset_index(drop=True)
        cohort_sorted = cohort.sort_values("season_year").reset_index(drop=True)
        # merge_asof requires non-NaN sort key; drop rows missing season_year and re-merge later
        valid_mask = cohort_sorted["season_year"].notna() & cohort_sorted["norm_name"].notna()
        valid = cohort_sorted[valid_mask].copy()
        valid["season_year"] = valid["season_year"].astype(int)
        valid = pd.merge_asof(
            valid,
            us_sorted,
            left_on="season_year", right_on="year",
            by="norm_name",
            direction="backward",
        ).drop(columns=["year"], errors="ignore")
        valid = valid.rename(columns={
            "attacking_z": "att_us", "creating_z": "cre_us"
        })
        # Stitch back: rows that didn't have a valid sort key get NaN att_us/cre_us
        cohort = cohort.merge(
            valid[["team", "match_date", "player_id", "att_us", "cre_us"]],
            on=["team", "match_date", "player_id"], how="left"
        )
    else:
        cohort["att_us"] = np.nan
        cohort["cre_us"] = np.nan

    # === Tier 2: Transfermarkt per-season backward-merge by player_id ===
    # Date-correct basic stats (Gls/Ast) for older Big 5 + all non-Big-5 leagues.
    # Joins by player_id (same TM IDs as tournament_squads), no name matching.
    if not tm_per_season.empty:
        tm_sorted = tm_per_season.sort_values("season").reset_index(drop=True)
        cohort_sorted_pid = cohort.sort_values("season_year").reset_index(drop=True)
        valid_mask_tm = (
            cohort_sorted_pid["season_year"].notna()
            & cohort_sorted_pid["player_id"].notna()
        )
        valid_tm = cohort_sorted_pid[valid_mask_tm].copy()
        valid_tm["season_year"] = valid_tm["season_year"].astype(int)
        valid_tm = pd.merge_asof(
            valid_tm, tm_sorted,
            left_on="season_year", right_on="season",
            by="player_id",
            direction="backward",
        ).drop(columns=["season"], errors="ignore")
        valid_tm = valid_tm.rename(columns={
            "attacking_z": "att_tm", "creating_z": "cre_tm"
        })
        cohort = cohort.merge(
            valid_tm[["team", "match_date", "player_id", "att_tm", "cre_tm"]],
            on=["team", "match_date", "player_id"], how="left"
        )
    else:
        cohort["att_tm"] = np.nan
        cohort["cre_tm"] = np.nan

    # === Tier 3: fotmob current snapshot fallback ===
    if not fm_current.empty:
        cohort = cohort.merge(
            fm_current.rename(columns={
                "attacking_z": "att_fm",
                "creating_z": "cre_fm",
                "defending_z": "def_fm",
            }),
            on="norm_name", how="left",
        )
    else:
        cohort["att_fm"] = np.nan
        cohort["cre_fm"] = np.nan
        cohort["def_fm"] = np.nan

    # === Tier 4: fbref per-season defending (Big 5 2017-2024) ===
    # Date-correct defending_z for Big 5 cohort players in the relevant seasons.
    # Same merge_asof pattern as Understat (by norm_name, on year, backward).
    if not fb_per_season.empty:
        fb_sorted = fb_per_season.sort_values("year").reset_index(drop=True)
        cohort_sorted_fb = cohort.sort_values("season_year").reset_index(drop=True)
        valid_mask_fb = (
            cohort_sorted_fb["season_year"].notna()
            & cohort_sorted_fb["norm_name"].notna()
        )
        valid_fb = cohort_sorted_fb[valid_mask_fb].copy()
        valid_fb["season_year"] = valid_fb["season_year"].astype(int)
        valid_fb = pd.merge_asof(
            valid_fb, fb_sorted,
            left_on="season_year", right_on="year",
            by="norm_name",
            direction="backward",
        ).drop(columns=["year"], errors="ignore")
        valid_fb = valid_fb.rename(columns={"defending_z": "def_fb"})
        cohort = cohort.merge(
            valid_fb[["team", "match_date", "player_id", "def_fb"]],
            on=["team", "match_date", "player_id"], how="left"
        )
    else:
        cohort["def_fb"] = np.nan

    # Resolve cascade. Priority order for attacking/creating:
    #   Tier 1 Understat-per-season (xG/xA, advanced) →
    #   Tier 2 Transfermarkt-per-season (basic Gls/Ast, date-correct) →
    #   Tier 3 fotmob current (advanced but anachronistic).
    # Defending: fotmob current → fbref misc fallback.
    cohort["attacking_z"] = (
        cohort["att_us"]
        .combine_first(cohort["att_tm"])
        .combine_first(cohort["att_fm"])
    )
    cohort["creating_z"] = (
        cohort["cre_us"]
        .combine_first(cohort["cre_tm"])
        .combine_first(cohort["cre_fm"])
    )
    cohort["defending_z"] = cohort["def_fm"].combine_first(cohort["def_fb"])

    def top_n_mean(series: pd.Series, n: int = 8) -> float:
        clean = series.dropna()
        if clean.empty:
            return float("nan")
        return float(clean.nlargest(n).mean())

    agg = cohort.groupby(["team", "match_date"], sort=False).agg(
        team_attacking_z=("attacking_z", top_n_mean),
        team_creating_z=("creating_z", top_n_mean),
        team_defending_z=("defending_z", top_n_mean),
    ).reset_index()

    home = agg.rename(columns={
        "team": "home_team", "match_date": "date",
        "team_attacking_z": "home_attacking_z",
        "team_creating_z": "home_creating_z",
        "team_defending_z": "home_defending_z",
    })
    df = df.merge(home, on=["home_team", "date"], how="left")

    away = agg.rename(columns={
        "team": "away_team", "match_date": "date",
        "team_attacking_z": "away_attacking_z",
        "team_creating_z": "away_creating_z",
        "team_defending_z": "away_defending_z",
    })
    df = df.merge(away, on=["away_team", "date"], how="left")

    df["attacking_z_diff"] = df["home_attacking_z"] - df["away_attacking_z"]
    df["creating_z_diff"] = df["home_creating_z"] - df["away_creating_z"]
    df["defending_z_diff"] = df["home_defending_z"] - df["away_defending_z"]
    return df


def add_caps(df: pd.DataFrame) -> pd.DataFrame:
    """Avg international caps over the date-correct tournament cohort.

    Note: caps come from playerstransfer.international_caps (a CURRENT-snapshot
    total), so cap *numbers* are still anachronistic for older matches even
    though the *cohort* is now date-correct. Documented limitation.
    """
    cohort = _build_tournament_cohort(df)

    players = pd.read_csv(
        DATA_RAW / "playerstransfer.csv",
        usecols=["player_id", "international_caps"],
    ).dropna(subset=["international_caps"])

    cohort = cohort.merge(players, on="player_id", how="inner")
    agg = cohort.groupby(["team", "match_date"], sort=False).agg(
        avg_caps=("international_caps", "mean"),
    ).reset_index()

    home = agg.rename(columns={"team": "home_team", "match_date": "date", "avg_caps": "home_avg_caps"})
    df = df.merge(home, on=["home_team", "date"], how="left")
    away = agg.rename(columns={"team": "away_team", "match_date": "date", "avg_caps": "away_avg_caps"})
    df = df.merge(away, on=["away_team", "date"], how="left")
    return df


def add_elo(df: pd.DataFrame) -> pd.DataFrame:
    """Attach pre-match Elo ratings (and their difference) for both sides.

    Elo is computed over the FULL results.csv history (1872+) so that ratings
    by 2010 already reflect decades of prior results. K-factor varies by
    tournament type and home advantage is applied for non-neutral games — see
    src/elo.py for the full eloratings.net formula.
    """
    full = pd.read_csv(DATA_RAW / "results.csv", parse_dates=["date"])
    full["home_team"] = full["home_team"].map(to_canonical)
    full["away_team"] = full["away_team"].map(to_canonical)
    full["neutral"] = full["neutral"].astype(bool)

    full_elo = compute_pre_match_elo(full)
    elo_lookup = full_elo[["date", "home_team", "away_team", "home_elo_pre", "away_elo_pre"]].rename(
        columns={"home_elo_pre": "home_elo", "away_elo_pre": "away_elo"}
    )

    df = df.merge(elo_lookup, on=["date", "home_team", "away_team"], how="left")
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    return df


def add_h2h(df: pd.DataFrame) -> pd.DataFrame:
    """Add the home team's prior win rate against the away team across all earlier meetings.

    NaN if the two teams have never played each other before this date.
    """
    long = _build_long_frame(df)
    long = long.dropna(subset=["gf", "ga"]).copy()
    long["win"] = (long["gf"] > long["ga"]).astype(float)
    long = long.sort_values(["team", "opp", "date"]).reset_index(drop=True)

    long["h2h_win_rate"] = long.groupby(["team", "opp"], group_keys=False)["win"].transform(
        lambda s: s.expanding().mean()
    )
    h2h = long[["team", "opp", "date", "h2h_win_rate"]].sort_values(["date", "team", "opp"])

    df = df.sort_values("date").reset_index(drop=True)
    h2h_named = h2h.rename(
        columns={"team": "home_team", "opp": "away_team", "h2h_win_rate": "home_h2h_win_rate"}
    )
    df = pd.merge_asof(
        df, h2h_named, on="date", by=["home_team", "away_team"],
        direction="backward", allow_exact_matches=False,
    )
    return df


def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df = load_base_frame()
    df = add_trailing_form(df, window=10)
    df = add_h2h(df)
    df = add_elo(df)
    df = add_fifa_rank(df)
    df = add_squad_value(df)
    df = add_caps(df)
    df = add_position_zscores(df)

    print(f"Total matches:        {len(df):,}")
    print(f"Date range:           {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Unique teams:         {pd.concat([df['home_team'], df['away_team']]).nunique()}")
    print(f"Neutral venue rate:   {df['neutral'].mean():.1%}")
    print()
    split_counts = df["split"].value_counts().reindex(["train", "val", "test", "predict"])
    print("Split sizes:")
    for split, n in split_counts.items():
        sub = df[df["split"] == split]
        date_range = f"{sub['date'].min().date()} -> {sub['date'].max().date()}" if len(sub) else "(empty)"
        print(f"  {split:<7} {n:>6,}   {date_range}")
    print()
    labeled = df[df["label"].notna()]
    print(f"Label distribution (overall, {len(labeled):,} labeled matches):")
    label_names = {2.0: "home win", 1.0: "draw", 0.0: "away win"}
    for label, count in labeled["label"].value_counts().sort_index(ascending=False).items():
        print(f"  {int(label)} ({label_names[label]:<8}) {count:>6,}   {count / len(labeled):.1%}")
    print()
    print("Label distribution by split (home win / draw / away win):")
    for split in ["train", "val", "test"]:
        sub = labeled[labeled["split"] == split]
        if not len(sub):
            continue
        rates = sub["label"].value_counts(normalize=True).reindex([2.0, 1.0, 0.0]).fillna(0)
        print(f"  {split:<5} {rates[2.0]:.1%}  {rates[1.0]:.1%}  {rates[0.0]:.1%}")

    predict = df[df["split"] == "predict"]
    if len(predict):
        wc_teams = sorted(set(predict["home_team"]) | set(predict["away_team"]))
        print(f"\n2026 WC fixtures to predict: {len(predict)} matches across {len(wc_teams)} teams")
        print(f"Teams: {', '.join(wc_teams)}")

    print("\nFeature coverage (% non-null) by split:")
    feature_cols = [c for c in df.columns if c.startswith((
        "home_form_", "away_form_", "home_h2h_",
        "home_elo", "away_elo", "elo_",
        "home_fifa_", "away_fifa_", "fifa_",
        "home_squad_", "away_squad_", "squad_value_",
        "home_top26_", "away_top26_", "top26_value_",
        "home_avg_value", "away_avg_value",
        "home_avg_caps", "away_avg_caps",
        "home_attacking_z", "home_creating_z", "home_defending_z",
        "away_attacking_z", "away_creating_z", "away_defending_z",
        "attacking_z_diff", "creating_z_diff", "defending_z_diff",
    ))]
    for split in ["train", "val", "test", "predict"]:
        sub = df[df["split"] == split]
        if not len(sub):
            continue
        coverage = {c: f"{sub[c].notna().mean():.0%}" for c in feature_cols}
        print(f"  {split:<7} {coverage}")

    print("\nTop 10 Elo ratings going into the 2026 WC (latest known per team):")
    predict = df[df["split"] == "predict"].copy()
    if len(predict):
        latest = pd.concat([
            predict[["home_team", "home_elo"]].rename(columns={"home_team": "team", "home_elo": "elo"}),
            predict[["away_team", "away_elo"]].rename(columns={"away_team": "team", "away_elo": "elo"}),
        ]).drop_duplicates("team").sort_values("elo", ascending=False)
        for _, r in latest.head(10).iterrows():
            print(f"  {r['team']:<25} {r['elo']:.0f}")
        print("  ...")
        print(f"  Bottom 3 of WC field:")
        for _, r in latest.tail(3).iterrows():
            print(f"  {r['team']:<25} {r['elo']:.0f}")

    print("\nTop 10 squad value (€) going into the 2026 WC (latest known per team):")
    if len(predict):
        latest_sv = pd.concat([
            predict[["home_team", "home_squad_value", "home_top26_value", "home_squad_size"]].rename(
                columns={"home_team": "team", "home_squad_value": "sv", "home_top26_value": "top26", "home_squad_size": "n"}
            ),
            predict[["away_team", "away_squad_value", "away_top26_value", "away_squad_size"]].rename(
                columns={"away_team": "team", "away_squad_value": "sv", "away_top26_value": "top26", "away_squad_size": "n"}
            ),
        ]).drop_duplicates("team").sort_values("sv", ascending=False)
        for _, r in latest_sv.head(10).iterrows():
            sv_str = f"{r['sv']/1e6:>6.0f}M" if pd.notna(r["sv"]) else "    —"
            top_str = f"{r['top26']/1e6:>6.0f}M" if pd.notna(r["top26"]) else "    —"
            n_str = f"{int(r['n'])}" if pd.notna(r["n"]) else "—"
            print(f"  {r['team']:<25} squad={sv_str}  top26={top_str}  n={n_str}")
        print(f"\n  Bottom 5 of WC field:")
        for _, r in latest_sv.tail(5).iterrows():
            sv_str = f"{r['sv']/1e6:>6.0f}M" if pd.notna(r["sv"]) else "    —"
            top_str = f"{r['top26']/1e6:>6.0f}M" if pd.notna(r["top26"]) else "    —"
            n_str = f"{int(r['n'])}" if pd.notna(r["n"]) else "—"
            print(f"  {r['team']:<25} squad={sv_str}  top26={top_str}  n={n_str}")

    print("\nSample predict-split rows (Brazil's matches):")
    sample = df[(df["split"] == "predict") & ((df["home_team"] == "Brazil") | (df["away_team"] == "Brazil"))]
    for _, r in sample.iterrows():
        h2h_str = f"{r['home_h2h_win_rate']:.2f}" if pd.notna(r["home_h2h_win_rate"]) else "first mtg"
        rank_h = f"#{int(r['home_fifa_rank'])}" if pd.notna(r["home_fifa_rank"]) else "—"
        rank_a = f"#{int(r['away_fifa_rank'])}" if pd.notna(r["away_fifa_rank"]) else "—"
        print(f"  {r['date'].date()}  {r['home_team']:<25} ({rank_h}) vs {r['away_team']:<25} ({rank_a})  "
              f"elo_diff={r['elo_diff']:+.0f}  h2h={h2h_str}")

    out = DATA_PROCESSED / "features.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out} ({len(df):,} rows, {len(df.columns)} columns)")


if __name__ == "__main__":
    main()

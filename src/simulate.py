"""Monte Carlo simulation of the 2026 FIFA World Cup.

Pipeline:
  1. Group stage  - 12 groups of 4 teams each, 6 matches per group (72 total).
                    Standings sorted by FIFA tiebreakers (points -> goal
                    difference -> goals for). Top 2 + 8 best 3rd-place teams
                    advance.
  2. Knockout     - 32-team single-elim bracket arranged in standard seeded
                    order so top seeds avoid each other in early rounds.
                    Draws in regulation are resolved by coin flip
                    (TODO: replace with Elo-weighted shootout model).

Each simulated match calls model.predict_proba(X) and samples an outcome from
the resulting (away, draw, home) probability vector. Elo updates after every
simulated match so a team's path through the tournament is reflected in their
later matches.

Run:
    python src/simulate.py                    # 1000 iterations, ensemble model
    python src/simulate.py --model lr_raw    # use a different model
    python src/simulate.py --iters 5000      # more iterations
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Make src/ importable so `from elo import ...` works when running as a script
# AND so joblib can resolve the SoftVoteEnsemble class on unpickle.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from elo import EloRating  # noqa: E402

# -----------------------------------------------------------------------------
# Paths and constants
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DATA_PATH = ROOT / "data" / "processed" / "features.csv"

# 2026 World Cup groups, derived programmatically from the predict-split
# fixtures in features.csv (each group's 4 teams play exactly 6 matches among
# themselves at group stage, so the groupings are recoverable from the fixture
# adjacency graph).
GROUPS_2026: dict[str, list[str]] = {
    "A": ["Algeria", "Argentina", "Austria", "Jordan"],
    "B": ["Australia", "Paraguay", "Turkey", "United States"],
    "C": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "D": ["Bosnia and Herzegovina", "Canada", "Qatar", "Switzerland"],
    "E": ["Brazil", "Haiti", "Morocco", "Scotland"],
    "F": ["Cape Verde", "Saudi Arabia", "Spain", "Uruguay"],
    "G": ["Colombia", "DR Congo", "Portugal", "Uzbekistan"],
    "H": ["Croatia", "England", "Ghana", "Panama"],
    "I": ["Curaçao", "Ecuador", "Germany", "Ivory Coast"],
    "J": ["Czech Republic", "Mexico", "South Africa", "South Korea"],
    "K": ["France", "Iraq", "Norway", "Senegal"],
    "L": ["Japan", "Netherlands", "Sweden", "Tunisia"],
}

# Diff features: each is computed as home_X - away_X. Earlier versions of
# run_match_sim missed these and they fell through to median imputation,
# which destroyed the z-score signal in particular.
DIFF_FEATURES: dict[str, tuple[str, str]] = {
    "fifa_rank_diff":   ("home_fifa_rank",   "away_fifa_rank"),
    "squad_value_diff": ("home_squad_value", "away_squad_value"),
    "top26_value_diff": ("home_top26_value", "away_top26_value"),
    "attacking_z_diff": ("home_attacking_z", "away_attacking_z"),
    "creating_z_diff":  ("home_creating_z",  "away_creating_z"),
    "defending_z_diff": ("home_defending_z", "away_defending_z"),
}

# Standard seeded 32-team single-elim bracket. Reading top-to-bottom, adjacent
# pairs are R32 matches and the structure ensures seed 1 only meets seed 2 in
# the final, seeds 1-4 are in different quarters, etc.
BRACKET_ORDER_32 = [
     1, 32, 16, 17,  9, 24,  8, 25,
     5, 28, 12, 21, 13, 20,  4, 29,
     3, 30, 14, 19, 11, 22,  6, 27,
     7, 26, 10, 23, 15, 18,  2, 31,
]

# FIFA 3-letter codes -> our canonical team names. Used to map the `Nation`
# column in players_data-2025_2026.csv (format "fr FRA") to the team-name
# strings we use everywhere else. Only the 48 WC 2026 teams are listed here;
# 5 of those teams (Cape Verde, Curaçao, Iran, Iraq, Qatar) have no players
# in the file because their squads play primarily outside Big-5 leagues, so
# they fall through to the Elo-based shootout fallback.
FIFA_CODE_TO_TEAM: dict[str, str] = {
    "ALG": "Algeria",   "ARG": "Argentina",  "AUS": "Australia",  "AUT": "Austria",
    "BEL": "Belgium",   "BIH": "Bosnia and Herzegovina", "BRA": "Brazil",
    "CAN": "Canada",    "CIV": "Ivory Coast","COD": "DR Congo",   "COL": "Colombia",
    "CRO": "Croatia",   "CZE": "Czech Republic", "ECU": "Ecuador","EGY": "Egypt",
    "ENG": "England",   "ESP": "Spain",      "FRA": "France",     "GER": "Germany",
    "GHA": "Ghana",     "HAI": "Haiti",      "JPN": "Japan",      "JOR": "Jordan",
    "KOR": "South Korea","KSA": "Saudi Arabia", "MAR": "Morocco", "MEX": "Mexico",
    "NED": "Netherlands","NOR": "Norway",    "NZL": "New Zealand","PAN": "Panama",
    "PAR": "Paraguay",  "POR": "Portugal",   "RSA": "South Africa","SCO": "Scotland",
    "SEN": "Senegal",   "SUI": "Switzerland","SWE": "Sweden",     "TUN": "Tunisia",
    "TUR": "Turkey",    "URU": "Uruguay",    "USA": "United States", "UZB": "Uzbekistan",
}

# Hand-picked coefficients for the shootout model. K_DATA scales the player-
# level signal; ELO_DAMP compresses Elo's effect so a 200-rating gap maps to
# roughly 60% odds (real shootouts are noisy even between mismatched teams).
SHOOTOUT_K_DATA = 5.0
SHOOTOUT_ELO_DAMP = 1000.0
SHOOTOUT_MIN_ATTEMPTS = 3   # min PKatt across top-5 takers to count as "has data"
SHOOTOUT_GK_FALLBACK_RATE = 0.22  # global mean GK PK save rate (~22% in football)
SHOOTOUT_GK_PRIOR_STRENGTH = 10.0  # EB shrinkage: 5-attempt sample is given 33% data weight
SHOOTOUT_BLEND_DATA_WEIGHT = 0.6  # tier-2 blend: 60% data signal, 40% Elo prior


def _shrunk_gk_rate(rate: float, attempts: int) -> float:
    """Empirical-Bayes shrinkage of GK save rate toward the global mean.

    A keeper's per-season penalty save rate is computed on tiny samples (often
    fewer than 10 attempts), so raw rates are dominated by noise — a 0/5
    keeper isn't really 0%, they just had a bad sample. We shrink toward
    SHOOTOUT_GK_FALLBACK_RATE with prior strength SHOOTOUT_GK_PRIOR_STRENGTH:
        shrunk = (saves + k*mu) / (attempts + k)
    With k=10, a 5-attempt sample carries 5/(5+10) = 33% data weight; a
    20-attempt sample carries 67%.
    """
    if pd.isna(rate) or attempts <= 0:
        return SHOOTOUT_GK_FALLBACK_RATE
    saves = rate * attempts
    return (saves + SHOOTOUT_GK_PRIOR_STRENGTH * SHOOTOUT_GK_FALLBACK_RATE) \
           / (attempts + SHOOTOUT_GK_PRIOR_STRENGTH)


# -----------------------------------------------------------------------------
# Model and feature loading
# -----------------------------------------------------------------------------
def load_artifacts(model_name: str = "ensemble"):
    """Load model + preprocessing artifacts.

    `model_name` is the stem of one of the files in models/ (e.g. "ensemble",
    "lr", "rf", "xgb", "lr_raw").

    The SoftVoteEnsemble (`ensemble.pkl`) bundles its own scaler internally and
    accepts unscaled features. The standalone uncalibrated `lr_raw.pkl` needs
    the external `scaler.pkl` applied before predict_proba.
    """
    # Required so joblib.load can resolve the SoftVoteEnsemble class on unpickle.
    from ensemble import SoftVoteEnsemble  # noqa: F401

    model = joblib.load(MODELS_DIR / f"{model_name}.pkl")
    fill_values = joblib.load(MODELS_DIR / "fill_values.pkl")
    feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
    classes = joblib.load(MODELS_DIR / "classes.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl") if model_name == "lr_raw" else None
    return model, fill_values, feature_names, classes, scaler


def build_team_snapshots(predict_df: pd.DataFrame) -> dict[str, dict]:
    """Build per-team feature snapshots from the predict-split rows.

    Team-level features (form_*, squad_value, fifa_rank, *_z, etc.) are nearly
    identical across a team's three group-stage rows because they're computed
    at the tournament date. This function takes the first available row for
    each team and stores their home_* and away_* feature values keyed without
    the prefix - so we can paste them onto either side in any future matchup.
    """
    snapshots: dict[str, dict] = {}
    teams = set(predict_df["home_team"]).union(predict_df["away_team"])
    for team in teams:
        snap: dict = {}
        home_rows = predict_df[predict_df["home_team"] == team]
        if not home_rows.empty:
            row = home_rows.iloc[0]
            for col in row.index:
                if col.startswith("home_"):
                    snap[col[5:]] = row[col]
        away_rows = predict_df[predict_df["away_team"] == team]
        if not away_rows.empty:
            row = away_rows.iloc[0]
            for col in row.index:
                if col.startswith("away_"):
                    key = col[5:]
                    if key not in snap or pd.isna(snap[key]):
                        snap[key] = row[col]
        snapshots[team] = snap
    return snapshots


def compute_h2h_win_rate(team_a: str, team_b: str, df: pd.DataFrame) -> float:
    """All-time h2h win rate of team_a vs team_b across historical matches.

    Returns NaN if there is no prior meeting (which is true for ~38% of WC
    group-stage matchups - that's why we have the *_missing flags).
    """
    mask = (((df["home_team"] == team_a) & (df["away_team"] == team_b)) |
            ((df["home_team"] == team_b) & (df["away_team"] == team_a))) & \
           (df["split"] != "predict")
    h2h = df[mask]
    if h2h.empty:
        return float("nan")
    a_wins = (((h2h["home_team"] == team_a) & (h2h["home_score"] > h2h["away_score"])).sum() +
              ((h2h["away_team"] == team_a) & (h2h["away_score"] > h2h["home_score"])).sum())
    return float(a_wins) / len(h2h)


def build_h2h_cache(teams: set[str], df: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Precompute h2h win rates for every pair of teams (both directions).

    Done once per Monte Carlo run rather than once per match — saves the cost
    of filtering the 15k-row historical dataframe on every simulated match.
    """
    historical = df[df["split"] != "predict"]
    cache: dict[tuple[str, str], float] = {}
    teams_list = sorted(teams)
    for i, a in enumerate(teams_list):
        for b in teams_list[i + 1:]:
            mask = (((historical["home_team"] == a) & (historical["away_team"] == b)) |
                    ((historical["home_team"] == b) & (historical["away_team"] == a)))
            h2h = historical[mask]
            if h2h.empty:
                cache[(a, b)] = float("nan")
                cache[(b, a)] = float("nan")
                continue
            a_wins = (((h2h["home_team"] == a) & (h2h["home_score"] > h2h["away_score"])).sum() +
                      ((h2h["away_team"] == a) & (h2h["away_score"] > h2h["home_score"])).sum())
            rate_a = float(a_wins) / len(h2h)
            cache[(a, b)] = rate_a
            cache[(b, a)] = 1.0 - rate_a if rate_a > 0 else (0.0 if len(h2h) > 0 else float("nan"))
    return cache


def build_predict_row_cache(predict_df: pd.DataFrame, feature_cols: list[str]) -> dict[tuple[str, str], dict]:
    """Precompute a {(home, away): {col: val}} dict for all 72 group-stage rows.

    Replaces a filter-by-pair on every call with a dict lookup.
    """
    cache: dict[tuple[str, str], dict] = {}
    base_cols = [c for c in feature_cols if not c.endswith("_missing")]
    for _, row in predict_df.iterrows():
        key = (row["home_team"], row["away_team"])
        cache[key] = {c: row[c] for c in base_cols if c in row.index}
    return cache


# -----------------------------------------------------------------------------
# Shootout model (used only for knockout draws)
# -----------------------------------------------------------------------------
def build_shootout_stats(players_data_path: Path) -> dict[str, dict]:
    """Build per-team penalty stats from the 2025-26 fbref players_data file.

    For each WC 2026 team, returns:
        takers_conv:      mean conversion rate of top 5 PK takers by attempts
        n_taker_attempts: total PK attempts among those top 5
        gk_save_rate:     PK save rate of the team's main GK (most attempts faced)
        n_gk_attempts:    total PK attempts faced by that GK

    NaN means no data for that team in the file (typical for non-Big-5 squads).
    """
    df = pd.read_csv(players_data_path)
    df["fifa_code"] = df["Nation"].dropna().str.split(" ").str[1]
    df["team"] = df["fifa_code"].map(FIFA_CODE_TO_TEAM)
    df = df.dropna(subset=["team"])

    stats: dict[str, dict] = {}
    for team in FIFA_CODE_TO_TEAM.values():
        team_players = df[df["team"] == team]

        # Top 5 takers by attempts (so we identify "designated takers", not
        # someone who scored 1/1 in a single match).
        takers = team_players[team_players["PKatt"].fillna(0) > 0].copy()
        takers = takers.sort_values("PKatt", ascending=False).head(5)
        if not takers.empty and takers["PKatt"].sum() > 0:
            takers_conv = float(takers["PK"].sum() / takers["PKatt"].sum())
            n_taker_attempts = int(takers["PKatt"].sum())
        else:
            takers_conv = float("nan")
            n_taker_attempts = 0

        # Main GK: the keeper on this team who has faced the most penalties.
        gks = team_players[team_players["PKatt_stats_keeper"].fillna(0) > 0].copy()
        gks = gks.sort_values("PKatt_stats_keeper", ascending=False).head(1)
        if not gks.empty and gks["PKatt_stats_keeper"].sum() > 0:
            gk_save_rate = float(gks["PKsv"].sum() / gks["PKatt_stats_keeper"].sum())
            n_gk_attempts = int(gks["PKatt_stats_keeper"].sum())
        else:
            gk_save_rate = float("nan")
            n_gk_attempts = 0

        stats[team] = {
            "takers_conv": takers_conv,
            "n_taker_attempts": n_taker_attempts,
            "gk_save_rate": gk_save_rate,
            "n_gk_attempts": n_gk_attempts,
        }
    return stats


def _elo_shootout_prob(home_elo: float, away_elo: float) -> float:
    """Dampened Elo expected-score for shootouts. Real shootouts are noisier
    than open play, so we compress the Elo gap (200-pt gap -> ~61% home win,
    not the ~76% that the standard Elo formula gives).
    """
    return 1.0 / (1.0 + 10.0 ** ((away_elo - home_elo) / SHOOTOUT_ELO_DAMP))


def simulate_shootout(home: str, away: str, shootout_stats: dict,
                      home_elo: float, away_elo: float) -> int:
    """Decide a knockout draw by penalty shootout. Returns 0 (away) or 2 (home).

    Three-tier fallback:
      1. Both teams have >= SHOOTOUT_MIN_ATTEMPTS PKatt: use the player model
         (top-5 taker conversion vs opposing GK save rate).
      2. Only one team has data: blend SHOOTOUT_BLEND_DATA_WEIGHT * data + rest * Elo prior.
      3. Neither has data: pure Elo prior.
    """
    h = shootout_stats.get(home, {})
    a = shootout_stats.get(away, {})
    h_has = (h.get("n_taker_attempts", 0) >= SHOOTOUT_MIN_ATTEMPTS
             and pd.notna(h.get("takers_conv")))
    a_has = (a.get("n_taker_attempts", 0) >= SHOOTOUT_MIN_ATTEMPTS
             and pd.notna(a.get("takers_conv")))

    p_elo = _elo_shootout_prob(home_elo, away_elo)

    if h_has and a_has:
        h_taker = h["takers_conv"]
        a_taker = a["takers_conv"]
        # GK rates are shrunk toward the global mean - small per-season samples
        # otherwise produce wonky values like Spain-keepers-saved-0%-on-8-attempts.
        h_gk = _shrunk_gk_rate(h.get("gk_save_rate"), h.get("n_gk_attempts", 0))
        a_gk = _shrunk_gk_rate(a.get("gk_save_rate"), a.get("n_gk_attempts", 0))
        # Net advantage for home: better takers AND better keeper.
        advantage = (h_taker - a_taker) + (h_gk - a_gk)
        p_home = 1.0 / (1.0 + np.exp(-SHOOTOUT_K_DATA * advantage))
    elif h_has or a_has:
        # Tier 2 blend: lean on the data we have, soften with Elo prior.
        p_home = SHOOTOUT_BLEND_DATA_WEIGHT * p_elo + (1 - SHOOTOUT_BLEND_DATA_WEIGHT) * 0.5
    else:
        p_home = p_elo

    return 2 if np.random.random() < p_home else 0


# -----------------------------------------------------------------------------
# Per-match featurization
# -----------------------------------------------------------------------------
def featurize_match(
    home: str,
    away: str,
    snapshots: dict[str, dict],
    predict_row_cache: dict[tuple[str, str], dict],
    h2h_cache: dict[tuple[str, str], float],
    elo_tracker: EloRating,
    fill_values: dict,
    feature_cols: list[str],
    neutral: bool = True,
) -> pd.DataFrame:
    """Build a single-row feature DataFrame for home-vs-away.

    Uses precomputed caches (predict-row lookup + h2h lookup) so a Monte Carlo
    run doesn't pay a 15k-row pandas filter cost on every simulated match.
    """
    feats: dict = {}

    cached = predict_row_cache.get((home, away))
    if cached is not None:
        # Group-stage path: copy every model feature directly from the cached row
        feats.update(cached)
    else:
        # Knockout path: build from team snapshots
        h_snap = snapshots.get(home, {})
        a_snap = snapshots.get(away, {})
        for key, val in h_snap.items():
            feats[f"home_{key}"] = val
        for key, val in a_snap.items():
            feats[f"away_{key}"] = val
        feats["home_h2h_win_rate"] = h2h_cache.get((home, away), float("nan"))

    # Live Elo (always overrides whatever the cached row had)
    feats["home_elo"] = elo_tracker.get(home)
    feats["away_elo"] = elo_tracker.get(away)
    feats["elo_diff"] = feats["home_elo"] - feats["away_elo"]

    # Recompute diff features
    for diff_col, (h_col, a_col) in DIFF_FEATURES.items():
        h_val = feats.get(h_col)
        a_val = feats.get(a_col)
        if h_val is not None and a_val is not None and pd.notna(h_val) and pd.notna(a_val):
            feats[diff_col] = h_val - a_val

    # Neutral flag
    feats["neutral"] = 1.0 if neutral else 0.0

    # Build the row in the right column order; dict comprehension is faster than DataFrame from dict.
    row_dict = {c: feats.get(c, float("nan")) for c in feature_cols}

    # Missingness indicators (computed BEFORE imputation)
    for col in feature_cols:
        if col.endswith("_missing"):
            base_col = col[: -len("_missing")]
            row_dict[col] = 1.0 if pd.isna(row_dict[base_col]) else 0.0

    # Median imputation
    for col, val in row_dict.items():
        if pd.isna(val) and col in fill_values:
            row_dict[col] = fill_values[col]

    return pd.DataFrame([row_dict])


# -----------------------------------------------------------------------------
# Match simulation
# -----------------------------------------------------------------------------
def run_match_sim(
    home: str,
    away: str,
    snapshots: dict,
    predict_row_cache: dict,
    h2h_cache: dict,
    model,
    elo_tracker: EloRating,
    fill_values: dict,
    feature_cols: list[str],
    classes: list,
    scaler=None,
    neutral: bool = True,
    knockout: bool = False,
    shootout_stats: dict | None = None,
) -> tuple[int, int, int]:
    """Simulate one match. Returns (outcome, home_score, away_score).

    Outcome encoding matches the model: 0 = away win, 1 = draw, 2 = home win.
    For knockout matches with a sampled draw, the winner is decided by
    simulate_shootout() if shootout_stats was provided, else a 50/50 coin flip.
    Goal counts are placeholder values that respect the win/draw/loss outcome
    - the model only predicts class probabilities, not score lines.
    """
    match_row = featurize_match(home, away, snapshots, predict_row_cache, h2h_cache,
                                elo_tracker, fill_values, feature_cols, neutral=neutral)
    X = match_row[feature_cols]
    if scaler is not None:
        X = scaler.transform(X)

    probs = model.predict_proba(X)[0]
    outcome = int(np.random.choice(classes, p=probs))

    if knockout and outcome == 1:
        if shootout_stats is not None:
            outcome = simulate_shootout(
                home, away, shootout_stats,
                elo_tracker.get(home), elo_tracker.get(away),
            )
        else:
            outcome = int(np.random.choice([0, 2]))

    if outcome == 2:
        h_score, a_score = 1, 0
    elif outcome == 0:
        h_score, a_score = 0, 1
    else:
        h_score, a_score = 1, 1  # draw scored 1-1 for goal-difference math

    elo_tracker.update(home, away, h_score, a_score,
                       tournament="FIFA World Cup", neutral=neutral)
    return outcome, h_score, a_score


# -----------------------------------------------------------------------------
# Group stage
# -----------------------------------------------------------------------------
def simulate_group(
    group_letter: str,
    teams: list[str],
    snapshots: dict,
    predict_df: pd.DataFrame,
    predict_row_cache: dict,
    h2h_cache: dict,
    model,
    elo_tracker: EloRating,
    fill_values: dict,
    feature_cols: list[str],
    classes: list,
    scaler=None,
) -> list[dict]:
    """Simulate all 6 matches of a 4-team group.

    Returns a list of standings dicts in finishing order, with keys:
        team, group, position (1-4), pts, gd, gf, ga, w, d, l
    Sorted by FIFA tiebreakers: pts -> gd -> gf -> ga (last is a sensible
    fallback - the official rules also include head-to-head and fair-play
    points, which are not material at the resolution of our simulation).
    """
    standings = {t: {"pts": 0, "gf": 0, "ga": 0, "w": 0, "d": 0, "l": 0}
                 for t in teams}

    # Use the actual fixtures (with their proper home/away/neutral) from features.csv
    fixtures = predict_df[
        (predict_df["home_team"].isin(teams)) & (predict_df["away_team"].isin(teams))
    ]

    for _, fix in fixtures.iterrows():
        h, a = fix["home_team"], fix["away_team"]
        neutral = bool(fix["neutral"])
        _, h_score, a_score = run_match_sim(
            h, a, snapshots, predict_row_cache, h2h_cache, model, elo_tracker,
            fill_values, feature_cols, classes, scaler=scaler,
            neutral=neutral, knockout=False,
        )
        standings[h]["gf"] += h_score
        standings[h]["ga"] += a_score
        standings[a]["gf"] += a_score
        standings[a]["ga"] += h_score
        if h_score > a_score:
            standings[h]["pts"] += 3; standings[h]["w"] += 1; standings[a]["l"] += 1
        elif h_score < a_score:
            standings[a]["pts"] += 3; standings[a]["w"] += 1; standings[h]["l"] += 1
        else:
            standings[h]["pts"] += 1; standings[a]["pts"] += 1
            standings[h]["d"] += 1; standings[a]["d"] += 1

    # Sort by FIFA tiebreakers
    sorted_teams = sorted(
        standings.items(),
        key=lambda kv: (-kv[1]["pts"], -(kv[1]["gf"] - kv[1]["ga"]), -kv[1]["gf"], kv[1]["ga"]),
    )
    return [
        {"team": t, "group": group_letter, "position": i + 1,
         "pts": s["pts"], "gd": s["gf"] - s["ga"], "gf": s["gf"], "ga": s["ga"],
         "w": s["w"], "d": s["d"], "l": s["l"]}
        for i, (t, s) in enumerate(sorted_teams)
    ]


def select_qualifiers(all_standings: list[list[dict]]) -> list[dict]:
    """Return the 32 advancing teams in seed order.

    Format: 12 group winners (best by pts/gd/gf), then 12 runners-up,
    then the 8 best 3rd-place teams (chosen across groups by pts/gd/gf).
    Runners-up are deliberately *not* sorted globally - their position in
    the seeding pool depends on which group they came from, but the bracket
    pairing function below uses overall rank anyway, so the order within a
    cohort matters less than the cohort itself.
    """
    winners = [s[0] for s in all_standings]
    runners_up = [s[1] for s in all_standings]
    thirds = [s[2] for s in all_standings]

    def by_perf(team_record):
        return (-team_record["pts"], -team_record["gd"], -team_record["gf"], team_record["ga"])

    winners.sort(key=by_perf)
    runners_up.sort(key=by_perf)
    thirds.sort(key=by_perf)
    best_thirds = thirds[:8]
    return winners + runners_up + best_thirds  # 12 + 12 + 8 = 32


# -----------------------------------------------------------------------------
# Knockout bracket
# -----------------------------------------------------------------------------
def seed_bracket(qualifiers: list[dict]) -> list[str]:
    """Arrange the 32 qualifying teams into standard seeded bracket order.

    `qualifiers[0]` is the strongest seed (top group winner), `qualifiers[31]`
    is the weakest (last 3rd-place qualifier). Returns a 32-team list where
    adjacent pairs are R32 matchups and the bracket is balanced - top seeds
    only meet in late rounds if they all keep winning.
    """
    return [qualifiers[s - 1]["team"] for s in BRACKET_ORDER_32]


def simulate_knockout(
    bracket_teams: list[str],
    snapshots: dict,
    predict_row_cache: dict,
    h2h_cache: dict,
    model,
    elo_tracker: EloRating,
    fill_values: dict,
    feature_cols: list[str],
    classes: list,
    scaler=None,
    verbose: bool = False,
    shootout_stats: dict | None = None,
) -> tuple[str, dict]:
    """Run a 32-team single-elim from R32 to final.

    Returns (champion, deepest_round_per_team) where deepest_round_per_team
    maps team -> "R32" / "R16" / "QF" / "SF" / "Final" / "Champion".
    """
    deepest: dict[str, str] = {t: "R32" for t in bracket_teams}
    round_names = ["R16", "QF", "SF", "Final", "Champion"]
    current = list(bracket_teams)
    round_idx = 0

    while len(current) > 1:
        next_round: list[str] = []
        for i in range(0, len(current), 2):
            home, away = current[i], current[i + 1]
            outcome, _, _ = run_match_sim(
                home, away, snapshots, predict_row_cache, h2h_cache, model, elo_tracker,
                fill_values, feature_cols, classes, scaler=scaler,
                neutral=True, knockout=True, shootout_stats=shootout_stats,
            )
            winner = home if outcome == 2 else away
            next_round.append(winner)
            if verbose:
                print(f"    {home:>30} vs {away:<30} -> {winner}")
        # Mark how far each *advancing* team has now reached
        for t in next_round:
            deepest[t] = round_names[round_idx]
        current = next_round
        round_idx += 1

    return current[0], deepest


# -----------------------------------------------------------------------------
# Whole-tournament simulation
# -----------------------------------------------------------------------------
def simulate_tournament(
    snapshots: dict,
    predict_df: pd.DataFrame,
    predict_row_cache: dict,
    h2h_cache: dict,
    model,
    fill_values: dict,
    feature_cols: list[str],
    classes: list,
    baseline_elos: dict[str, float],
    scaler=None,
    verbose: bool = False,
    shootout_stats: dict | None = None,
) -> dict:
    """Run one full tournament: group stage then knockout.

    Returns a dict with:
        champion: str
        runner_up: str
        deepest_round: dict[team -> round_name]
        group_finishes: dict[team -> position (1-4) or 'eliminated']
    """
    elo_tracker = EloRating(ratings=baseline_elos.copy())

    all_standings = []
    group_finishes: dict[str, str] = {}
    for letter, teams in GROUPS_2026.items():
        group_table = simulate_group(
            letter, teams, snapshots, predict_df, predict_row_cache, h2h_cache,
            model, elo_tracker, fill_values, feature_cols, classes, scaler=scaler,
        )
        all_standings.append(group_table)
        for entry in group_table:
            group_finishes[entry["team"]] = entry["position"]
        if verbose:
            print(f"  Group {letter}: " + ", ".join(
                f"{e['team']}({e['pts']}pts,{e['gd']:+d}gd)" for e in group_table))

    qualifiers = select_qualifiers(all_standings)
    bracket = seed_bracket(qualifiers)
    champion, deepest = simulate_knockout(
        bracket, snapshots, predict_row_cache, h2h_cache, model, elo_tracker,
        fill_values, feature_cols, classes, scaler=scaler, verbose=verbose,
        shootout_stats=shootout_stats,
    )

    # Find runner-up: the team that lost the final
    final_round_teams = [t for t, r in deepest.items() if r == "Final"]
    runner_up = next((t for t in final_round_teams if t != champion), None)

    return {
        "champion": champion,
        "runner_up": runner_up,
        "deepest_round": deepest,
        "group_finishes": group_finishes,
    }


# -----------------------------------------------------------------------------
# Monte Carlo
# -----------------------------------------------------------------------------
def monte_carlo_simulation(
    df: pd.DataFrame,
    predict_df: pd.DataFrame,
    model,
    fill_values: dict,
    feature_cols: list[str],
    classes: list,
    scaler=None,
    iters: int = 1000,
    seed: int | None = None,
    use_shootouts: bool = True,
) -> pd.DataFrame:
    """Run `iters` independent tournaments. Returns a per-team summary table."""
    if seed is not None:
        np.random.seed(seed)

    snapshots = build_team_snapshots(predict_df)
    # Precompute these once. Each saves a 15k-row pandas filter on every match.
    all_teams = {t for g in GROUPS_2026.values() for t in g}
    print("Building caches...")
    h2h_cache = build_h2h_cache(all_teams, df)
    predict_row_cache = build_predict_row_cache(predict_df, feature_cols)

    # Pre-tournament Elo for each team comes from the home_elo / away_elo of
    # their first predict-split match. features.py builds this by iterating
    # through all of history chronologically, so it already includes the Elo
    # update from the team's most recent pre-WC result.
    baseline_elos: dict[str, float] = {}
    for team in all_teams:
        h = predict_df[predict_df["home_team"] == team].head(1)
        a = predict_df[predict_df["away_team"] == team].head(1)
        if not h.empty:
            baseline_elos[team] = float(h["home_elo"].iloc[0])
        elif not a.empty:
            baseline_elos[team] = float(a["away_elo"].iloc[0])
        else:
            baseline_elos[team] = 1500.0
            print(f"  warn: no predict-split data for {team}, using default Elo 1500")

    # Build the shootout stats once if enabled. Pretty-print coverage so the
    # caller can see how many of the 48 teams have meaningful penalty data.
    shootout_stats = None
    if use_shootouts:
        shootout_path = ROOT / "data" / "raw" / "players_data-2025_2026.csv"
        if shootout_path.exists():
            shootout_stats = build_shootout_stats(shootout_path)
            with_data = sum(1 for s in shootout_stats.values()
                            if s["n_taker_attempts"] >= SHOOTOUT_MIN_ATTEMPTS)
            print(f"Shootout model loaded: {with_data}/{len(FIFA_CODE_TO_TEAM)} teams "
                  f"have >= {SHOOTOUT_MIN_ATTEMPTS} taker attempts; "
                  f"others fall back to Elo prior.")
        else:
            print(f"  warn: {shootout_path} not found, knockout draws -> coin flip")

    print(f"Running Monte Carlo: {iters} iterations...")
    counts: dict[str, dict[str, int]] = defaultdict(lambda: Counter())
    champ_counts: Counter = Counter()
    for i in range(iters):
        result = simulate_tournament(
            snapshots, predict_df, predict_row_cache, h2h_cache, model,
            fill_values, feature_cols, classes, baseline_elos,
            scaler=scaler, verbose=False, shootout_stats=shootout_stats,
        )
        champ_counts[result["champion"]] += 1
        for team, round_name in result["deepest_round"].items():
            counts[team][round_name] += 1
        if iters >= 10 and (i + 1) % max(1, iters // 10) == 0:
            print(f"  progress {(i + 1) / iters * 100:.0f}%")

    # Build per-team summary. Each team has exactly one "deepest round" per
    # iteration, so the cumulative percentages just sum the relevant buckets.
    rows = []
    for team in sorted({t for g in GROUPS_2026.values() for t in g}):
        c = counts[team]
        rows.append({
            "team": team,
            "champion %":   100 * c["Champion"] / iters,
            "final %":      100 * (c["Final"] + c["Champion"]) / iters,
            "semis %":      100 * (c["SF"] + c["Final"] + c["Champion"]) / iters,
            "qfs %":        100 * (c["QF"] + c["SF"] + c["Final"] + c["Champion"]) / iters,
            "r16 %":        100 * (c["R16"] + c["QF"] + c["SF"] + c["Final"] + c["Champion"]) / iters,
        })
    return pd.DataFrame(rows).sort_values("champion %", ascending=False).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Monte Carlo simulate the 2026 World Cup")
    parser.add_argument("--model", default="ensemble",
                        help="Model name in models/ (ensemble, lr, rf, xgb, lr_raw)")
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default=None,
                        help="Optional path to save the per-team summary as CSV")
    parser.add_argument("--no-shootouts", action="store_true",
                        help="Skip the player-level shootout model; use coin flips on knockout draws")
    args = parser.parse_args()

    print(f"Loading model={args.model} ...")
    model, fill_values, feature_names, classes, scaler = load_artifacts(args.model)
    df = pd.read_csv(DATA_PATH)
    predict_df = df[df["split"] == "predict"].copy()

    print(f"Predict split: {len(predict_df)} fixtures across "
          f"{len({t for g in GROUPS_2026.values() for t in g})} teams in 12 groups\n")

    summary = monte_carlo_simulation(df, predict_df, model, fill_values, feature_names, classes,
                          scaler=scaler, iters=args.iters, seed=args.seed,
                          use_shootouts=not args.no_shootouts)

    print("\n--- Tournament summary (top 16 by champion %) ---")
    print(summary.head(16).to_string(index=False, float_format=lambda x: f"{x:6.2f}"))
    if args.output:
        summary.to_csv(args.output, index=False)
        print(f"\nSaved full summary to {args.output}")


if __name__ == "__main__":
    main()

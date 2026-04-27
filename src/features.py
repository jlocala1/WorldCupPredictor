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
    feature_cols = [c for c in df.columns if c.startswith(("home_form_", "away_form_", "home_h2h_", "home_elo", "away_elo", "elo_"))]
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

    print("\nSample predict-split rows (Brazil's matches):")
    sample = df[(df["split"] == "predict") & ((df["home_team"] == "Brazil") | (df["away_team"] == "Brazil"))]
    for _, r in sample.iterrows():
        h2h_str = f"{r['home_h2h_win_rate']:.2f}" if pd.notna(r["home_h2h_win_rate"]) else "first mtg"
        print(f"  {r['date'].date()}  {r['home_team']:<25} vs {r['away_team']:<25} "
              f"elos={r['home_elo']:.0f}/{r['away_elo']:.0f}  diff={r['elo_diff']:+.0f}  h2h={h2h_str}")

    out = DATA_PROCESSED / "features.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out} ({len(df):,} rows, {len(df.columns)} columns)")


if __name__ == "__main__":
    main()

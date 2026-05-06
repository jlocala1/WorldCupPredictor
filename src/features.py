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

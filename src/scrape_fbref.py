"""Scrape fbref player season stats via soccerdata library.

soccerdata uses an undetected Chrome browser under the hood which gets past
fbref's Cloudflare blocking that defeats plain `requests`/`cloudscraper`. It
also caches each (league, season, stat_type) fetch to disk, so re-runs are
near-instant.

Scope:
  - Big 5 European Leagues Combined: 2017-18 through 2024-25 (8 seasons; xG and
    advanced stats only available from 2017-18 onward via fbref's StatsBomb feed).
  - MLS, Saudi Pro League, Liga MX, Brazilian Serie A: most recent season only.
    Players in these leagues are mostly stable year-to-year, so a current
    snapshot is acceptable. Historical advanced stats for these competitions
    aren't reliably available on fbref anyway.

Per (league, season), pulls 5 stat tables and merges:
  - standard:    MP, Min, 90s, Gls, Ast (we already have these)
  - shooting:    xG, npxG, Sh, SoT
  - passing:     xA, PrgP, KP
  - defense:     Tkl (total), Int, Blocks
  - possession:  PrgC

Output: data/raw/fbref_player_stats.csv with one row per (player, league, season)
and all advanced columns merged in.

Usage: python src/scrape_fbref.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import soccerdata as sd

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "fbref_player_stats.csv"

# Big 5 combined view: works from 2017-18 onwards (xG and advanced metrics start there)
BIG5_SEASONS = ["2017-2018", "2018-2019", "2019-2020", "2020-2021",
                "2021-2022", "2022-2023", "2023-2024", "2024-2025"]

# Non-Big-5 leagues are NOT supported by soccerdata (it only knows Big 5 + INT-* +
# ENG/ESP/FRA/GER/ITA individual leagues). MLS / Liga MX / Saudi / Brazil players
# will simply have NaN position-z-scores — documented limitation.
NON_BIG5: list[tuple[str, list[str]]] = []

# Stat types soccerdata exposes for FBref: standard, keeper, shooting, playing_time, misc.
# It does NOT expose passing/defense/possession, so we lose xA, PrgP, KP, total Tkl,
# Blocks, PrgC. Documented as known limitation; xG and npxG (in shooting) are the most
# important advanced metrics we still capture.
STAT_TYPES = ["standard", "shooting", "misc"]


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """fbref tables come back with a multi-level column index. Flatten to single
    underscore-joined names so we can pick what we want by string match."""
    if df.columns.nlevels > 1:
        df.columns = ["_".join(str(c) for c in col if c and "Unnamed" not in str(c)).strip("_")
                      for col in df.columns.values]
    return df


def _scrape_competition(league: str, seasons) -> pd.DataFrame:
    """Pull all 5 stat tables for one league across given seasons; merge."""
    print(f"\n=== {league} (seasons: {seasons}) ===")
    fbref = sd.FBref(leagues=league, seasons=seasons)
    parts: dict[str, pd.DataFrame] = {}

    for stat in STAT_TYPES:
        try:
            print(f"  fetching {stat}...", flush=True)
            df = fbref.read_player_season_stats(stat_type=stat)
            df = _flatten_columns(df.reset_index())
            parts[stat] = df
            print(f"    -> {df.shape[0]:,} rows × {df.shape[1]} cols")
        except Exception as e:
            print(f"    FAILED: {e}")
            parts[stat] = None

    # Merge tables on (league, season, team, player)
    base = parts.get("standard")
    if base is None or base.empty:
        print(f"  no standard stats; skipping league")
        return pd.DataFrame()

    join_keys = [c for c in ["league", "season", "team", "player"] if c in base.columns]
    merged = base.copy()
    for stat in STAT_TYPES[1:]:  # everything after `standard`
        df = parts.get(stat)
        if df is None or df.empty:
            continue
        new_cols = [c for c in df.columns if c not in merged.columns or c in join_keys]
        merged = merged.merge(df[new_cols], on=join_keys, how="left", suffixes=("", f"_{stat}"))
    return merged


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_dfs: list[pd.DataFrame] = []

    # Big 5 combined, historical
    big5 = _scrape_competition("Big 5 European Leagues Combined", BIG5_SEASONS)
    if not big5.empty:
        all_dfs.append(big5)

    # Non-Big-5 leagues, current season(s)
    for league, seasons in NON_BIG5:
        if not seasons:
            continue
        df = _scrape_competition(league, seasons)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        print("\nNo data scraped.")
        return 1

    final = pd.concat(all_dfs, ignore_index=True)
    final.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(final):,} rows × {len(final.columns)} cols -> {OUTPUT_PATH}")

    # Sanity: print which advanced columns we got
    advanced_cols = [c for c in final.columns
                     if any(k in c for k in ["xG", "npxG", "xA", "Prg", "Tkl", "Int", "Blocks"])]
    print(f"\nAdvanced-metric columns present ({len(advanced_cols)}):")
    for c in sorted(advanced_cols):
        n_nonnull = final[c].notna().sum()
        print(f"  {c:<30}  {n_nonnull:>6,} non-null rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())

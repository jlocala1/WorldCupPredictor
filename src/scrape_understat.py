"""Scrape player season stats from Understat across Big 5 + Russian Premier League.

Understat's public site recently moved away from the well-documented
``var playersData = JSON.parse('...')`` pattern, so plain HTTP scraping and
older libraries (`understat`, `understatapi`, `ScraperFC`) are broken. The
data is still served, though, via an internal endpoint:

    GET https://understat.com/getLeagueData/{LEAGUE}/{YEAR}
        with header X-Requested-With: XMLHttpRequest

The endpoint returns ~500KB of JSON keyed `teams`, `players`, `dates`. We use
the `players` array, which contains per-player season totals including
**id, player_name, team, position, games, time (mins), goals, xG, npxG,
assists, xA, key_passes, shots, xGChain, xGBuildup, npg, yellow/red cards**.

The endpoint requires Cloudflare-cleared cookies, so we go via an undetected
Chrome session (uc_driver, installed by soccerdata earlier) and call fetch()
from within the page so cookies/CSRF are satisfied automatically.

Coverage:
  - Leagues: EPL (Premier League), La_liga, Bundesliga, Serie_A, Ligue_1, RPL
  - Years: 2014-2024 (start year of season — 2014 = 2014/15, 2024 = 2024/25)
  - 6 leagues x 11 seasons = 66 API calls. Each ~1s after Cloudflare warm-up.

Output: data/raw/understat_player_stats.csv
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
from seleniumbase import Driver

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "understat_player_stats.csv"

# Understat's internal league codes
LEAGUES = ["EPL", "La_liga", "Bundesliga", "Serie_A", "Ligue_1", "RPL"]

# Year here is the START year of the season (Understat's URL convention)
YEARS = list(range(2014, 2025))  # 2014/15 through 2024/25

# Columns to keep from each player row
PLAYER_COLS = [
    "id", "player_name", "team_title", "position",
    "games", "time", "goals", "xG", "npxG", "assists", "xA",
    "shots", "key_passes", "xGChain", "xGBuildup",
    "npg", "yellow_cards", "red_cards",
]


def fetch_league_year(driver, league: str, year: int) -> pd.DataFrame:
    """Use the Cloudflare-warmed browser to call Understat's getLeagueData endpoint."""
    js = f"""
        const callback = arguments[arguments.length - 1];
        fetch('/getLeagueData/{league}/{year}', {{headers: {{'X-Requested-With': 'XMLHttpRequest'}}}})
          .then(r => r.text()).then(t => callback(t))
          .catch(e => callback(JSON.stringify({{__error__: String(e)}})));
    """
    raw = driver.execute_async_script(js)
    if not raw or "__error__" in raw[:50]:
        raise RuntimeError(f"fetch error for {league}/{year}: {raw[:200]}")
    data = json.loads(raw)
    players = data.get("players", [])
    if not players:
        return pd.DataFrame()
    df = pd.DataFrame(players)
    keep = [c for c in PLAYER_COLS if c in df.columns]
    df = df[keep].copy()
    df["league"] = league
    df["year"] = year
    return df


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Starting undetected Chrome session...")
    driver = Driver(uc=True, headless=True)

    # Warm up Cloudflare cookies via any league page
    print("Warming Cloudflare via league page...")
    driver.uc_open_with_reconnect("https://understat.com/league/EPL/2024", 6)
    time.sleep(5)

    all_rows: list[pd.DataFrame] = []
    failures: list[tuple[str, int, str]] = []

    try:
        for league in LEAGUES:
            for year in YEARS:
                print(f"  fetching {league}/{year}...", end="", flush=True)
                try:
                    df = fetch_league_year(driver, league, year)
                    if df.empty:
                        print(" (empty)")
                        continue
                    all_rows.append(df)
                    print(f" {len(df):,} players")
                except Exception as e:
                    print(f" FAILED: {e}")
                    failures.append((league, year, str(e)))
                time.sleep(0.5)  # gentle on the host
    finally:
        driver.quit()

    if not all_rows:
        print("\nNo data scraped.")
        return 1

    full = pd.concat(all_rows, ignore_index=True)
    # Cast numeric columns
    numeric_cols = ["games", "time", "goals", "xG", "npxG", "assists", "xA",
                    "shots", "key_passes", "xGChain", "xGBuildup", "npg",
                    "yellow_cards", "red_cards"]
    for c in numeric_cols:
        if c in full.columns:
            full[c] = pd.to_numeric(full[c], errors="coerce")

    full.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(full):,} rows -> {OUTPUT_PATH}")
    print(f"Coverage: {full.groupby(['league','year']).size().unstack(fill_value=0)}")
    if failures:
        print(f"\n{len(failures)} failures:")
        for lg, yr, err in failures:
            print(f"  {lg}/{yr}: {err}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

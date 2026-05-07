"""Scrape player season stats from fotmob across major leagues.

Why fotmob: it has full advanced metrics (xG, xA, big chances, tackles,
interceptions, blocks, recoveries) for far more leagues than Understat —
notably MLS, Saudi Pro, Liga MX, Brasileirão, Eredivisie, Liga Portugal,
Belgian Pro — which is essential for 2026 WC coverage where many star
players are outside Big 5 (Saudi: Ronaldo/Mané/Brozović; MLS:
Messi/Suárez; etc).

How it works:
  1. GET https://www.fotmob.com/api/data/leagues?id={LEAGUE_ID} returns
     league metadata including stats.players[*].fetchAllUrl pointing to
     per-stat JSON dumps on data.fotmob.com.
  2. Each fetchAllUrl returns a TopLists payload with one row per player
     (ParticiantId, ParticipantName, TeamName, StatValue, MinutesPlayed,
     MatchesPlayed, ParticipantCountryCode, Positions).
  3. We hit only the stats we need (goals, assists, xG, xA, big chances,
     tackles, interceptions, blocks, recoveries) and join by player id.

Both endpoints respond to plain `requests` with a Referer header — no
Cloudflare/Selenium needed for the data CDN, despite www.fotmob.com using
Turnstile for browsers.

Output: data/raw/fotmob_player_stats.csv (one row per
(player, league, season) with all wanted stats joined).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "fotmob_player_stats.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Referer": "https://www.fotmob.com/",
    "Accept": "application/json",
}

LEAGUES = [
    # (display name, fotmob id) — current-season scrape only (latest snapshot per player)
    ("EPL", 47),
    ("La Liga", 87),
    ("Bundesliga", 54),
    ("Serie A", 55),
    ("Ligue 1", 53),
    ("MLS", 130),
    ("Saudi Pro", 536),
    ("Liga MX", 230),
    ("Eredivisie", 57),
    ("Liga Portugal", 61),
    ("Brasileirão", 268),
    ("Belgian Pro", 40),
]

# fotmob's internal stat names → our column names. Mix of totals and per-90
# (we'll compute consistent per-90 in features.py from MinutesPlayed).
STATS = {
    "goals":               "goals_total",
    "goal_assist":         "assists_total",
    "expected_goals":      "xG_total",
    "expected_assists":    "xA_total",
    "big_chance_created":  "big_chance_total",
    "total_tackle":        "tackles_per90",     # already per 90
    "interception":        "int_per90",         # already per 90
    "outfielder_block":    "blocks_per90",      # already per 90
    "ball_recovery":       "recoveries_per90",  # already per 90
}


def fetch_league_metadata(league_id: int) -> dict[str, str]:
    """Return {stat_name -> fetchAllUrl} for a league's CURRENT season."""
    r = requests.get(
        f"https://www.fotmob.com/api/data/leagues?id={league_id}",
        headers=HEADERS, timeout=20,
    )
    r.raise_for_status()
    data = r.json()
    return {
        cat["name"]: cat["fetchAllUrl"]
        for cat in data.get("stats", {}).get("players", [])
        if "fetchAllUrl" in cat
    }


def fetch_stat(url: str) -> list[dict]:
    """Return list of player records for one (league, season, stat) combo."""
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()
    top_lists = data.get("TopLists", [])
    if not top_lists:
        return []
    return top_lists[0].get("StatList", [])


def scrape_league(name: str, league_id: int) -> pd.DataFrame:
    """Scrape all wanted stats for a league. Returns one row per player."""
    print(f"\n=== {name} (id={league_id}) ===")
    try:
        stat_urls = fetch_league_metadata(league_id)
    except Exception as e:
        print(f"  metadata fetch failed: {e}")
        return pd.DataFrame()

    players: dict[int, dict] = {}
    for stat_name, our_col in STATS.items():
        url = stat_urls.get(stat_name)
        if not url:
            print(f"  no URL for {stat_name}")
            continue
        try:
            stat_list = fetch_stat(url)
            for item in stat_list:
                pid = item.get("ParticiantId")
                if pid is None:
                    continue
                if pid not in players:
                    players[pid] = {
                        "player_id":   pid,
                        "player_name": item.get("ParticipantName"),
                        "team_id":     item.get("TeamId"),
                        "team_name":   item.get("TeamName"),
                        "nation":      item.get("ParticipantCountryCode"),
                        "positions":   ",".join(str(p) for p in item.get("Positions", [])),
                        "minutes":     item.get("MinutesPlayed"),
                        "matches":     item.get("MatchesPlayed"),
                    }
                else:
                    # Update minutes/matches in case earlier stat had stale numbers
                    if item.get("MinutesPlayed"):
                        players[pid]["minutes"] = item["MinutesPlayed"]
                    if item.get("MatchesPlayed"):
                        players[pid]["matches"] = item["MatchesPlayed"]
                players[pid][our_col] = item.get("StatValue")
            print(f"  {stat_name:<22} -> {len(stat_list):>4} players")
        except Exception as e:
            print(f"  {stat_name}: FAILED {e}")
        time.sleep(0.3)  # gentle on the CDN

    df = pd.DataFrame(list(players.values()))
    df["league"] = name
    return df


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_dfs: list[pd.DataFrame] = []
    for name, lid in LEAGUES:
        df = scrape_league(name, lid)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        print("\nNo data scraped.")
        return 1

    full = pd.concat(all_dfs, ignore_index=True)
    full.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(full):,} rows → {OUTPUT_PATH}")
    print(f"\nCoverage by league:")
    print(full.groupby("league").size())

    advanced_cols = [c for c in full.columns
                     if any(k in c for k in ("xG", "xA", "tackles", "int", "blocks"))]
    print(f"\nAdvanced columns: {advanced_cols}")
    for c in advanced_cols:
        print(f"  {c}: {full[c].notna().sum():>5,} non-null")
    return 0


if __name__ == "__main__":
    sys.exit(main())

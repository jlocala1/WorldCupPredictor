"""Scrape per-league per-season top-scorer/assist stats from Transfermarkt.

Why: our position z-score cascade needs date-correct basic stats (Goals,
Assists) for matches and players that Understat doesn't cover —
specifically pre-2014 Big 5 matches and ALL non-Big-5 league players.
Transfermarkt has these going back ~20 years per league.

Endpoint: /league-slug/scorerliste/wettbewerb/{CODE}/saison_id/{YEAR}
Returns the league's top combined-scorer-and-assist players for that season.
First page = top ~25 players. Player names link to TM profile URLs from
which we extract `player_id` — same IDs as `tournament_squads.csv` and
`player_valuations.csv` so joins are exact.

Output: data/raw/transfermarkt_player_seasons.csv with columns:
  league, season, player_id, player_name, club, nation, age,
  apps, goals, assists, points

Each row = one player's totals for that league-season.
"""

from __future__ import annotations

import csv
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "transfermarkt_player_seasons.csv"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}
BASE_URL = "https://www.transfermarkt.com"
DELAY_SECONDS = 3
TIMEOUT_SECONDS = 30

# (league_label, transfermarkt_code, slug-for-url)
LEAGUES = [
    ("Premier League",  "GB1",  "premier-league"),
    ("La Liga",         "ES1",  "laliga"),
    ("Serie A",         "IT1",  "serie-a"),
    ("Bundesliga",      "L1",   "bundesliga"),
    ("Ligue 1",         "FR1",  "ligue-1"),
    ("MLS",             "MLS1", "major-league-soccer"),
    ("Saudi Pro",       "SA1",  "saudi-professional-league"),
    ("Liga MX",         "MEX1", "liga-mx-clausura"),
    ("Eredivisie",      "NL1",  "eredivisie"),
    ("Liga Portugal",   "PO1",  "liga-portugal"),
    ("Brasileirão",     "BRA1", "campeonato-brasileiro-serie-a"),
    ("Belgian Pro",     "BE1",  "jupiler-pro-league"),
]

YEARS = list(range(2008, 2025))  # 2008-09 through 2024-25 (start year)

CSV_FIELDS = [
    "league", "league_code", "season", "player_id", "player_name", "club",
    "nation", "age", "apps", "goals", "assists", "points",
]


def parse_int(text: str) -> int | None:
    """Parse an integer from a TM cell. '-' or '' becomes None."""
    if not text:
        return None
    t = text.strip().replace(",", "")
    if t in {"-", "", "—"}:
        return None
    try:
        return int(t)
    except ValueError:
        return None


def fetch_league_season(session: requests.Session, slug: str, code: str, year: int) -> list[dict]:
    """Hit page 1 of scorerliste; return parsed rows. Top ~25 players per call."""
    url = f"{BASE_URL}/{slug}/scorerliste/wettbewerb/{code}/saison_id/{year}"
    response = session.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    rows: list[dict] = []
    # The relevant table has class "items" — typical TM convention
    table = soup.select_one("table.items")
    if not table:
        return rows

    # Each "logical" player row spans 3 <tr>s. The first tr has the rank,
    # name link, club, nation, age, and per-stat numeric cells.
    href_pattern = re.compile(r"/profil/spieler/(\d+)")
    for tr in table.select("tbody > tr"):
        cells = tr.find_all("td", recursive=False)
        if len(cells) < 8:
            continue
        # First cell: rank number
        rank_text = cells[0].get_text(strip=True)
        if not rank_text or not rank_text.replace(".", "").isdigit():
            continue

        # Player link
        anchor = tr.select_one("a[href*='/profil/spieler/']")
        if not anchor:
            continue
        match = href_pattern.search(anchor.get("href") or "")
        if not match:
            continue
        player_id = int(match.group(1))
        player_name = anchor.get("title") or anchor.get_text(strip=True)

        # Position (sometimes shown under the name); we don't need it but skip cleanly
        # Club: usually has an inner <a> with the club name
        club_anchor = cells[2].find("a") if len(cells) > 2 else None
        club = club_anchor.get_text(strip=True) if club_anchor else cells[2].get_text(" ", strip=True) if len(cells) > 2 else None

        # Nation: <img> with title
        nat_img = cells[3].find("img") if len(cells) > 3 else None
        nation = nat_img.get("title") if nat_img else None

        # Age, Apps, Goals, Assists, Points cells (positions vary slightly by layout
        # but typically: cells[4]=age, cells[5]=apps, cells[6]=goals, cells[7]=assists, cells[8]=points)
        age = parse_int(cells[4].get_text(strip=True)) if len(cells) > 4 else None
        apps = parse_int(cells[5].get_text(strip=True)) if len(cells) > 5 else None
        goals = parse_int(cells[6].get_text(strip=True)) if len(cells) > 6 else None
        assists = parse_int(cells[7].get_text(strip=True)) if len(cells) > 7 else None
        points = parse_int(cells[8].get_text(strip=True)) if len(cells) > 8 else None

        rows.append({
            "player_id": player_id, "player_name": player_name,
            "club": club, "nation": nation, "age": age,
            "apps": apps, "goals": goals, "assists": assists, "points": points,
        })
    return rows


def _load_completed() -> set[tuple[str, int]]:
    """Set of (league_code, season) pairs already in the output CSV (for resumability)."""
    if not OUTPUT_PATH.exists():
        return set()
    done: set[tuple[str, int]] = set()
    with OUTPUT_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add((row["league_code"], int(row["season"])))
    return done


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    completed = _load_completed()
    if completed:
        print(f"Resuming: {len(completed)} (league, season) pairs already in CSV.")

    new_file = not OUTPUT_PATH.exists()
    fh = OUTPUT_PATH.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    if new_file:
        writer.writeheader()

    session = requests.Session()
    failures: list[tuple[str, int, str]] = []
    rows_written = 0

    try:
        for label, code, slug in LEAGUES:
            for year in YEARS:
                if (code, year) in completed:
                    print(f"  {label} {year}: cached, skipping")
                    continue
                print(f"  {label} {year} ...", end="", flush=True)
                try:
                    players = fetch_league_season(session, slug, code, year)
                except requests.HTTPError as e:
                    code_status = e.response.status_code if e.response else "?"
                    print(f" HTTP {code_status}")
                    failures.append((code, year, f"HTTP {code_status}"))
                    time.sleep(DELAY_SECONDS)
                    continue
                except Exception as e:
                    print(f" parse error: {e}")
                    failures.append((code, year, f"parse: {e}"))
                    time.sleep(DELAY_SECONDS)
                    continue

                for p in players:
                    writer.writerow({
                        "league": label, "league_code": code, "season": year,
                        **p,
                    })
                fh.flush()
                rows_written += len(players)
                print(f" {len(players)} players")
                time.sleep(DELAY_SECONDS)
    finally:
        fh.close()

    print(f"\nWrote {rows_written:,} new player-season rows total.")
    print(f"Output: {OUTPUT_PATH}")
    if failures:
        print(f"\n{len(failures)} failures:")
        for code, year, reason in failures[:20]:
            print(f"  {code} {year}: {reason}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

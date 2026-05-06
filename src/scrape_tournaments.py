"""Scrape Transfermarkt tournament squad pages so features.py can build
date-correct national-team cohorts (per CLAUDE.md's original spec).

The flow per tournament:
  1. Fetch the participants page
       /x/teilnehmer/pokalwettbewerb/{CODE}/saison_id/{SAISON}
     -> list of participating teams + their Transfermarkt team_ids.
  2. For each team, fetch its tournament-specific squad page
       /{team-slug}/kader/verein/{team_id}/saison_id/{SAISON}/plus/1
     -> player rows with Transfermarkt player_ids that link directly to
        data/raw/player_valuations.csv (same ID system).
  3. Append rows to data/raw/tournament_squads.csv as we go (resumable).

Usage:
  python src/scrape_tournaments.py            # scrape all configured tournaments
  python src/scrape_tournaments.py --limit 1  # smoke test on the first tournament
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}
BASE_URL = "https://www.transfermarkt.com"
DELAY_SECONDS = 3
TIMEOUT_SECONDS = 30

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "tournament_squads.csv"
CSV_FIELDS = [
    "tournament_id",        # e.g. WC2022
    "tournament_name",      # human-readable
    "tournament_code",      # transfermarkt code (FIWC, EURO, etc.)
    "tournament_saison_id", # saison_id we hit
    "tournament_date",      # ISO date the tournament started (for cohort window)
    "team_id",              # transfermarkt national team id (matches national_teams.csv)
    "team_name",            # transfermarkt's team label
    "player_id",            # transfermarkt player id (matches player_valuations.csv)
    "player_name",          # transfermarkt's player label
]


@dataclass
class Tournament:
    """One tournament instance to scrape."""
    tournament_id: str        # WC2022
    name: str                 # FIFA World Cup 2022
    code: str                 # FIWC, EURO, COPA, AFCN, AFAC
    saison_id: int            # value to send as saison_id; year before kickoff or year of, varies
    start_date: str           # tournament kickoff date (ISO)


# Tournament list across the 5 major continental tournaments that davidcariboo's
# competitions.csv tags as national_team_competition. saison_id values were
# determined by following Transfermarkt's own redirect chain and the season
# selector dropdown on each competition's main page — see scrape_tournaments
# probe history. Copa America 2016 (Centenario) is intentionally omitted because
# Transfermarkt didn't track it as a distinct season in their selector.
TOURNAMENTS: list[Tournament] = [
    # --- World Cup ---
    Tournament("WC2006",  "FIFA World Cup 2006",  "FIWC", 2005, "2006-06-09"),
    Tournament("WC2010",  "FIFA World Cup 2010",  "FIWC", 2009, "2010-06-11"),
    Tournament("WC2014",  "FIFA World Cup 2014",  "FIWC", 2013, "2014-06-12"),
    Tournament("WC2018",  "FIFA World Cup 2018",  "FIWC", 2017, "2018-06-14"),
    Tournament("WC2022",  "FIFA World Cup 2022",  "FIWC", 2021, "2022-11-20"),
    # --- UEFA Euro ---
    Tournament("EU2008",  "UEFA Euro 2008",       "EURO", 2007, "2008-06-07"),
    Tournament("EU2012",  "UEFA Euro 2012",       "EURO", 2011, "2012-06-08"),
    Tournament("EU2016",  "UEFA Euro 2016",       "EURO", 2015, "2016-06-10"),
    Tournament("EU2020",  "UEFA Euro 2020",       "EURO", 2020, "2021-06-11"),
    Tournament("EU2024",  "UEFA Euro 2024",       "EURO", 2023, "2024-06-14"),
    # --- Copa America ---
    Tournament("CA2007",  "Copa América 2007",    "COPA", 2006, "2007-06-26"),
    Tournament("CA2011",  "Copa América 2011",    "COPA", 2010, "2011-07-01"),
    Tournament("CA2015",  "Copa América 2015",    "COPA", 2014, "2015-06-11"),
    Tournament("CA2019",  "Copa América 2019",    "COPA", 2018, "2019-06-14"),
    Tournament("CA2021",  "Copa América 2021",    "COPA", 2020, "2021-06-13"),
    Tournament("CA2024",  "Copa América 2024",    "COPA", 2023, "2024-06-20"),
    # --- Africa Cup of Nations ---
    Tournament("AF2010",  "AFCON 2010",           "AFCN", 2009, "2010-01-10"),
    Tournament("AF2012",  "AFCON 2012",           "AFCN", 2011, "2012-01-21"),
    Tournament("AF2013",  "AFCON 2013",           "AFCN", 2012, "2013-01-19"),
    Tournament("AF2015",  "AFCON 2015",           "AFCN", 2014, "2015-01-17"),
    Tournament("AF2017",  "AFCON 2017",           "AFCN", 2016, "2017-01-14"),
    Tournament("AF2019",  "AFCON 2019",           "AFCN", 2018, "2019-06-21"),
    Tournament("AF2021",  "AFCON 2021",           "AFCN", 2021, "2022-01-09"),
    Tournament("AF2023",  "AFCON 2023",           "AFCN", 2023, "2024-01-13"),
    Tournament("AF2025",  "AFCON 2025",           "AFCN", 2024, "2025-12-21"),
    # --- AFC Asian Cup ---
    Tournament("AS2007",  "Asian Cup 2007",       "AFAC", 2006, "2007-07-07"),
    Tournament("AS2011",  "Asian Cup 2011",       "AFAC", 2010, "2011-01-07"),
    Tournament("AS2015",  "Asian Cup 2015",       "AFAC", 2014, "2015-01-09"),
    Tournament("AS2019",  "Asian Cup 2019",       "AFAC", 2018, "2019-01-05"),
    Tournament("AS2023",  "Asian Cup 2023",       "AFAC", 2022, "2024-01-12"),
]


def _load_completed() -> set[tuple[str, str]]:
    """Return the set of (tournament_id, team_id) pairs already in the output CSV.

    Lets the scraper resume after a crash or block without re-scraping.
    """
    if not OUTPUT_PATH.exists():
        return set()
    completed = set()
    with OUTPUT_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add((row["tournament_id"], row["team_id"]))
    return completed


def _open_writer():
    """Open the output CSV in append mode, writing the header if new."""
    new_file = not OUTPUT_PATH.exists()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = OUTPUT_PATH.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    if new_file:
        writer.writeheader()
        fh.flush()
    return fh, writer


def fetch_team_links(session: requests.Session, t: Tournament) -> list[tuple[int, str, str]]:
    """Hit the participants page; return [(team_id, team_name, team_url_path), ...]."""
    url = f"{BASE_URL}/x/teilnehmer/pokalwettbewerb/{t.code}/saison_id/{t.saison_id}"
    response = session.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS, allow_redirects=True)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    teams: list[tuple[int, str, str]] = []
    seen: set[int] = set()
    href_pattern = re.compile(r"^(?P<path>/[^/]+/startseite/verein/(?P<id>\d+))")
    for anchor in soup.select("a[href*='/startseite/verein/']"):
        href = anchor.get("href") or ""
        match = href_pattern.match(href)
        if not match:
            continue
        team_id = int(match.group("id"))
        if team_id in seen:
            continue
        text = anchor.get_text(" ", strip=True)
        if not text:
            continue
        seen.add(team_id)
        teams.append((team_id, text, match.group("path")))
    return teams


def fetch_squad(session: requests.Session, team_path: str, team_id: int, saison_id: int) -> list[tuple[int, str]]:
    """Fetch a team's tournament-specific squad. Returns [(player_id, player_name), ...]."""
    # team_path is /slug/startseite/verein/{id}; rewrite to /slug/kader/verein/{id}/...
    parts = team_path.strip("/").split("/")
    slug = parts[0]
    url = f"{BASE_URL}/{slug}/kader/verein/{team_id}/saison_id/{saison_id}/plus/1"
    response = session.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS, allow_redirects=True)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    players: list[tuple[int, str]] = []
    seen: set[int] = set()
    href_pattern = re.compile(r"/profil/spieler/(\d+)")
    for anchor in soup.select("a[href*='/profil/spieler/']"):
        href = anchor.get("href") or ""
        match = href_pattern.search(href)
        if not match:
            continue
        player_id = int(match.group(1))
        if player_id in seen:
            continue
        name = anchor.get_text(" ", strip=True)
        if not name:
            continue
        seen.add(player_id)
        players.append((player_id, name))
    return players


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only scrape the first N tournaments (smoke test).")
    args = parser.parse_args()

    tournaments = TOURNAMENTS[: args.limit] if args.limit else TOURNAMENTS

    completed = _load_completed()
    if completed:
        print(f"Resuming: {len({t for t, _ in completed})} tournaments / {len(completed)} team-rows already in CSV.")

    fh, writer = _open_writer()
    session = requests.Session()
    failures: list[tuple[str, str]] = []
    rows_written = 0

    try:
        for t_idx, t in enumerate(tournaments, start=1):
            print(f"\n[{t_idx}/{len(tournaments)}] {t.tournament_id} — {t.name} (code={t.code}, saison_id={t.saison_id})")
            try:
                team_links = fetch_team_links(session, t)
            except requests.RequestException as e:
                print(f"  participants fetch failed: {e}")
                failures.append((t.tournament_id, f"participants: {e}"))
                time.sleep(DELAY_SECONDS)
                continue
            print(f"  found {len(team_links)} teams")
            time.sleep(DELAY_SECONDS)

            for team_idx, (team_id, team_name, team_path) in enumerate(team_links, start=1):
                key = (t.tournament_id, str(team_id))
                if key in completed:
                    print(f"    [{team_idx}/{len(team_links)}] {team_name:<25} (cached, skipping)")
                    continue
                print(f"    [{team_idx}/{len(team_links)}] {team_name:<25} ", end="", flush=True)
                try:
                    players = fetch_squad(session, team_path, team_id, t.saison_id)
                except requests.RequestException as e:
                    print(f"FAILED: {e}")
                    failures.append((t.tournament_id, f"{team_name}: {e}"))
                    time.sleep(DELAY_SECONDS)
                    continue
                except Exception as e:
                    print(f"PARSE ERROR: {e}")
                    failures.append((t.tournament_id, f"{team_name}: parse: {e}"))
                    time.sleep(DELAY_SECONDS)
                    continue

                for player_id, player_name in players:
                    writer.writerow({
                        "tournament_id": t.tournament_id,
                        "tournament_name": t.name,
                        "tournament_code": t.code,
                        "tournament_saison_id": t.saison_id,
                        "tournament_date": t.start_date,
                        "team_id": team_id,
                        "team_name": team_name,
                        "player_id": player_id,
                        "player_name": player_name,
                    })
                fh.flush()  # safe to ctrl-c
                rows_written += len(players)
                print(f"{len(players)} players")
                time.sleep(DELAY_SECONDS)
    finally:
        fh.close()

    print(f"\nDone. Wrote {rows_written:,} new player-rows total.")
    print(f"Output: {OUTPUT_PATH}")
    if failures:
        print(f"\n{len(failures)} failures:")
        for t_id, reason in failures:
            print(f"  - {t_id}: {reason}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

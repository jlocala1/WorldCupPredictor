"""Scrape current squad market values for 2026 World Cup teams from Transfermarkt.

For each national team listed in TEAMS, this script:
  1. Hits Transfermarkt's quick-search to find the national-team page URL.
  2. Loads that team's startseite (overview) page.
  3. Pulls total squad market value, squad size, and average player value.

Output: data/raw/transfermarkt_squad_values.csv
Columns: team_name, total_value_eur, squad_size, avg_value_eur
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
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
SEARCH_URL = BASE_URL + "/schnellsuche/ergebnis/schnellsuche"
DELAY_SECONDS = 3
TIMEOUT_SECONDS = 30

TEAMS: list[str] = [
    "USA", "Mexico", "Canada",
    "Argentina", "Brazil", "Uruguay", "Ecuador", "Colombia", "Venezuela",
    "Chile", "Peru", "Bolivia", "Paraguay",
    "France", "Germany", "Spain", "England", "Portugal", "Netherlands",
    "Belgium", "Italy", "Croatia", "Switzerland", "Austria", "Denmark",
    "Serbia", "Czech Republic", "Poland", "Hungary", "Slovakia", "Slovenia",
    "Albania", "Romania", "Ukraine", "Turkey", "Georgia", "Scotland",
    "Morocco", "Senegal", "Nigeria", "Cameroon", "South Africa", "Egypt",
    "Tunisia", "DR Congo", "Ivory Coast", "Mali",
    "Japan", "South Korea", "Australia", "Iran", "Saudi Arabia",
    "New Zealand", "Qatar",
]

# Some names search poorly on Transfermarkt; pre-rewrite them to a form that
# returns a clean national-team match. Anything not listed uses the team name as-is.
SEARCH_QUERY_OVERRIDES: dict[str, str] = {
    "USA": "United States",
    "South Korea": "Korea Republic",
    "Czech Republic": "Czechia",
    "Turkey": "Turkiye",
    "DR Congo": "Democratic Republic of the Congo",
    "Ivory Coast": "Cote d'Ivoire",
}

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "transfermarkt_squad_values.csv"


def parse_euro_amount(text: str) -> float | None:
    """Convert a Transfermarkt money string ('€1.20bn', '€850.00m', '€500Th.') to euros."""
    if not text:
        return None
    cleaned = text.replace("\xa0", " ").strip()
    match = re.search(r"€\s*([\d.,]+)\s*(bn|m|Th\.?|k)?", cleaned, flags=re.IGNORECASE)
    if not match:
        return None
    number_str = match.group(1).replace(",", "")
    try:
        value = float(number_str)
    except ValueError:
        return None
    suffix = (match.group(2) or "").lower().rstrip(".")
    multipliers = {"bn": 1_000_000_000, "m": 1_000_000, "th": 1_000, "k": 1_000, "": 1}
    return value * multipliers.get(suffix, 1)


def find_national_team_url(session: requests.Session, team: str) -> str | None:
    """Search Transfermarkt for the team and return its overview-page URL.

    Transfermarkt lumps senior national teams into the "Clubs" search box rather
    than giving them their own category, so we have to match by link text instead
    of relying on a section header. Youth-team variants (U-17/U-20/U-23) and
    women's teams are filtered out.
    """
    query = SEARCH_QUERY_OVERRIDES.get(team, team)
    response = session.get(SEARCH_URL, params={"query": query}, headers=HEADERS, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    candidates: list[tuple[str, str]] = []
    youth_pattern = re.compile(r"\bU-?\d+\b", re.IGNORECASE)
    for anchor in soup.select("a[href*='/startseite/verein/']"):
        text = anchor.get_text(" ", strip=True)
        href = anchor.get("href") or ""
        if not text or not href:
            continue
        if youth_pattern.search(text) or youth_pattern.search(href):
            continue
        lower_href = href.lower()
        if "frauen" in lower_href or "women" in text.lower():
            continue
        candidates.append((text, urljoin(BASE_URL, href)))

    targets = {team.lower(), query.lower()}

    # Prefer an exact (case-insensitive) match on the link text.
    for text, url in candidates:
        if text.lower() in targets:
            return url

    # Then accept a substring match (e.g. "Korea, Republic of" for "Korea Republic").
    for text, url in candidates:
        text_lower = text.lower()
        if any(t in text_lower for t in targets):
            return url

    return None


def scrape_team_page(session: requests.Session, team_url: str) -> dict[str, float | int | None]:
    """Pull total value, squad size, and average value off a national-team overview page."""
    response = session.get(team_url, headers=HEADERS, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    total_value: float | None = None
    squad_size: int | None = None
    avg_value: float | None = None

    # Headline total market value lives in the page header.
    header_value = soup.select_one(
        "a.data-header__market-value-wrapper, div.data-header__market-value-wrapper"
    )
    if header_value:
        total_value = parse_euro_amount(header_value.get_text(" ", strip=True))

    # The squad table holds player rows; its footer repeats total + average.
    squad_table = soup.select_one("table.items")
    if squad_table:
        player_rows = [row for row in squad_table.select("tbody tr") if row.select_one("td.hauptlink")]
        if player_rows:
            squad_size = len(player_rows)

        footer = squad_table.find("tfoot")
        if footer:
            for cell in footer.find_all("td"):
                cell_text = cell.get_text(" ", strip=True)
                value = parse_euro_amount(cell_text)
                if value is None:
                    continue
                # The "Ø" (average) cell is distinct from the totals cell.
                if "Ø" in cell_text or "average" in cell_text.lower():
                    avg_value = value
                elif total_value is None:
                    total_value = value

    if avg_value is None and total_value is not None and squad_size:
        avg_value = total_value / squad_size

    return {"total_value_eur": total_value, "squad_size": squad_size, "avg_value_eur": avg_value}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only scrape the first N teams (useful for testing the parser).",
    )
    args = parser.parse_args()

    teams = TEAMS[: args.limit] if args.limit else TEAMS

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    rows: list[dict] = []
    failures: list[tuple[str, str]] = []

    for index, team in enumerate(teams, start=1):
        print(f"[{index:2d}/{len(teams)}] {team:18s} ... ", end="", flush=True)
        try:
            team_url = find_national_team_url(session, team)
            if not team_url:
                print("no team URL found in search results")
                failures.append((team, "no URL"))
                time.sleep(DELAY_SECONDS)
                continue
            time.sleep(DELAY_SECONDS)
            data = scrape_team_page(session, team_url)
            rows.append({"team_name": team, **data})
            print(
                f"size={data['squad_size']}  "
                f"total={data['total_value_eur']}  "
                f"avg={data['avg_value_eur']}"
            )
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            print(f"HTTP {status}")
            failures.append((team, f"HTTP {status}"))
        except requests.RequestException as exc:
            print(f"network error: {exc}")
            failures.append((team, f"network: {exc}"))
        except Exception as exc:  # parsing errors, unexpected HTML, etc.
            print(f"parse error: {exc}")
            failures.append((team, f"parse: {exc}"))
        time.sleep(DELAY_SECONDS)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["team_name", "total_value_eur", "squad_size", "avg_value_eur"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {OUTPUT_PATH}")
    if failures:
        print(f"\n{len(failures)} failures:")
        for team, reason in failures:
            print(f"  - {team}: {reason}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

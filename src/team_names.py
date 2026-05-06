"""Team-name normalization across data sources.

Different datasets use different conventions for the same country (e.g.
'USA' vs 'United States', 'Korea Republic' vs 'South Korea', 'IR Iran' vs
'Iran'). All downstream merges go through to_canonical(), which rewrites
any known variant to the form used by results.csv (Kaggle, martj42) — our
project-wide canonical naming.

Add new variants here as they are discovered while joining datasets.
"""

from __future__ import annotations

# Anything not in this map is assumed to already be canonical.
ALIASES: dict[str, str] = {
    # fifa_ranking (Kaggle, cashncarry)
    "USA": "United States",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "Congo DR": "DR Congo",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Czechia": "Czech Republic",
    "IR Iran": "Iran",
    "Cabo Verde": "Cape Verde",
    "Curacao": "Curaçao",

    # davidcariboo countries.csv
    "Korea, South": "South Korea",
    "Korea, North": "North Korea",

    # davidcariboo national_teams.csv
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
}


def to_canonical(name: str | None) -> str | None:
    """Rewrite a team name to its canonical results.csv form.

    Whitespace is stripped. Unknown names pass through unchanged so that
    teams we never explicitly mapped (e.g. minor associate members) still
    work in features computed only from results.csv.
    """
    if name is None:
        return None
    cleaned = name.strip()
    return ALIASES.get(cleaned, cleaned)

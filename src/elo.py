"""Elo rating system for international football matches.

Implements the full eloratings.net methodology:
  - Initial rating 1500 for every new team
  - Match-importance K factor varying by tournament type (60 / 50 / 40 / 30 / 20)
  - Home-advantage offset of +100 for the home team in non-neutral games,
    applied to the *expected* outcome calculation only (not stored on ratings)
  - Goal-difference multiplier so blowouts move ratings more than narrow wins

Two consumers share this logic:
  - features.py uses compute_pre_match_elo() to attach pre-match ratings as
    features over the full results.csv history.
  - simulate.py (later) will use EloRating to maintain ratings live during
    bracket simulation, applying updates after each predicted outcome.

Note: ratings produced here are computed *with* the eloratings.net formula
but will not exactly match their published numbers — they have been running
their system live since 1997 with their own initial conditions and data
sources, while we replay from scratch over results.csv (martj42).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

INITIAL_RATING = 1500.0
HOME_ADVANTAGE = 100.0  # eloratings.net: applied to home rating in non-neutral games
DEFAULT_K = 30.0  # used only when tournament is unknown


def k_for_tournament(tournament: str | None) -> float:
    """Map a results.csv `tournament` string to eloratings.net's K-factor.

    Tiers:
      60 — World Cup final tournament
      50 — Continental championship final tournament + intercontinental finals
           (Copa América, Euro, AFCON, Asian Cup, Gold Cup/CONCACAF Championship,
           Confederations Cup, Oceania Nations Cup)
      40 — All *qualification competitions and Nations League (UEFA, CONCACAF)
      20 — Friendly
      30 — Everything else (minor regional tournaments, Olympic Games, etc.)
    """
    if tournament is None:
        return DEFAULT_K
    t = tournament.lower()

    if "friendly" in t:
        return 20.0

    is_qualifier = "qualif" in t

    if "world cup" in t:
        return 40.0 if is_qualifier else 60.0

    continental_finals = (
        "copa américa", "copa america",
        "african cup of nations",
        "afc asian cup",
        "uefa euro",
        "gold cup",
        "concacaf championship",
        "confederations cup",
        "oceania nations cup",
    )
    if any(c in t for c in continental_finals):
        return 40.0 if is_qualifier else 50.0

    if "nations league" in t:  # UEFA + CONCACAF, plus their qualifiers
        return 40.0

    return DEFAULT_K


@dataclass
class EloRating:
    """Mutable per-team rating store. Used by both feature computation and live simulation."""

    initial: float = INITIAL_RATING
    home_advantage: float = HOME_ADVANTAGE
    k_default: float = DEFAULT_K
    ratings: dict[str, float] = field(default_factory=dict)

    def get(self, team: str) -> float:
        return self.ratings.get(team, self.initial)

    @staticmethod
    def expected_home(home_rating: float, away_rating: float) -> float:
        """Pr(home wins) under the standard Elo logistic, ignoring draws."""
        return 1.0 / (1.0 + 10.0 ** ((away_rating - home_rating) / 400.0))

    @staticmethod
    def _goal_diff_multiplier(goal_diff: float) -> float:
        """eloratings.net margin-of-victory bonus, applied as a K multiplier.

        Returns 1.0 for margins of 0-1, 1.5 at 2, 1.75 at 3, then (11+g)/8 for
        larger margins (1.875 at 4, 2.0 at 5, 2.125 at 6, ...).
        """
        g = abs(goal_diff)
        if g <= 1:
            return 1.0
        if g == 2:
            return 1.5
        if g == 3:
            return 1.75
        return (11 + g) / 8

    def update(
        self,
        home: str,
        away: str,
        home_score: float,
        away_score: float,
        tournament: str | None = None,
        neutral: bool = False,
    ) -> tuple[float, float]:
        """Apply a zero-sum Elo update and return the pre-match ratings.

        Home advantage is added to the home rating *only inside* the expected-
        outcome calculation; the stored rating is the raw, unadjusted rating.
        """
        r_home = self.get(home)
        r_away = self.get(away)
        r_home_eff = r_home + (0.0 if neutral else self.home_advantage)

        if home_score > away_score:
            actual_home = 1.0
        elif home_score == away_score:
            actual_home = 0.5
        else:
            actual_home = 0.0

        k = k_for_tournament(tournament) if tournament is not None else self.k_default
        multiplier = self._goal_diff_multiplier(home_score - away_score)
        delta = k * multiplier * (actual_home - self.expected_home(r_home_eff, r_away))
        self.ratings[home] = r_home + delta
        self.ratings[away] = r_away - delta
        return r_home, r_away


def compute_pre_match_elo(matches: pd.DataFrame) -> pd.DataFrame:
    """Iterate matches chronologically and emit pre-match ratings for each side.

    Input columns required: date, home_team, away_team, home_score, away_score,
    tournament, neutral. Returns a copy sorted by date with two added columns:
    home_elo_pre, away_elo_pre.

    Matches with NaN scores (scheduled but unplayed) receive the latest known
    ratings but do not themselves contribute updates — that way pre-tournament
    fixtures get sensible Elo without being treated as outcomes.
    """
    matches = matches.sort_values("date", kind="stable").reset_index(drop=True)
    rating = EloRating()
    home_pre: list[float] = []
    away_pre: list[float] = []
    for row in matches.itertuples(index=False):
        if pd.isna(row.home_score) or pd.isna(row.away_score):
            home_pre.append(rating.get(row.home_team))
            away_pre.append(rating.get(row.away_team))
        else:
            ph, pa = rating.update(
                row.home_team, row.away_team,
                row.home_score, row.away_score,
                tournament=row.tournament,
                neutral=bool(row.neutral),
            )
            home_pre.append(ph)
            away_pre.append(pa)
    out = matches.copy()
    out["home_elo_pre"] = home_pre
    out["away_elo_pre"] = away_pre
    return out

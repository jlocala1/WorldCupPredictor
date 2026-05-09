"""LLM baselines for the World Cup predictor.

This module adds three prompt-controlled LLM prediction tracks beside the
existing ML ensemble:

  - feature_only_blind: anonymized Team A/B, engineered features only.
  - feature_plus_rag: real teams, engineered features, controlled historical
    context retrieved from this repository's data.
  - knowledge_only: real teams and fixture metadata only, no engineered
    features or retrieval.

The expected probability order matches the ML pipeline everywhere:
    [away_win, draw, home_win] == labels [0, 1, 2]

Example smoke run without an API key:
    python src/llm_baselines.py --provider heuristic --limit 5 --splits test

Example OpenAI-compatible run:
    set LLM_API_KEY=...
    set LLM_MODEL=...
    python src/llm_baselines.py --provider openai-compatible --splits val test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import accuracy_score, f1_score, log_loss

# Make sibling modules importable when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensemble import SoftVoteEnsemble  # noqa: F401, E402  # needed for joblib unpickle

ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
MODELS_DIR = ROOT / "models"
ENV_PATH = ROOT / ".env"

CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = {0: "away_win", 1: "draw", 2: "home_win"}

PROMPT_PROFILES = ("feature_only_blind", "feature_plus_rag", "knowledge_only")
OUTPUT_FILES = {
    "feature_only_blind": "llm_predictions_feature_only.csv",
    "feature_plus_rag": "llm_predictions_feature_plus_rag.csv",
    "knowledge_only": "llm_predictions_knowledge_only.csv",
}

FEATURE_DEFINITIONS = {
    "form": "Trailing 10-match form: win rate, goals for, and goals against.",
    "h2h": "Home-side prior head-to-head win rate against the away side.",
    "elo": "Pre-match Elo rating and home-minus-away Elo difference.",
    "fifa_rank": "Latest available FIFA ranking and home-minus-away rank difference; lower rank is better.",
    "squad_value": "Date-correct squad market value, top-26 value, average value, and squad size.",
    "z_scores": "Top-player attacking, creating, and defending z-scores plus home-minus-away differences.",
    "neutral": "Whether the match is played at a neutral venue.",
    "missing_flags": "A true missing flag means the corresponding source value was unavailable before imputation.",
}

ENGINEERED_FEATURE_HINTS = (
    "elo", "fifa", "rank", "form", "h2h", "squad", "value", "top26",
    "avg_value", "attacking_z", "creating_z", "defending_z", "z_diff",
)


@dataclass
class RAGContext:
    text: str
    max_source_date: str | None
    n_sources: int


@dataclass
class ParsedPrediction:
    valid: bool
    probs: dict[str, float]
    predicted_label: int | None
    confidence: float | None
    feature_factors: list[str]
    explanation: str
    warnings: list[str]
    raw_text: str
    invalid_reason: str | None = None


def stable_match_id(row: pd.Series) -> str:
    """Stable row id across runs without relying on CSV index persistence."""
    raw = "|".join(
        str(row.get(c, ""))
        for c in ("date", "home_team", "away_team", "tournament", "split")
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def load_dotenv(path: Path = ENV_PATH) -> None:
    """Load simple KEY=VALUE pairs from .env without requiring python-dotenv.

    Existing process environment values win over .env values. This keeps shell
    overrides explicit while making local runs easier.
    """
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_feature_names() -> list[str]:
    """Load the exact feature column order used by the saved ML artifacts."""
    return joblib.load(MODELS_DIR / "feature_names.pkl")


def base_feature_names(feature_names: list[str]) -> list[str]:
    return [c for c in feature_names if not c.endswith("_missing")]


def format_number(value: Any) -> Any:
    """Convert numpy/pandas values to compact JSON-safe scalars."""
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 4)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def row_feature_payload(row: pd.Series, feature_names: list[str]) -> dict[str, dict[str, Any]]:
    """Return engineered features with explicit missingness, no metadata."""
    payload: dict[str, dict[str, Any]] = {}
    for col in base_feature_names(feature_names):
        if col not in row.index:
            continue
        payload[col] = {
            "value": format_number(row[col]),
            "missing": bool(pd.isna(row[col])),
        }
    return payload


def build_ml_feature_frame(df: pd.DataFrame, feature_names: list[str], fill_values: dict[str, float]) -> pd.DataFrame:
    """Build an ML-ready feature frame from raw rows in the saved feature order."""
    base_cols = base_feature_names(feature_names)
    x = pd.DataFrame(index=df.index)
    for col in base_cols:
        x[col] = df[col] if col in df.columns else np.nan
    for col in feature_names:
        if col.endswith("_missing"):
            raw_col = col[: -len("_missing")]
            x[col] = x[raw_col].isna().astype(int) if raw_col in x.columns else 1
    for col, fill in fill_values.items():
        if col in x.columns:
            x[col] = x[col].fillna(fill)
    return x[feature_names]


class ControlledRAG:
    """Small date-filtered retriever over the project's own feature table."""

    def __init__(self, df: pd.DataFrame, max_recent_matches: int = 8) -> None:
        self.df = df.copy()
        self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce")
        self.max_recent_matches = max_recent_matches

    def build_context(self, row: pd.Series) -> RAGContext:
        match_date = pd.to_datetime(row["date"], errors="coerce")
        home = str(row["home_team"])
        away = str(row["away_team"])
        historical = self.df[
            (self.df["label"].notna())
            & (self.df["date"].notna())
            & (self.df["date"] < match_date)
        ].copy()

        blocks: list[str] = []
        max_date: pd.Timestamp | None = None

        for team in (home, away):
            text, team_max = self._team_recent_summary(team, historical)
            blocks.append(text)
            max_date = self._max_date(max_date, team_max)

        h2h_text, h2h_max = self._h2h_summary(home, away, historical)
        blocks.append(h2h_text)
        max_date = self._max_date(max_date, h2h_max)

        blocks.append(
            "Project feature notes: "
            + "; ".join(f"{k}: {v}" for k, v in FEATURE_DEFINITIONS.items())
        )

        return RAGContext(
            text="\n".join(blocks),
            max_source_date=max_date.date().isoformat() if max_date is not None else None,
            n_sources=len(blocks),
        )

    @staticmethod
    def _max_date(current: pd.Timestamp | None, candidate: pd.Timestamp | None) -> pd.Timestamp | None:
        if candidate is None or pd.isna(candidate):
            return current
        if current is None or candidate > current:
            return candidate
        return current

    def _team_recent_summary(self, team: str, historical: pd.DataFrame) -> tuple[str, pd.Timestamp | None]:
        mask = (historical["home_team"] == team) | (historical["away_team"] == team)
        matches = historical[mask].sort_values("date").tail(self.max_recent_matches)
        if matches.empty:
            return f"{team} recent project history before this match: no prior rows available.", None

        wins = draws = losses = gf = ga = 0
        for _, m in matches.iterrows():
            is_home = m["home_team"] == team
            team_goals = float(m["home_score"] if is_home else m["away_score"])
            opp_goals = float(m["away_score"] if is_home else m["home_score"])
            gf += team_goals
            ga += opp_goals
            if team_goals > opp_goals:
                wins += 1
            elif team_goals < opp_goals:
                losses += 1
            else:
                draws += 1

        latest = matches["date"].max()
        return (
            f"{team} recent project history before this match: "
            f"{wins}W-{draws}D-{losses}L over last {len(matches)} rows, "
            f"goals_for={gf:.0f}, goals_against={ga:.0f}, latest_source_date={latest.date().isoformat()}.",
            latest,
        )

    def _h2h_summary(self, home: str, away: str, historical: pd.DataFrame) -> tuple[str, pd.Timestamp | None]:
        mask = (
            ((historical["home_team"] == home) & (historical["away_team"] == away))
            | ((historical["home_team"] == away) & (historical["away_team"] == home))
        )
        matches = historical[mask].sort_values("date")
        if matches.empty:
            return f"Head-to-head project history before this match: no prior {home} vs {away} rows.", None

        home_wins = draws = away_wins = 0
        for _, m in matches.iterrows():
            home_goals = float(m["home_score"] if m["home_team"] == home else m["away_score"])
            away_goals = float(m["away_score"] if m["home_team"] == home else m["home_score"])
            if home_goals > away_goals:
                home_wins += 1
            elif home_goals < away_goals:
                away_wins += 1
            else:
                draws += 1
        latest = matches["date"].max()
        return (
            f"Head-to-head project history before this match: {len(matches)} prior rows; "
            f"{home}_wins={home_wins}, draws={draws}, {away}_wins={away_wins}, "
            f"latest_source_date={latest.date().isoformat()}.",
            latest,
        )


class PromptBuilder:
    """Build profile-specific prompts with strict field boundaries."""

    def __init__(self, feature_names: list[str], knowledge_date_granularity: str = "year") -> None:
        self.feature_names = feature_names
        self.knowledge_date_granularity = knowledge_date_granularity

    def build_messages(self, profile: str, row: pd.Series, rag_context: RAGContext | None = None) -> list[dict[str, str]]:
        if profile not in PROMPT_PROFILES:
            raise ValueError(f"Unknown prompt profile: {profile}")
        system = (
            "You are a cautious football match prediction analyst. "
            "Return only valid JSON with keys: p_away_win, p_draw, p_home_win, "
            "predicted_label, confidence, feature_factors, explanation, warnings. "
            "Probabilities must be numeric and sum to 1. Labels are 0=away_win, "
            "1=draw, 2=home_win. Keep explanation under 80 words."
        )
        if profile == "feature_only_blind":
            user = self._feature_only_prompt(row)
        elif profile == "feature_plus_rag":
            if rag_context is None:
                raise ValueError("feature_plus_rag requires rag_context")
            user = self._feature_plus_rag_prompt(row, rag_context)
        else:
            user = self._knowledge_only_prompt(row)
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _feature_only_prompt(self, row: pd.Series) -> str:
        payload = {
            "profile": "feature_only_blind",
            "fixture": {
                "home": "Team A",
                "away": "Team B",
                "home_away_meaning": "Team A is home side; Team B is away side.",
            },
            "allowed_information": (
                "Only the engineered numeric features below may be used. "
                "Do not infer team identity, tournament, date, country, city, or outside football history."
            ),
            "feature_definitions": FEATURE_DEFINITIONS,
            "features": row_feature_payload(row, self.feature_names),
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    def _feature_plus_rag_prompt(self, row: pd.Series, rag_context: RAGContext) -> str:
        payload = {
            "profile": "feature_plus_rag",
            "fixture": {
                "date": str(row["date"]),
                "home_team": str(row["home_team"]),
                "away_team": str(row["away_team"]),
                "tournament": str(row.get("tournament", "")),
                "neutral": bool(row.get("neutral", False)),
            },
            "allowed_information": (
                "Use the engineered features and the controlled project-data context only. "
                "The retrieved context is date-filtered to sources before this fixture."
            ),
            "feature_definitions": FEATURE_DEFINITIONS,
            "features": row_feature_payload(row, self.feature_names),
            "retrieved_context": rag_context.text,
            "retrieved_context_max_source_date": rag_context.max_source_date,
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    def _knowledge_only_prompt(self, row: pd.Series) -> str:
        date = pd.to_datetime(row["date"], errors="coerce")
        if self.knowledge_date_granularity == "full":
            date_value = date.date().isoformat() if pd.notna(date) else None
        else:
            date_value = int(date.year) if pd.notna(date) else None
        payload = {
            "profile": "knowledge_only",
            "fixture": {
                "date": date_value,
                "date_granularity": "full" if self.knowledge_date_granularity == "full" else "year",
                "home_team": str(row["home_team"]),
                "away_team": str(row["away_team"]),
                "tournament": str(row.get("tournament", "")),
                "neutral": bool(row.get("neutral", False)),
                "home_away_meaning": "home_team is the home side; away_team is the away side.",
            },
            "allowed_information": (
                "Use only your general pretrained football knowledge and the fixture metadata above. "
                "Do not use engineered feature values, retrieved documents, labels, scores, or ML probabilities."
            ),
        }
        return json.dumps(payload, indent=2, sort_keys=True)


class LLMClient:
    provider_name = "base"

    def complete(self, messages: list[dict[str, str]], profile: str, row: pd.Series) -> str:
        raise NotImplementedError


class OpenAICompatibleClient(LLMClient):
    """Minimal OpenAI-compatible chat/completions client using requests."""

    provider_name = "openai-compatible"

    def __init__(
        self,
        api_key: str,
        model: str,
        api_base: str = "https://api.openai.com/v1",
        temperature: float | None = None,
        max_tokens: int = 700,
        reasoning_effort: str | None = None,
        timeout: int = 60,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout

    def complete(self, messages: list[dict[str, str]], profile: str, row: pd.Series) -> str:
        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        if self._uses_max_completion_tokens():
            payload["max_completion_tokens"] = self.max_tokens
        else:
            payload["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        if not response.ok:
            raise RuntimeError(
                f"OpenAI-compatible API request failed: HTTP {response.status_code} "
                f"{response.reason}. Body: {response.text}"
            )
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _uses_max_completion_tokens(self) -> bool:
        model = self.model.lower()
        return model.startswith(("gpt-5", "o1", "o3", "o4"))


class HeuristicClient(LLMClient):
    """Deterministic local stand-in for smoke tests; not a real LLM baseline."""

    provider_name = "heuristic"

    def complete(self, messages: list[dict[str, str]], profile: str, row: pd.Series) -> str:
        if profile == "knowledge_only":
            probs = self._knowledge_prior(row)
            factors = ["general team-name prior", "home/away designation", "neutral venue"]
        else:
            probs = self._feature_prior(row)
            factors = ["elo_diff", "fifa_rank_diff", "form", "squad_value_diff"]
            if profile == "feature_plus_rag":
                factors.append("controlled historical context")
        pred = int(np.argmax(probs))
        body = {
            "p_away_win": round(float(probs[0]), 6),
            "p_draw": round(float(probs[1]), 6),
            "p_home_win": round(float(probs[2]), 6),
            "predicted_label": pred,
            "confidence": round(float(max(probs)), 6),
            "feature_factors": factors,
            "explanation": (
                "Deterministic heuristic response for pipeline validation; "
                "replace --provider heuristic with --provider openai-compatible for real LLM runs."
            ),
            "warnings": ["heuristic provider is not an LLM"],
        }
        return json.dumps(body)

    @staticmethod
    def _feature_prior(row: pd.Series) -> np.ndarray:
        score = 0.0
        if pd.notna(row.get("elo_diff")):
            score += float(row["elo_diff"]) / 450.0
        if pd.notna(row.get("fifa_rank_diff")):
            score += -float(row["fifa_rank_diff"]) / 120.0
        if pd.notna(row.get("squad_value_diff")):
            score += np.tanh(float(row["squad_value_diff"]) / 600_000_000.0)
        if pd.notna(row.get("home_form_win_rate")) and pd.notna(row.get("away_form_win_rate")):
            score += float(row["home_form_win_rate"] - row["away_form_win_rate"])
        draw = 0.24
        home = (1.0 - draw) / (1.0 + math.exp(-score))
        away = 1.0 - draw - home
        return np.array([away, draw, home])

    @staticmethod
    def _knowledge_prior(row: pd.Series) -> np.ndarray:
        # A deliberately weak, auditable fallback: mild home-side prior only.
        draw = 0.27
        home = 0.39 if not bool(row.get("neutral", False)) else 0.365
        away = 1.0 - draw - home
        return np.array([away, draw, home])


class ResponseCache:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def key(self, provider: str, model: str, profile: str, match_id: str, messages: list[dict[str, str]]) -> str:
        prompt_hash = hashlib.sha1(json.dumps(messages, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        raw = f"{provider}|{model}|{profile}|{match_id}|{prompt_hash}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> str | None:
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)["raw_text"]

    def set(self, key: str, raw_text: str) -> None:
        path = self.cache_dir / f"{key}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump({"raw_text": raw_text}, f, indent=2)


def parse_prediction(raw_text: str) -> ParsedPrediction:
    data = _extract_json(raw_text)
    if data is None:
        return ParsedPrediction(False, {}, None, None, [], "", [], raw_text, "response was not valid JSON")

    try:
        probs = {
            "p_away_win": float(data["p_away_win"]),
            "p_draw": float(data["p_draw"]),
            "p_home_win": float(data["p_home_win"]),
        }
    except (KeyError, TypeError, ValueError):
        return ParsedPrediction(False, {}, None, None, [], "", [], raw_text, "missing or nonnumeric probabilities")

    values = np.array([probs["p_away_win"], probs["p_draw"], probs["p_home_win"]], dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        return ParsedPrediction(False, probs, None, None, [], "", [], raw_text, "probabilities outside [0, 1]")

    total = float(values.sum())
    if abs(total - 1.0) > 0.03:
        return ParsedPrediction(False, probs, None, None, [], "", [], raw_text, f"probabilities sum to {total:.4f}")
    if abs(total - 1.0) > 1e-6:
        values = values / total
        probs = {
            "p_away_win": float(values[0]),
            "p_draw": float(values[1]),
            "p_home_win": float(values[2]),
        }

    predicted_label = data.get("predicted_label")
    if predicted_label not in CLASS_LABELS:
        predicted_label = int(np.argmax(values))

    confidence = data.get("confidence", max(values))
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = float(max(values))

    return ParsedPrediction(
        valid=True,
        probs=probs,
        predicted_label=int(predicted_label),
        confidence=confidence,
        feature_factors=[str(x) for x in data.get("feature_factors", [])],
        explanation=str(data.get("explanation", "")),
        warnings=[str(x) for x in data.get("warnings", [])],
        raw_text=raw_text,
    )


def _extract_json(raw_text: str) -> dict[str, Any] | None:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw_text, flags=re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def leakage_flags(profile: str, row: pd.Series, parsed: ParsedPrediction, rag_context: RAGContext | None) -> list[str]:
    flags: list[str] = []
    text = " ".join(
        [parsed.explanation, " ".join(parsed.feature_factors), " ".join(parsed.warnings), parsed.raw_text]
    ).lower()
    if profile == "feature_only_blind":
        for forbidden in (str(row["home_team"]), str(row["away_team"]), str(row.get("tournament", ""))):
            if forbidden and forbidden.lower() in text:
                flags.append(f"mentions_forbidden_identity:{forbidden}")
        date = str(row.get("date", ""))
        if date and date.lower() in text:
            flags.append("mentions_forbidden_date")
    elif profile == "feature_plus_rag":
        if rag_context and rag_context.max_source_date:
            max_date = pd.to_datetime(rag_context.max_source_date)
            match_date = pd.to_datetime(row["date"])
            if max_date >= match_date:
                flags.append("rag_context_not_strictly_prior")
    elif profile == "knowledge_only":
        flags.append("not_leakage_auditable")
        for hint in ENGINEERED_FEATURE_HINTS:
            if hint in text:
                flags.append(f"mentions_engineered_feature:{hint}")
                break
    return flags


def run_profile(
    profile: str,
    rows: pd.DataFrame,
    full_df: pd.DataFrame,
    feature_names: list[str],
    client: LLMClient,
    model_name: str,
    output_dir: Path,
    use_cache: bool = True,
    sleep_seconds: float = 0.0,
    knowledge_date_granularity: str = "year",
) -> pd.DataFrame:
    builder = PromptBuilder(feature_names, knowledge_date_granularity=knowledge_date_granularity)
    rag = ControlledRAG(full_df)
    cache = ResponseCache(output_dir / "llm_cache")
    records: list[dict[str, Any]] = []

    for i, (_, row) in enumerate(rows.iterrows(), start=1):
        match_id = stable_match_id(row)
        rag_context = rag.build_context(row) if profile == "feature_plus_rag" else None
        messages = builder.build_messages(profile, row, rag_context=rag_context)
        key = cache.key(client.provider_name, model_name, profile, match_id, messages)
        raw_text = cache.get(key) if use_cache else None
        if raw_text is not None and not parse_prediction(raw_text).valid:
            raw_text = None
        if raw_text is None:
            raw_text = client.complete(messages, profile, row)
            parsed = parse_prediction(raw_text)
            if not parsed.valid and client.provider_name != "heuristic":
                retry_messages = messages + [{
                    "role": "user",
                    "content": (
                        "Your previous response was invalid. Return only JSON with numeric "
                        "p_away_win, p_draw, p_home_win summing to 1."
                    ),
                }]
                raw_text = client.complete(retry_messages, profile, row)
            cache.set(key, raw_text)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        parsed = parse_prediction(raw_text)
        flags = leakage_flags(profile, row, parsed, rag_context)

        records.append({
            "match_id": match_id,
            "split": row.get("split"),
            "date": row.get("date"),
            "home_team": row.get("home_team"),
            "away_team": row.get("away_team"),
            "tournament": row.get("tournament"),
            "label": format_number(row.get("label")),
            "profile": profile,
            "provider": client.provider_name,
            "model": model_name,
            "valid": parsed.valid,
            "invalid_reason": parsed.invalid_reason,
            "p_away_win": parsed.probs.get("p_away_win"),
            "p_draw": parsed.probs.get("p_draw"),
            "p_home_win": parsed.probs.get("p_home_win"),
            "predicted_label": parsed.predicted_label,
            "confidence": parsed.confidence,
            "feature_factors": json.dumps(parsed.feature_factors, ensure_ascii=False),
            "explanation": parsed.explanation,
            "warnings": json.dumps(parsed.warnings, ensure_ascii=False),
            "leakage_flags": json.dumps(flags, ensure_ascii=False),
            "rag_context_max_source_date": rag_context.max_source_date if rag_context else None,
            "rag_n_sources": rag_context.n_sources if rag_context else 0,
            "raw_response": raw_text,
        })
        print(f"  {profile}: {i}/{len(rows)} rows", end="\r")
    print()
    return pd.DataFrame(records)


def evaluate_predictions(preds: pd.DataFrame) -> dict[str, Any]:
    """Evaluate one prediction table by split; predict rows are skipped."""
    summary: dict[str, Any] = {}
    for split, split_df in preds.groupby("split"):
        labeled = split_df[split_df["label"].notna()].copy()
        if labeled.empty:
            continue
        total = len(labeled)
        valid = labeled[labeled["valid"] == True].copy()  # noqa: E712
        item: dict[str, Any] = {
            "n_rows": int(total),
            "n_valid": int(len(valid)),
            "invalid_response_rate": float(1.0 - len(valid) / total),
        }
        if not valid.empty:
            y = valid["label"].astype(int).to_numpy()
            probs = valid[["p_away_win", "p_draw", "p_home_win"]].astype(float).to_numpy()
            preds_argmax = probs.argmax(axis=1)
            item.update({
                "accuracy": float(accuracy_score(y, preds_argmax)),
                "macro_f1": float(f1_score(y, preds_argmax, average="macro", zero_division=0)),
                "log_loss": float(log_loss(y, probs, labels=CLASS_LABELS)),
                "brier": multiclass_brier(y, probs),
                "calibration_bins": calibration_bins(y, probs),
            })
        summary[str(split)] = item
    return summary


def multiclass_brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    one_hot = np.zeros_like(probs, dtype=float)
    for i, label in enumerate(y_true):
        one_hot[i, int(label)] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def calibration_bins(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 5) -> list[dict[str, Any]]:
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    bins: list[dict[str, Any]] = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
        if not mask.any():
            continue
        bins.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "count": int(mask.sum()),
            "avg_confidence": float(conf[mask].mean()),
            "accuracy": float((pred[mask] == y_true[mask]).mean()),
        })
    return bins


def load_ml_ensemble_predictions(rows: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame | None, str | None]:
    required = [MODELS_DIR / "ensemble.pkl", MODELS_DIR / "fill_values.pkl"]
    if not all(p.exists() for p in required):
        return None, "missing ensemble.pkl or fill_values.pkl"
    try:
        model = joblib.load(MODELS_DIR / "ensemble.pkl")
        fill_values = joblib.load(MODELS_DIR / "fill_values.pkl")
    except Exception as exc:  # noqa: BLE001 - report environment/model load failures, keep LLM run alive.
        return None, f"could not load ML ensemble artifacts: {exc}"
    x = build_ml_feature_frame(rows, feature_names, fill_values)
    probs = model.predict_proba(x)
    out = pd.DataFrame({
        "match_id": [stable_match_id(r) for _, r in rows.iterrows()],
        "split": rows["split"].values,
        "date": rows["date"].values,
        "home_team": rows["home_team"].values,
        "away_team": rows["away_team"].values,
        "tournament": rows["tournament"].values,
        "label": rows["label"].values,
        "profile": "ml_ensemble",
        "valid": True,
        "p_away_win": probs[:, 0],
        "p_draw": probs[:, 1],
        "p_home_win": probs[:, 2],
        "predicted_label": probs.argmax(axis=1),
        "confidence": probs.max(axis=1),
    })
    return out, None


def pairwise_probability_distances(prediction_tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    pairs = [
        ("ml_ensemble", "feature_only_blind"),
        ("feature_only_blind", "feature_plus_rag"),
        ("knowledge_only", "feature_only_blind"),
        ("knowledge_only", "ml_ensemble"),
    ]
    result: dict[str, Any] = {}
    for left, right in pairs:
        if left not in prediction_tables or right not in prediction_tables:
            continue
        merged = prediction_tables[left].merge(
            prediction_tables[right],
            on=["match_id", "split"],
            suffixes=("_left", "_right"),
        )
        merged = merged[(merged.get("valid_left", True) == True) & (merged.get("valid_right", True) == True)]  # noqa: E712
        if merged.empty:
            continue
        for split, split_df in merged.groupby("split"):
            lp = split_df[["p_away_win_left", "p_draw_left", "p_home_win_left"]].astype(float).to_numpy()
            rp = split_df[["p_away_win_right", "p_draw_right", "p_home_win_right"]].astype(float).to_numpy()
            key = f"{left}_vs_{right}:{split}"
            result[key] = {
                "n": int(len(split_df)),
                "mean_l1_distance": float(np.abs(lp - rp).sum(axis=1).mean()),
                "mean_l2_distance": float(np.sqrt(((lp - rp) ** 2).sum(axis=1)).mean()),
            }
    return result


def select_rows(df: pd.DataFrame, splits: list[str], include_predict: bool, limit: int | None) -> pd.DataFrame:
    chosen_splits = list(splits)
    if include_predict and "predict" not in chosen_splits:
        chosen_splits.append("predict")
    rows = df[df["split"].isin(chosen_splits)].copy()
    rows = rows.sort_values(["split", "date", "home_team", "away_team"])
    if limit is not None:
        rows = rows.groupby("split", group_keys=False).head(limit)
    return rows.reset_index(drop=True)


def build_client(args: argparse.Namespace) -> tuple[LLMClient, str]:
    if args.provider == "heuristic":
        return HeuristicClient(), "heuristic"

    api_key = args.api_key or os.getenv(args.api_key_env) or os.getenv("OPENAI_API_KEY")
    model = args.model or os.getenv("LLM_MODEL")
    if not api_key:
        raise RuntimeError(
            f"{args.provider} requires an API key via --api-key, {args.api_key_env}, or OPENAI_API_KEY."
        )
    if not model:
        raise RuntimeError(f"{args.provider} requires --model or LLM_MODEL.")
    client = OpenAICompatibleClient(
        api_key=api_key,
        model=model,
        api_base=args.api_base,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        timeout=args.timeout,
    )
    return client, model


def optional_float(value: str | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run LLM baselines for WorldCupPredictor")
    parser.add_argument("--features", default=str(FEATURES_PATH))
    parser.add_argument("--output-dir", default=str(MODELS_DIR))
    parser.add_argument("--profiles", nargs="+", default=list(PROMPT_PROFILES), choices=PROMPT_PROFILES)
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test", "predict"])
    parser.add_argument("--include-predict", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Rows per split; useful for smoke tests")
    parser.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "heuristic"),
                        choices=["heuristic", "openai-compatible"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-base", default=os.getenv("LLM_API_BASE", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--temperature", type=optional_float, default=optional_float(os.getenv("LLM_TEMPERATURE")))
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("LLM_MAX_TOKENS", "2000")))
    parser.add_argument("--reasoning-effort", default=os.getenv("LLM_REASONING_EFFORT") or None)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--knowledge-date-granularity", default="year", choices=["year", "full"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features_path = Path(args.features)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(features_path)
    rows = select_rows(df, args.splits, args.include_predict, args.limit)
    feature_names = load_feature_names()
    client, model_name = build_client(args)

    print(f"Rows selected: {len(rows)} across splits {sorted(rows['split'].unique())}")
    print(f"Provider: {client.provider_name}; model: {model_name}")

    prediction_tables: dict[str, pd.DataFrame] = {}
    ml_warning = None
    ml_preds, ml_warning = load_ml_ensemble_predictions(rows[rows["split"] != "predict"].copy(), feature_names)
    if ml_preds is not None and not ml_preds.empty:
        prediction_tables["ml_ensemble"] = ml_preds
        ml_preds.to_csv(output_dir / "ml_ensemble_predictions.csv", index=False)
    elif ml_warning:
        print(f"Skipping ml_ensemble comparison: {ml_warning}")

    for profile in args.profiles:
        print(f"Running {profile}...")
        preds = run_profile(
            profile=profile,
            rows=rows,
            full_df=df,
            feature_names=feature_names,
            client=client,
            model_name=model_name,
            output_dir=output_dir,
            use_cache=not args.no_cache,
            sleep_seconds=args.sleep_seconds,
            knowledge_date_granularity=args.knowledge_date_granularity,
        )
        prediction_tables[profile] = preds
        preds.to_csv(output_dir / OUTPUT_FILES[profile], index=False)

    summary = {
        "notes": {
            "class_order": "[away_win, draw, home_win] == labels [0, 1, 2]",
            "knowledge_only_warning": (
                "knowledge_only is a qualitative pretrained-knowledge prior; "
                "it is not leakage-auditable on historical splits."
            ),
            "heuristic_warning": (
                "provider=heuristic is for smoke tests only and is not an LLM baseline."
                if client.provider_name == "heuristic" else None
            ),
            "ml_ensemble_warning": ml_warning,
        },
        "profiles": {
            name: evaluate_predictions(table)
            for name, table in prediction_tables.items()
        },
        "pairwise_probability_distances": pairwise_probability_distances(prediction_tables),
    }
    with (output_dir / "llm_eval_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved predictions and summary to {output_dir}")


if __name__ == "__main__":
    main()

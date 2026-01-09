"""Persistence helpers for news sentiment data."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain_api.core.news_sentiment.models import SymbolSentiment


def get_raw_news_path(base_path: Path, run_id: str, attempt: int, symbol: str) -> Path:
    """Get path for raw news JSON file.

    Args:
        base_path: Base data directory
        run_id: Run identifier (e.g., "paper:2025-12-30")
        attempt: Attempt number
        symbol: Stock ticker symbol

    Returns:
        Path to the raw news JSON file
    """
    # Sanitize run_id for filesystem (replace colons)
    safe_run_id = run_id.replace(":", "_")
    return base_path / "raw" / safe_run_id / str(attempt) / "news" / f"{symbol}.json"


def get_features_path(base_path: Path, run_id: str, attempt: int) -> Path:
    """Get path for aggregated features JSON file.

    Args:
        base_path: Base data directory
        run_id: Run identifier
        attempt: Attempt number

    Returns:
        Path to the news_sentiment.json file
    """
    safe_run_id = run_id.replace(":", "_")
    return base_path / "features" / safe_run_id / str(attempt) / "news_sentiment.json"


def save_raw_news(
    base_path: Path,
    run_id: str,
    attempt: int,
    symbol: str,
    sentiment: SymbolSentiment,
) -> Path:
    """Save raw news data for a symbol.

    Args:
        base_path: Base data directory
        run_id: Run identifier
        attempt: Attempt number
        symbol: Stock ticker symbol
        sentiment: SymbolSentiment to save

    Returns:
        Path where data was saved
    """
    path = get_raw_news_path(base_path, run_id, attempt, symbol)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(sentiment.to_dict(), f, indent=2)

    return path


def save_features(
    base_path: Path,
    run_id: str,
    attempt: int,
    as_of_date: str,
    sentiments: list[SymbolSentiment],
) -> Path:
    """Save aggregated sentiment features.

    Args:
        base_path: Base data directory
        run_id: Run identifier
        attempt: Attempt number
        as_of_date: Reference date (ISO format)
        sentiments: List of SymbolSentiment results

    Returns:
        Path where data was saved
    """
    path = get_features_path(base_path, run_id, attempt)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "run_id": run_id,
        "attempt": attempt,
        "as_of_date": as_of_date,
        "timestamp": datetime.now(UTC).isoformat(),
        "per_symbol": [s.to_dict() for s in sentiments],
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    return path


def load_cached_features(
    base_path: Path,
    run_id: str,
    attempt: int,
) -> dict[str, Any] | None:
    """Load cached features if they exist.

    Args:
        base_path: Base data directory
        run_id: Run identifier
        attempt: Attempt number

    Returns:
        Cached features dict if exists, None otherwise
    """
    path = get_features_path(base_path, run_id, attempt)
    if not path.exists():
        return None

    with open(path) as f:
        return json.load(f)


def load_cached_symbol(
    base_path: Path,
    run_id: str,
    attempt: int,
    symbol: str,
) -> SymbolSentiment | None:
    """Load cached sentiment for a single symbol if it exists.

    Args:
        base_path: Base data directory
        run_id: Run identifier
        attempt: Attempt number
        symbol: Stock ticker symbol

    Returns:
        SymbolSentiment if cached, None otherwise
    """
    path = get_raw_news_path(base_path, run_id, attempt, symbol)
    if not path.exists():
        return None

    with open(path) as f:
        data = json.load(f)
        return SymbolSentiment.from_dict(data)

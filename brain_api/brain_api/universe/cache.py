"""File-based monthly cache for universe results.

Cache key: {universe_name}_{YYYY-MM}.json
Staleness policy: new calendar month = automatic cache miss.
Write-time cleanup: saving a new entry deletes older files for the same universe.
"""

import json
import logging
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

UNIVERSE_CACHE_DIR = Path("data/cache/universe")


def _month_key(cache_date: date) -> str:
    """Format date as YYYY-MM for monthly cache granularity."""
    return f"{cache_date.year}-{cache_date.month:02d}"


def _cache_path(universe_name: str, cache_date: date) -> Path:
    return UNIVERSE_CACHE_DIR / f"{universe_name}_{_month_key(cache_date)}.json"


def load_cached_universe(
    universe_name: str, cache_date: date | None = None
) -> dict | None:
    """Return cached universe dict, or None on miss.

    Uses monthly granularity: all dates in the same month share one cache file.

    Args:
        universe_name: Identifier like "halal", "halal_filtered", etc.
        cache_date: Date key (defaults to today). Only year+month are used.

    Returns:
        Cached dict if file exists and is valid JSON, else None.
    """
    cache_date = cache_date or date.today()
    path = _cache_path(universe_name, cache_date)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        logger.info(f"Universe cache hit: {path.name}")
        return data
    except (json.JSONDecodeError, OSError):
        logger.warning(f"Universe cache corrupt/unreadable, treating as miss: {path}")
        return None


def save_universe_cache(
    universe_name: str, data: dict, cache_date: date | None = None
) -> None:
    """Write universe dict to a month-keyed JSON file.

    After writing, deletes older cache files for the same universe name.

    Args:
        universe_name: Identifier like "halal", "halal_filtered", etc.
        data: Universe dict to cache.
        cache_date: Date key (defaults to today). Only year+month are used.
    """
    cache_date = cache_date or date.today()
    UNIVERSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(universe_name, cache_date)
    path.write_text(json.dumps(data, default=str))
    logger.info(f"Universe cache saved: {path.name}")
    _cleanup_old_cache(universe_name, keep_date=cache_date)


def _cleanup_old_cache(universe_name: str, keep_date: date) -> None:
    """Delete cache files for this universe whose month differs from keep_date's month."""
    keep_name = _cache_path(universe_name, keep_date).name
    for old in UNIVERSE_CACHE_DIR.glob(f"{universe_name}_*.json"):
        if old.name != keep_name:
            old.unlink(missing_ok=True)
            logger.info(f"Universe cache cleaned up: {old.name}")

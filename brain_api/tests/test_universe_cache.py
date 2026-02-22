"""Tests for universe month-based file cache."""

import json
from datetime import date

import brain_api.universe.cache as cache_mod
from brain_api.universe.cache import (
    _cache_path,
    _cleanup_old_cache,
    load_cached_universe,
    save_universe_cache,
)

SAMPLE_UNIVERSE = {
    "stocks": [
        {"symbol": "AAPL", "name": "Apple Inc", "max_weight": 10.0},
        {"symbol": "MSFT", "name": "Microsoft Corp", "max_weight": 9.5},
    ],
    "total_stocks": 2,
    "fetched_at": "2026-02-18T12:00:00+00:00",
}


# ============================================================================
# load_cached_universe
# ============================================================================


def test_load_returns_none_on_cache_miss():
    assert load_cached_universe("halal", date(2026, 2, 18)) is None


def test_load_returns_none_when_dir_missing():
    assert load_cached_universe("halal_filtered", date(2099, 1, 1)) is None


# ============================================================================
# save + load roundtrip
# ============================================================================


def test_save_then_load_roundtrip():
    d = date(2026, 2, 18)
    save_universe_cache("halal", SAMPLE_UNIVERSE, cache_date=d)
    loaded = load_cached_universe("halal", d)

    assert loaded is not None
    assert loaded["total_stocks"] == 2
    assert loaded["stocks"][0]["symbol"] == "AAPL"


def test_same_month_different_days_share_cache():
    """Feb 5 and Feb 28 should share the same cache file (monthly granularity)."""
    save_universe_cache("halal", SAMPLE_UNIVERSE, cache_date=date(2026, 2, 5))

    loaded_same_day = load_cached_universe("halal", date(2026, 2, 5))
    loaded_later_day = load_cached_universe("halal", date(2026, 2, 28))

    assert loaded_same_day is not None
    assert loaded_later_day is not None
    assert loaded_same_day["total_stocks"] == loaded_later_day["total_stocks"]


def test_different_months_are_independent():
    jan_data = {**SAMPLE_UNIVERSE, "total_stocks": 10}
    feb_data = {**SAMPLE_UNIVERSE, "total_stocks": 20}

    save_universe_cache("halal", jan_data, cache_date=date(2026, 1, 15))
    save_universe_cache("halal", feb_data, cache_date=date(2026, 2, 10))

    assert load_cached_universe("halal", date(2026, 1, 15)) is None  # cleaned up
    loaded = load_cached_universe("halal", date(2026, 2, 10))
    assert loaded is not None
    assert loaded["total_stocks"] == 20


def test_different_universes_are_independent():
    d = date(2026, 2, 18)
    save_universe_cache("halal", SAMPLE_UNIVERSE, cache_date=d)
    assert load_cached_universe("halal_filtered", d) is None


# ============================================================================
# Corrupt / unreadable cache
# ============================================================================


def test_corrupt_json_treated_as_miss():
    d = date(2026, 2, 18)
    cache_dir = cache_mod.UNIVERSE_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path("halal", d)
    path.write_text("{invalid json!!")

    assert load_cached_universe("halal", d) is None


# ============================================================================
# Cache dir auto-creation
# ============================================================================


def test_save_creates_cache_dir():
    """save_universe_cache creates the cache directory if it doesn't exist."""
    cache_dir = cache_mod.UNIVERSE_CACHE_DIR
    assert not cache_dir.exists()
    save_universe_cache("sp500", SAMPLE_UNIVERSE, cache_date=date(2026, 2, 18))
    assert cache_dir.exists()


# ============================================================================
# Write-time cleanup
# ============================================================================


def test_cleanup_deletes_older_month_files_for_same_universe():
    cache_dir = cache_mod.UNIVERSE_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    old_path = _cache_path("halal_filtered", date(2026, 1, 10))
    old_path.write_text(json.dumps(SAMPLE_UNIVERSE))

    save_universe_cache("halal_filtered", SAMPLE_UNIVERSE, cache_date=date(2026, 2, 18))

    assert not old_path.exists()
    assert _cache_path("halal_filtered", date(2026, 2, 18)).exists()


def test_cleanup_does_not_delete_other_universes():
    cache_dir = cache_mod.UNIVERSE_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    halal_path = _cache_path("halal", date(2026, 1, 16))
    halal_path.write_text(json.dumps(SAMPLE_UNIVERSE))

    save_universe_cache("halal_filtered", SAMPLE_UNIVERSE, cache_date=date(2026, 2, 18))

    assert halal_path.exists(), "Cleanup should not touch other universes"


def test_cleanup_halal_does_not_nuke_halal_filtered_or_halal_new():
    """Saving 'halal' must not delete 'halal_filtered' or 'halal_new' caches.

    Regression: glob('halal_*.json') matched halal_new_* and halal_filtered_*.
    """
    cache_dir = cache_mod.UNIVERSE_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    d = date(2026, 2, 18)
    filtered_path = _cache_path("halal_filtered", d)
    new_path = _cache_path("halal_new", d)
    filtered_path.write_text(json.dumps(SAMPLE_UNIVERSE))
    new_path.write_text(json.dumps(SAMPLE_UNIVERSE))

    save_universe_cache("halal", SAMPLE_UNIVERSE, cache_date=d)

    assert filtered_path.exists(), "halal save must not delete halal_filtered cache"
    assert new_path.exists(), "halal save must not delete halal_new cache"


def test_cleanup_called_via_internal_helper():
    cache_dir = cache_mod.UNIVERSE_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    old1 = _cache_path("sp500", date(2025, 11, 1))
    old2 = _cache_path("sp500", date(2025, 12, 15))
    keep = _cache_path("sp500", date(2026, 2, 1))

    for p in (old1, old2, keep):
        p.write_text(json.dumps(SAMPLE_UNIVERSE))

    _cleanup_old_cache("sp500", keep_date=date(2026, 2, 1))

    assert not old1.exists()
    assert not old2.exists()
    assert keep.exists()


# ============================================================================
# Cache path format
# ============================================================================


def test_cache_path_uses_monthly_format():
    path = _cache_path("halal_filtered", date(2026, 2, 19))
    assert path.name == "halal_filtered_2026-02.json"


def test_cache_path_same_month_same_file():
    path_early = _cache_path("halal", date(2026, 3, 1))
    path_late = _cache_path("halal", date(2026, 3, 31))
    assert path_early == path_late


def test_cache_path_different_months_different_files():
    path_feb = _cache_path("halal", date(2026, 2, 15))
    path_mar = _cache_path("halal", date(2026, 3, 15))
    assert path_feb != path_mar


# ============================================================================
# Default cache_date (today)
# ============================================================================


def test_save_and_load_default_to_today():
    save_universe_cache("halal", SAMPLE_UNIVERSE)
    loaded = load_cached_universe("halal")
    assert loaded is not None
    assert loaded["total_stocks"] == SAMPLE_UNIVERSE["total_stocks"]

"""Pytest configuration and fixtures for all tests.

This module ensures tests run in isolation from production environment variables.
"""

import os

import pytest

# HuggingFace-related environment variables that should not affect tests
HF_ENV_VARS = [
    "HF_LSTM_MODEL_REPO",
    "HF_PATCHTST_MODEL_REPO",
    "HF_SAC_MODEL_REPO",  # SAC allocator (unified, dual forecasts)
    "HF_NEWS_SENTIMENT_REPO",
    "HF_TWITTER_SENTIMENT_REPO",
    "HF_DATASET_REPO",
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "STORAGE_BACKEND",
]


@pytest.fixture(autouse=True)
def isolate_from_env():
    """Clear HuggingFace env vars before each test to prevent external API calls.

    This fixture runs automatically for every test (autouse=True).
    It saves original values, clears them for the test, then restores after.
    """
    # Save original values
    original_values = {}
    for var in HF_ENV_VARS:
        if var in os.environ:
            original_values[var] = os.environ.pop(var)

    yield

    # Restore original values after test
    for var, value in original_values.items():
        os.environ[var] = value


@pytest.fixture(autouse=True)
def isolate_universe_cache(tmp_path, monkeypatch):
    """Route universe cache to a temp directory so tests never read/write production cache."""
    monkeypatch.setattr(
        "brain_api.universe.cache.UNIVERSE_CACHE_DIR", tmp_path / "universe_cache"
    )


@pytest.fixture(autouse=True)
def isolate_sticky_history_db(tmp_path, monkeypatch):
    """Route the sticky/screening history DB defaults to a temp file.

    Both ``StickyHistoryRepository`` and ``ScreeningHistoryRepository``
    use ``DEFAULT_DB_PATH`` when no path is passed. Universe builders
    (e.g. ``halal_filtered``) instantiate the repository directly rather
    than via FastAPI dependency injection, so test isolation has to
    happen at the constant level rather than the fixture level. Tests
    that explicitly pass a ``db_path`` (the dedicated repo unit tests)
    are unaffected because the constructor argument wins over the
    module default.
    """
    monkeypatch.setattr(
        "brain_api.storage.sticky_history.DEFAULT_DB_PATH",
        tmp_path / "sticky_history.db",
    )
    monkeypatch.setattr(
        "brain_api.storage.screening_history.DEFAULT_DB_PATH",
        tmp_path / "sticky_history.db",
    )

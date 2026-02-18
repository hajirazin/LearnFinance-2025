"""Pytest configuration and fixtures for all tests.

This module ensures tests run in isolation from production environment variables.
"""

import os

import pytest

# HuggingFace-related environment variables that should not affect tests
HF_ENV_VARS = [
    "HF_LSTM_MODEL_REPO",
    "HF_PATCHTST_MODEL_REPO",
    "HF_PPO_MODEL_REPO",  # PPO allocator (unified, dual forecasts)
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

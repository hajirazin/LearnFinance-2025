"""Pytest configuration and fixtures for all tests.

This module ensures tests run in isolation from production environment variables.
"""

import os

import pytest

# HuggingFace-related environment variables that should not affect tests.
# Includes both the legacy unbucketed names (kept for backwards compat
# with tests that still reference them) and the bucket-keyed names that
# the universe-keyed registry uses today. Without the bucket-keyed
# names, a developer ``.env`` containing ``HF_LSTM_HALAL_NEW_MODEL_REPO``
# would leak into tests and trigger real HF uploads now that training
# routes ungate uploads on policy.
HF_ENV_VARS = [
    # Legacy unbucketed (still cleared for safety)
    "HF_LSTM_MODEL_REPO",
    "HF_PATCHTST_MODEL_REPO",
    "HF_SAC_MODEL_REPO",
    # Bucket-keyed model repos (one per (model, universe) bucket)
    "HF_LSTM_HALAL_NEW_MODEL_REPO",
    "HF_PATCHTST_HALAL_NEW_MODEL_REPO",
    "HF_PATCHTST_NIFTY_SHARIAH_500_MODEL_REPO",
    "HF_SAC_HALAL_FILTERED_MODEL_REPO",
    "HF_SAC_HALAL_MODEL_REPO",
    # Dataset repos + auth + policy switch
    "HF_NEWS_SENTIMENT_REPO",
    "HF_TWITTER_SENTIMENT_REPO",
    "HF_DATASET_REPO",
    "HF_MODEL_REPO",
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "STORAGE_BACKEND",
]


# `STORAGE_BACKEND` is consulted at *import* time by `brain_api.main`
# (boot-time policy validation). Many test modules import `app` at the
# top level, which runs before any pytest fixture; setting an
# explicit valid value here at conftest module load makes sure those
# collections run with the safe default regardless of the developer
# shell env *and* regardless of what their local ``.env`` file says
# (``brain_api.main`` calls ``dotenv.load_dotenv()`` on import, which
# is non-overriding -- so as long as we set ``STORAGE_BACKEND`` here
# first, the ``.env`` value is ignored). Test cases that need a
# specific value still override via ``monkeypatch.setenv``.
_HOST_STORAGE_BACKEND = os.environ.pop("STORAGE_BACKEND", None)
os.environ["STORAGE_BACKEND"] = "local_first"

# Drop bucket-keyed HF model repo envs at conftest module load so a
# developer ``.env`` (loaded by ``brain_api.main`` via ``load_dotenv``)
# cannot reintroduce them. The autouse fixture below also clears them
# per-test, but tests that import ``brain_api.main`` at module top
# (most of the route tests) are bound by import-time env -- not the
# fixture lifecycle. Setting an empty string (rather than ``pop``) is
# what blocks dotenv's non-overriding fallback from re-populating them.
_HF_BUCKET_REPO_ENVS = (
    "HF_LSTM_HALAL_NEW_MODEL_REPO",
    "HF_PATCHTST_HALAL_NEW_MODEL_REPO",
    "HF_PATCHTST_NIFTY_SHARIAH_500_MODEL_REPO",
    "HF_SAC_HALAL_FILTERED_MODEL_REPO",
    "HF_SAC_HALAL_MODEL_REPO",
    "HF_LSTM_MODEL_REPO",
    "HF_PATCHTST_MODEL_REPO",
    "HF_SAC_MODEL_REPO",
    "HF_NEWS_SENTIMENT_REPO",
    "HF_TWITTER_SENTIMENT_REPO",
    "HF_DATASET_REPO",
    "HF_MODEL_REPO",
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
)
_HOST_HF_REPO_ENVS = {
    name: os.environ.pop(name) for name in _HF_BUCKET_REPO_ENVS if name in os.environ
}
for _name in _HF_BUCKET_REPO_ENVS:
    os.environ[_name] = ""


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


def pytest_sessionfinish(session, exitstatus):
    """Restore host env vars (``STORAGE_BACKEND`` + HF repo envs) after the suite."""
    if _HOST_STORAGE_BACKEND is None:
        os.environ.pop("STORAGE_BACKEND", None)
    else:
        os.environ["STORAGE_BACKEND"] = _HOST_STORAGE_BACKEND
    for _name in _HF_BUCKET_REPO_ENVS:
        if _name in _HOST_HF_REPO_ENVS:
            os.environ[_name] = _HOST_HF_REPO_ENVS[_name]
        else:
            os.environ.pop(_name, None)


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

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


# Static halal_new universe used by the autouse network isolation
# fixture below. Must contain enough symbols to satisfy the
# ``min_history`` filter in ``halal_filtered`` plus the LSTM/PatchTST
# default ``n_stocks=15`` slate. Symbols are intentionally short so
# ``compute_model_hash`` digests in the snapshot inventory unit tests
# stay readable.
_FAKE_HALAL_NEW_UNIVERSE: dict = {
    "stocks": [
        {"symbol": f"S{i:02d}", "name": f"Test Stock {i}", "max_weight": 0.05}
        for i in range(20)
    ],
    "etfs_used": ["SPUS", "SPTE", "SPWO", "HLAL", "UMMA"],
    "total_stocks": 20,
    "fetched_at": "2026-01-01T00:00:00+00:00",
}


@pytest.fixture(autouse=True)
def isolate_external_universe_calls(monkeypatch):
    """Block real network access for every universe builder.

    Three pre-existing tests (``test_storage_policy.py::TestEnsureSnapshotForBucketContract::*``
    and ``test_forecaster_snapshots_walkforward.py::TestWalkForwardForecasts::test_build_forecast_features_raises_on_missing_snapshots``)
    transitively invoke ``ensure_snapshot_for_bucket`` ->
    ``lstm_walkforward_expectation_bundle`` ->
    ``halal_new_lstm_resolver_symbols`` ->
    ``get_halal_new_universe`` -> ``fetch_alpaca_tradable_symbols``,
    which hits the real Alpaca API and the SP-Funds / Wahed scrapers.
    Without this fixture each costs ~8 s of real network and silently
    consumes Alpaca quota whenever ``ALPACA_API_KEY`` leaks from
    ``.env`` (the host's ``isolate_from_env`` only clears HF vars).

    Mocks at the universe-builder boundary (``halal_new`` / ``halal``
    / ``nifty_shariah_500``) so any call to a resolver returns the
    static :data:`_FAKE_HALAL_NEW_UNIVERSE` slate instead of touching
    the network. Tests that need a specific symbol slate already
    monkeypatch the rebound resolver name on the consumer module
    (e.g. ``halal_filtered.get_halal_new_universe``); those patches
    stack ON TOP of this autouse layer per pytest's monkeypatch
    semantics, so behaviour is unchanged for any test that already
    does its own bucket binding.

    Also clears ``ALPACA_API_KEY`` / ``ALPACA_API_SECRET`` so a test
    that bypasses the universe mock (e.g. via a bucket rebind that
    keeps the real ``symbols_resolver``) raises a loud
    ``RuntimeError`` instead of silently calling Alpaca.
    """

    def _fake_get_halal_new_universe() -> dict:
        return dict(_FAKE_HALAL_NEW_UNIVERSE)

    def _fake_get_halal_new_symbols() -> list[str]:
        return [s["symbol"] for s in _FAKE_HALAL_NEW_UNIVERSE["stocks"]]

    def _fake_alpaca_tradable_symbols() -> set[str]:
        return {s["symbol"] for s in _FAKE_HALAL_NEW_UNIVERSE["stocks"]} | {
            "SPUS",
            "SPTE",
            "SPWO",
            "HLAL",
            "UMMA",
        }

    monkeypatch.setattr(
        "brain_api.universe.halal_new.get_halal_new_universe",
        _fake_get_halal_new_universe,
    )
    monkeypatch.setattr(
        "brain_api.universe.halal_new.get_halal_new_symbols",
        _fake_get_halal_new_symbols,
    )
    monkeypatch.setattr(
        "brain_api.universe.halal_new.fetch_alpaca_tradable_symbols",
        _fake_alpaca_tradable_symbols,
    )
    # Belt and braces: any leftover direct call into the scrapers
    # (network endpoints) returns an empty list rather than raising,
    # so a test that exercises the merge path without re-mocking sees
    # an empty universe instead of a connection error.
    monkeypatch.setattr(
        "brain_api.universe.halal_new.scrape_sp_funds",
        lambda _slug: [],
    )
    monkeypatch.setattr(
        "brain_api.universe.halal_new.scrape_wahed",
        lambda _slug: [],
    )
    # Final guardrail: clear Alpaca creds so a test that somehow
    # reaches ``fetch_alpaca_tradable_symbols`` directly fails loudly
    # with the RuntimeError it raises when keys are missing.
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)


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


@pytest.fixture(autouse=True)
def isolate_forecaster_snapshots(tmp_path, monkeypatch):
    """Forbid ``SnapshotLocalStorage`` from writing into the production
    data dir.

    ``SnapshotLocalStorage`` defaults its ``base_path`` to
    ``DEFAULT_DATA_PATH = Path("data")`` from
    :mod:`brain_api.storage.base`. The training routes instantiate it
    with no explicit path, so without this fail-safe a test-triggered
    backfill would land snapshots in
    ``brain_api/data/models/<bucket>/snapshot-*/`` next to real
    production artifacts. Routing the rebound name on
    ``brain_api.storage.forecaster_snapshots.local`` (the module that
    actually instantiates the storage) to a per-test tmp dir guarantees
    isolation regardless of whether the test author remembered to
    monkeypatch heavy backfill dependencies.
    """
    monkeypatch.setattr(
        "brain_api.storage.forecaster_snapshots.local.DEFAULT_DATA_PATH",
        tmp_path / "snapshot_data",
    )


@pytest.fixture(autouse=True)
def isolate_experience_storage(tmp_path, monkeypatch):
    """Forbid ``ExperienceStorage`` from writing into the production
    ``data/experience/`` dir.

    The experience route imports ``DEFAULT_DATA_PATH`` at module top
    and uses it as the default ``base_path`` whenever the route
    instantiates ``ExperienceStorage()`` without an explicit path
    (which happens via the FastAPI ``Depends`` chain on every
    ``/experience/*`` call). Without this fail-safe, tests posting to
    those endpoints write ``paper_*_sac.json`` files into
    ``brain_api/data/experience/`` next to real run artifacts. Mirrors
    :func:`isolate_forecaster_snapshots` -- routing the rebound name
    on the route module itself guarantees isolation regardless of
    whether the test author remembered to monkeypatch the storage.
    """
    monkeypatch.setattr(
        "brain_api.routes.experience.DEFAULT_DATA_PATH",
        tmp_path / "experience_data",
    )


@pytest.fixture(autouse=True)
def disable_route_memory_cleanup(monkeypatch):
    """No-op the LSTM/PatchTST training routes' memory-hygiene calls.

    Production training pipelines train on 10-year price tensors and
    free hundreds of MB after ``del dataset, prices``; the routes call
    ``gc.collect()`` (and ``torch.mps.empty_cache`` / ``cuda.empty_cache``
    in the LSTM backfill loop) to keep RSS bounded across a single
    monthly retrain. In tests, the mocked dataset is < 1 KB, so those
    calls free nothing -- yet ``gc.collect()`` still pays a full
    mark-and-sweep over the pytest-accumulated object graph, which the
    debug-mode timing breakdown measured at ~83 ms per call.

    Across the LSTM / PatchTST training tests this dominates the wall
    clock: e.g. ``test_train_lstm_skip_snapshot_false_writes_snapshots``
    pays 1 main-path ``gc.collect`` + 3 backfill iterations (~83 ms x
    4 = 332 ms of the 408 ms total). The
    ``*_not_promoted_when_worse_than_prior`` and
    ``*_current_unchanged_when_not_promoted`` tests pay 4 background
    runs x 83 ms = 332 ms in pure ``gc.collect``.

    ``gc.collect`` / ``mps.empty_cache`` are NOT tested side effects --
    they have no observable behavior contract. Per AGENTS.md "side
    effects mocked, never skipped", we still let ``storage.write_artifacts``
    write real torch artifacts on disk and exercise ``snapshot_storage.
    snapshot_exists_anywhere`` (the actual contracts under test). Only
    the memory-hygiene helpers are no-oped.

    Patches ``gc.collect`` and the torch cache cleaners on the LSTM /
    PatchTST / shared-snapshot-phase modules' rebound names. The
    snapshot helpers were extracted into
    ``brain_api.routes.training.snapshot_phase`` for the AGENTS.md
    600-line file ceiling, so ``gc.collect`` / ``torch.mps.empty_cache``
    inside the backfill loops now resolves on that module rather than
    the original route files. SAC route does not call ``gc.collect``
    so it is not affected.
    """
    import contextlib
    import importlib

    _no_op = lambda: None  # noqa: E731

    for module_path in (
        "brain_api.routes.training.lstm",
        "brain_api.routes.training.patchtst",
        "brain_api.routes.training.snapshot_phase",
    ):
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            continue
        if hasattr(module, "gc"):
            monkeypatch.setattr(module.gc, "collect", _no_op)
        if hasattr(module, "torch"):
            with contextlib.suppress(AttributeError, ImportError):
                monkeypatch.setattr(module.torch.mps, "empty_cache", _no_op)
            with contextlib.suppress(AttributeError, ImportError):
                monkeypatch.setattr(module.torch.cuda, "empty_cache", _no_op)

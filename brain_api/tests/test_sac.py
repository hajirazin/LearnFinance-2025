"""API-level tests for SAC training and inference endpoints (PatchTST-only forecasts).

Tests focus on:
- Endpoint contract (status codes, response structure)
- Constraint enforcement (long-only simplex and cash buffer)
- Promotion gate behavior (Sharpe-first)
"""

import tempfile
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

from brain_api.core.model_buckets import ModelType, get_bucket
from brain_api.core.portfolio_rl.sac_networks import GaussianActor, TwinCritic
from brain_api.core.portfolio_rl.scaler import PortfolioScaler
from brain_api.core.portfolio_rl.state import LEARNED_STATE_DIM, STATE_DIM
from brain_api.core.sac import (
    SACConfig,
    SACTrainingResult,
)
from brain_api.core.sac.regime_hmm import RegimeHMMArtifact
from brain_api.main import app
from brain_api.storage.sac import (
    SACHalalFilteredModelStorage,
    SACV3AuxiliaryArtifacts,
    create_sac_metadata,
)


def _override_sac_bucket(
    monkeypatch,
    temp_storage,
    symbols_fn,
    universe: str = "halal_filtered",
    drop_validator: bool = True,
):
    """Swap a ``(SAC, universe)`` registry entry for tests.

    SAC training resolves storage and symbols inside the endpoint via
    the bucket registry, so tests mutate the registry rather than using
    ``Depends`` overrides. ``monkeypatch.setitem`` restores the original
    bucket at teardown.

    The production halal_filtered bucket pins the slate to exactly 15
    symbols via :func:`_validate_halal_filtered_count`, which would
    reject the small mock slates used here for speed. Tests that want
    to exercise the validator pass ``drop_validator=False``.
    """
    from brain_api.core import model_buckets

    original = get_bucket(ModelType.SAC, universe)
    patched = replace(
        original,
        local_storage_class=lambda: temp_storage,
        symbols_resolver=symbols_fn,
        symbol_validator=None if drop_validator else original.symbol_validator,
    )
    monkeypatch.setitem(
        model_buckets._BUCKETS,
        (ModelType.SAC, universe),
        patched,
    )


# ============================================================================
# Test fixtures and mocks
# ============================================================================


def mock_symbols() -> list[str]:
    """Return the minimum production-eligible canonical slate."""
    return [
        "AAPL",
        "ADBE",
        "AMD",
        "AMZN",
        "AVGO",
        "GOOGL",
        "META",
        "MSFT",
        "NVDA",
        "TSLA",
    ]


def _mock_forecasts(
    symbols, forecaster_type="patchtst", as_of_date=None
) -> dict[str, float]:
    """Deterministic PatchTST stubs for SAC inference feature bundles."""
    del forecaster_type, as_of_date
    return dict.fromkeys(symbols, 0.01)


def _mock_feature_bundle(symbols: list[str] | None = None) -> dict:
    """Canonical PatchTST-only feature bundle for SAC inference tests."""
    symbols = symbols if symbols is not None else mock_symbols()
    return {
        "symbols": symbols,
        "adjusted_closes": {
            symbol: [100.0 + index * 0.1 for index in range(253)] for symbol in symbols
        },
        "news_sentiment": dict.fromkeys(symbols, 0.1),
        "news_article_counts": dict.fromkeys(symbols, 1),
        "patchtst_forecasts": _mock_forecasts(symbols),
        "execution_prices": dict.fromkeys(symbols, 125.2),
        "market_history": [
            {
                "date": value.date().isoformat(),
                "spy_adjusted_close": 621.0 + index,
                "vix_close": 16.0 + index * 0.1,
            }
            for index, value in enumerate(pd.bdate_range("2026-08-03", "2026-08-07"))
        ],
        "provenance": {"test": True},
    }


def mock_config() -> SACConfig:
    """Return a minimal config for fast testing."""
    return SACConfig(
        n_stocks=10,
        total_timesteps=100,  # Very small for fast tests
        hidden_sizes=(16, 16),  # Small network
        batch_size=8,
        warmup_steps=10,
    )


def create_mock_training_result(
    config: SACConfig,
    eval_cagr: float = 0.13,
) -> SACTrainingResult:
    """Create a mock training result for testing.

    Default ``eval_cagr=0.13`` is *above* the 0.12 promotion floor so
    the always-promote-with-guardrails policy promotes by default.
    Tests that need to exercise the rejection path pass an explicit
    sub-floor value (e.g. ``eval_cagr=0.10``).

    Args:
        config: SAC config (``n_stocks`` determines network dims).
        eval_cagr: CAGR to include in the result metadata.
    """
    actor = GaussianActor(
        hidden_sizes=config.hidden_sizes,
        activation=config.activation,
    )

    critic = TwinCritic(
        hidden_sizes=config.hidden_sizes,
        activation=config.activation,
    )

    critic_target = TwinCritic(
        hidden_sizes=config.hidden_sizes,
        activation=config.activation,
    )

    log_alpha = torch.tensor(np.log(0.2), dtype=torch.float32)

    scaler = PortfolioScaler.create(n_stocks=config.n_stocks)
    dummy_states = np.zeros((10, STATE_DIM))
    dummy_states[:, LEARNED_STATE_DIM : LEARNED_STATE_DIM + config.n_stocks] = 1.0
    dummy_states[:, 210] = np.linspace(-0.02, 0.02, 10)
    scaler.fit(dummy_states)

    symbol_order = sorted(mock_symbols()[: config.n_stocks])

    return SACTrainingResult(
        actor=actor,
        critic=critic,
        critic_target=critic_target,
        log_alpha=log_alpha,
        scaler=scaler,
        config=config,
        symbol_order=symbol_order,
        regime_hmm=_mock_regime_hmm(),
        audit_metadata={"sac_schema_version": 3, "architecture": "masked_attention"},
        final_actor_loss=0.1,
        final_critic_loss=0.05,
        avg_episode_return=0.02,
        avg_episode_sharpe=0.5,
        eval_sharpe=0.6,
        eval_cagr=eval_cagr,
        eval_max_drawdown=0.15,
    )


def _mock_regime_hmm() -> RegimeHMMArtifact:
    tail_dates = list(pd.bdate_range(end="2026-07-31", periods=21).date)
    return RegimeHMMArtifact(
        start_probability=np.asarray([0.4, 0.3, 0.3]),
        transition=np.asarray([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]),
        means=np.zeros((3, 4)),
        variances=np.ones((3, 4)),
        scaler_mean=np.zeros(4),
        scaler_scale=np.ones(4),
        label_map={"calm": 0, "transition": 1, "stress": 2},
        terminal_posterior=np.asarray([0.7, 0.2, 0.1]),
        training_cutoff_date=date(2026, 7, 31),
        fit_start_date=date(2020, 1, 1),
        iterations=10,
        log_likelihood=-1.0,
        spy_tail=np.linspace(600.0, 620.0, 21),
        vix_tail=np.linspace(15.0, 17.0, 21),
        tail_dates=tail_dates,
    )


def _v3_auxiliary(result: SACTrainingResult) -> SACV3AuxiliaryArtifacts:
    return SACV3AuxiliaryArtifacts(
        regime_hmm=result.regime_hmm,
        median_patchtst_scaler={
            "mean": result.scaler.median_mean,
            "scale": result.scaler.median_scale,
        },
        audit_metadata=result.audit_metadata,
    )


@pytest.fixture
def temp_storage():
    """Create a temporary storage directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield SACHalalFilteredModelStorage(base_path=tmpdir)


@pytest.fixture
def trained_model_storage():
    """Create storage with a pre-trained model for inference tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = SACHalalFilteredModelStorage(base_path=tmpdir)
        config = mock_config()
        result = create_mock_training_result(config)

        version = "v2025-01-01_test123"
        metadata = create_sac_metadata(
            version=version,
            data_window_start="2020-01-01",
            data_window_end="2025-01-01",
            symbols=result.symbol_order,
            config=config,
            promoted=True,
            prior_version=None,
            actor_loss=result.final_actor_loss,
            critic_loss=result.final_critic_loss,
            avg_episode_return=result.avg_episode_return,
            avg_episode_sharpe=result.avg_episode_sharpe,
            eval_sharpe=result.eval_sharpe,
            eval_cagr=result.eval_cagr,
            eval_max_drawdown=result.eval_max_drawdown,
        )

        storage.write_artifacts(
            version=version,
            actor=result.actor,
            critic=result.critic,
            critic_target=result.critic_target,
            log_alpha=result.log_alpha,
            scaler=result.scaler,
            config=config,
            symbol_order=result.symbol_order,
            metadata=metadata,
            v3_auxiliary=_v3_auxiliary(result),
        )
        storage.promote_version(version)

        yield storage


@pytest.fixture
def inference_client(trained_model_storage, monkeypatch):
    """Create test client with trained model for inference tests.

    SAC inference now resolves storage via the bucket registry, so the
    test seam is :func:`_override_sac_bucket` (replaces the
    ``(SAC, halal_filtered)`` registry entry with a tmpdir-backed
    storage that has mock artifacts written into it). The legacy
    ``app.dependency_overrides[get_inference_storage]`` no longer has
    any effect on the live route.
    """

    def _trained_symbols() -> list[str]:
        return list(
            trained_model_storage.read_metadata(
                trained_model_storage.read_current_version()
            )["symbols"]
        )

    _override_sac_bucket(monkeypatch, trained_model_storage, _trained_symbols)

    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()


# ============================================================================
# Inference endpoint tests
# ============================================================================


class TestSACLSTMInference:
    """Tests for /inference/sac endpoint."""

    def test_inference_returns_valid_weights(self, inference_client):
        """Test that inference returns weights summing to 1."""
        response = inference_client.post(
            "/inference/sac?universe=halal_filtered",
            json={
                "portfolio": {
                    "cash": 10000.0,
                    "positions": [
                        {"symbol": "AAPL", "market_value": 2000.0},
                        {"symbol": "MSFT", "market_value": 2000.0},
                    ],
                },
                "feature_bundle": _mock_feature_bundle(),
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "target_weights" in data
        assert "turnover" in data
        assert "model_version" in data

        # Check weights sum to ~1
        weights = data["target_weights"]
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected ~1.0"

    def test_inference_rejects_partial_decision_date_market_row(self, inference_client):
        feature_bundle = _mock_feature_bundle()
        feature_bundle["market_history"].append(
            {
                "date": "2026-08-10",
                "spy_adjusted_close": 999.0,
                "vix_close": 999.0,
            }
        )
        response = inference_client.post(
            "/inference/sac?universe=halal_filtered",
            json={
                "as_of_date": "2026-08-10",
                "portfolio": {"cash": 10000.0, "positions": []},
                "feature_bundle": feature_bundle,
            },
        )
        assert response.status_code == 422
        assert "extra=['2026-08-10']" in response.json()["detail"]

    def test_inference_respects_cash_buffer(self, inference_client):
        """Test that CASH weight >= cash_buffer (2%)."""
        response = inference_client.post(
            "/inference/sac?universe=halal_filtered",
            json={
                "portfolio": {
                    "cash": 100.0,  # Small cash
                    "positions": [
                        {"symbol": "AAPL", "market_value": 9900.0},
                    ],
                },
                "feature_bundle": _mock_feature_bundle(),
            },
        )

        assert response.status_code == 200
        data = response.json()
        weights = data["target_weights"]

        # CASH weight should be at least 2% (allow tiny float inaccuracy)
        assert weights.get("CASH", 0) >= 0.0199, (
            f"CASH weight {weights.get('CASH')} < 2%"
        )

    def test_off_slate_liquidation_value_is_cash_in_exact_actor_state(
        self, inference_client
    ):
        feature_bundle = _mock_feature_bundle()
        feature_bundle["execution_prices"]["NFLX"] = 90.0
        response = inference_client.post(
            "/inference/sac?universe=halal_filtered",
            json={
                "portfolio": {
                    "cash": 0.0,
                    "positions": [
                        {"symbol": "AAPL", "market_value": 5000.0},
                        {"symbol": "NFLX", "market_value": 5000.0},
                    ],
                },
                "feature_bundle": feature_bundle,
            },
        )

        assert response.status_code == 200
        data = response.json()
        weights = data["decision_state"]["context"]["current_weights"]
        assert weights["AAPL"] == pytest.approx(0.5)
        assert weights["CASH"] == pytest.approx(0.5)
        assert data["forced_liquidations"] == [
            {
                "symbol": "NFLX",
                "market_value": 5000.0,
                "execution_price": 90.0,
                "reason": "outside_active_sac_symbol_set",
            }
        ]
        # True portfolio was 50% AAPL / 50% NFLX / 0% cash. Target sets NFLX
        # to 0, so turnover must include that forced sell (folded cash view
        # alone understates churn whenever cash also moves).
        assert data["turnover"] >= 0.25

    def test_inference_without_model_returns_503(self, temp_storage, monkeypatch):
        """Test that inference without a trained model returns 503.

        Inference resolves storage via the bucket registry now, so the
        test override goes through ``_override_sac_bucket`` rather than
        the legacy ``get_inference_storage`` ``Depends`` shim. The empty
        ``temp_storage`` has no ``current`` pointer, which the
        ``local_first`` policy surfaces as 503 (no local + no HF).
        """
        _override_sac_bucket(monkeypatch, temp_storage, mock_symbols)

        client = TestClient(app)
        response = client.post(
            "/inference/sac?universe=halal_filtered",
            json={
                "portfolio": {
                    "cash": 10000.0,
                    "positions": [],
                },
                "feature_bundle": _mock_feature_bundle(),
            },
        )

        assert response.status_code == 503

    def test_inference_invalid_payload_returns_422(self, inference_client):
        """Test that invalid request payload returns 422."""
        # Missing required portfolio field
        response = inference_client.post(
            "/inference/sac?universe=halal_filtered",
            json={},
        )
        assert response.status_code == 422

    def test_inference_requires_explicit_universe(self, inference_client):
        response = inference_client.post(
            "/inference/sac",
            json={
                "portfolio": {"cash": 10000.0, "positions": []},
                "feature_bundle": _mock_feature_bundle(),
            },
        )
        assert response.status_code == 422
        assert "universe" in response.text

    @pytest.mark.parametrize("invalid_price", [None, 0.0, -1.0, "NaN"])
    def test_positive_off_slate_holding_requires_safe_execution_price(
        self, inference_client, invalid_price
    ):
        feature_bundle = _mock_feature_bundle()
        if invalid_price is not None:
            feature_bundle["execution_prices"]["NFLX"] = invalid_price
        response = inference_client.post(
            "/inference/sac?universe=halal_filtered",
            json={
                "portfolio": {
                    "cash": 5000.0,
                    "positions": [{"symbol": "NFLX", "market_value": 5000.0}],
                },
                "feature_bundle": feature_bundle,
            },
        )
        assert response.status_code == 422
        assert "execution_prices[NFLX]" in response.text


# ============================================================================
# SAC training fixtures
# ============================================================================


def mock_price_loader(symbols, start_date, end_date):
    """Return mock price data for testing."""
    # Daily rows are required because SAC trades at the first XNYS session
    # open of each week. Business-day fixtures contain every requested XNYS
    # session, including holiday-shifted Tuesdays.
    import exchange_calendars as xcals
    import pandas as pd

    dates = (
        xcals.get_calendar("XNYS")
        .sessions_in_range(pd.Timestamp(start_date), pd.Timestamp(end_date))
        .tz_localize(None)
    )
    prices = {}
    for i, symbol in enumerate(symbols):
        base = 100 + i * 10
        prices[symbol] = pd.DataFrame(
            {
                "open": [base] * len(dates),
                "high": [base * 1.01] * len(dates),
                "low": [base * 0.99] * len(dates),
                "close": [base * 1.005] * len(dates),
                "volume": [1000000] * len(dates),
            },
            index=dates,
        )
    return prices


# Tiny SAC config used by full-training tests so the route's
# ``make_sac_config_for_n_stocks(DEFAULT_SAC_CONFIG, len(symbols))``
# resolves to a network/loop that fits in a few hundred ms instead of
# the production ``total_timesteps=10_000`` walk that took ~210s per
# test before this fixture existed. ``n_stocks`` is overridden inside
# ``make_sac_config_for_n_stocks``, so the value here is irrelevant
# beyond satisfying the constructor invariants.
_TINY_SAC_BASE_CONFIG = SACConfig(
    n_stocks=10,
    total_timesteps=100,
    hidden_sizes=(8, 8),
    batch_size=8,
    warmup_steps=10,
)


def _mock_patchtst_forecasts(
    weekly_prices,
    weekly_dates,
    symbols,
    shutdown_event=None,
    target_dates=None,
):
    """Stand-in for :func:`build_patchtst_forecast_features` used by the
    SAC full-training tests so they don't run real walk-forward
    PatchTST fits on top of mock prices.
    """
    n = len(weekly_dates)
    return {s: np.zeros(n) for s in symbols if s in weekly_prices}


_RL_SIGNAL_KEYS: tuple[str, ...] = (
    "news_sentiment",
    "momentum_1w",
    "momentum_4w",
    "momentum_12_1",
    "realized_vol_20d",
)


def _mock_rl_training_signals(
    prices_dict,
    symbols,
    start_date,
    end_date,
    *,
    weekly_cutoffs=None,
) -> dict[str, dict[str, np.ndarray]]:
    """Stand-in for :func:`build_rl_training_signals` used by the SAC
    full-training tests so the route does not hit real
    parquet I/O for historical news sentiment.

    Returns one zero array per ``(symbol, signal)`` pair sized off the
    weekly index of ``prices_dict[symbol]``. Length is intentionally
    generous (``len(weekly)``) so the route's slice-or-pad branch in
    the full-training path takes the slice path with a deterministic
    shape.

    Per AGENTS.md rule: side effects mocked, never skipped. Replaces
    the unmocked ~520ms parquet read timing breakdown.
    """
    signals: dict[str, dict[str, np.ndarray]] = {}
    for symbol in symbols:
        df = prices_dict.get(symbol)
        if df is None or len(df) == 0:
            continue
        weekly_len = (
            len(weekly_cutoffs)
            if weekly_cutoffs is not None
            else len(df["close"].resample("W-FRI").last().dropna())
        )
        n = max(weekly_len, 1)
        signals[symbol] = {key: np.zeros(n) for key in _RL_SIGNAL_KEYS}
    return signals


def _patch_sac_full_training_internals(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock the heavy side effects of ``/train/sac/full``.

    The route's promotion-check, metadata-write, HF-upload-gate, and
    ``complete_job`` flow all still run end-to-end; we only replace
    the three real-compute helpers and resize the bound config:

    * ``load_prices_yfinance`` -> :func:`mock_price_loader`
    * ``DEFAULT_SAC_CONFIG`` -> :data:`_TINY_SAC_BASE_CONFIG`
      (``make_sac_config_for_n_stocks`` ``replace(...)``\\s from
      whatever base is bound on the route module, so this is the
      cheapest seam)
    * ``build_patchtst_forecast_features`` -> :func:`_mock_patchtst_forecasts`
    * ``train_sac`` -> returns :func:`create_mock_training_result`
      with the *resolved* (per-bucket) config so the actor/critic
      shapes match downstream metadata writes.

    Per AGENTS.md rule: side effects mocked, never skipped.
    """
    from brain_api.core.sac.readiness import SACTrainingReadiness
    from brain_api.routes.training.sac import full as sac_full_route

    monkeypatch.setattr(sac_full_route, "load_prices_yfinance", mock_price_loader)
    monkeypatch.setattr(
        sac_full_route,
        "assess_sac_training_readiness",
        lambda universe, *, force=False: SACTrainingReadiness.from_issues(
            universe=universe,
            symbols=mock_symbols(),
            missing=[],
            errors=[],
        ),
    )
    monkeypatch.setattr(sac_full_route, "DEFAULT_SAC_CONFIG", _TINY_SAC_BASE_CONFIG)
    monkeypatch.setattr(
        sac_full_route, "build_patchtst_forecast_features", _mock_patchtst_forecasts
    )
    monkeypatch.setattr(
        sac_full_route, "build_rl_training_signals", _mock_rl_training_signals
    )
    monkeypatch.setattr(
        sac_full_route,
        "train_sac",
        lambda training_data, config, shutdown_event=None: create_mock_training_result(
            config
        ),
    )


# Full training endpoint tests
# ============================================================================


class TestSACFullTraining:
    """Tests for /train/sac/full endpoint."""

    def test_full_training_returns_202_then_200(self, temp_storage, monkeypatch):
        """Test that full training endpoint returns 202 then 200 on rerun."""
        _patch_sac_full_training_internals(monkeypatch)

        app.dependency_overrides.clear()
        _override_sac_bucket(monkeypatch, temp_storage, mock_symbols)

        client = TestClient(app)

        try:
            response1 = client.post("/train/sac/full")
            assert response1.status_code == 202

            response = client.post("/train/sac/full")
            assert response.status_code == 200

            data = response.json()
            assert "version" in data
            assert "data_window_start" in data
            assert "data_window_end" in data
            assert "promoted" in data
            assert "symbols_used" in data
            assert data["promoted"] is True
            assert temp_storage.read_current_version() == data["version"]
        finally:
            app.dependency_overrides.clear()

    def test_legacy_current_metadata_never_short_circuits_v3_training(
        self, temp_storage, monkeypatch
    ):
        """A symbol-matching flat artifact cannot suppress the required rebuild."""
        from brain_api.routes.training.sac import full as sac_full_route

        _patch_sac_full_training_internals(monkeypatch)
        _override_sac_bucket(monkeypatch, temp_storage, mock_symbols)
        monkeypatch.setattr(
            sac_full_route,
            "get_prior_metadata_for_bucket",
            lambda **kwargs: {
                "version": "legacy-flat",
                "symbols": mock_symbols(),
            },
        )

        response = TestClient(app).post("/train/sac/full")
        assert response.status_code == 202

    def test_full_training_is_idempotent(self, temp_storage, monkeypatch):
        """Test that running full training twice with same data produces same version."""
        _patch_sac_full_training_internals(monkeypatch)

        app.dependency_overrides.clear()
        _override_sac_bucket(monkeypatch, temp_storage, mock_symbols)

        client = TestClient(app)

        try:
            response1 = client.post("/train/sac/full")
            assert response1.status_code == 202

            response2 = client.post("/train/sac/full")
            assert response2.status_code == 200
            version2 = response2.json()["version"]

            response3 = client.post("/train/sac/full")
            assert response3.status_code == 200
            version3 = response3.json()["version"]

            assert version2 == version3
        finally:
            app.dependency_overrides.clear()

    def test_full_training_unknown_universe_returns_422(
        self, temp_storage, monkeypatch
    ):
        """Unknown universe in body must be rejected with 422."""
        app.dependency_overrides.clear()
        _override_sac_bucket(monkeypatch, temp_storage, mock_symbols)

        client = TestClient(app)
        try:
            response = client.post(
                "/train/sac/full", json={"universe": "not_a_universe"}
            )
            assert response.status_code == 422
            assert "not_a_universe" in response.text
        finally:
            app.dependency_overrides.clear()

    def test_full_training_halal_filtered_wrong_count_returns_422(
        self, temp_storage, monkeypatch
    ):
        """The halal_filtered bucket pins the slate to exactly 15 symbols.

        Producing a different count silently would build a different-
        shaped SAC actor/critic and break the bucket's compute_version
        hash + ``current`` artifact lineage. Per AGENTS.md rule #1 we
        raise 422 from the bucket symbol_validator; this test exercises
        that path explicitly with the validator left in place.
        """
        app.dependency_overrides.clear()

        def _fourteen_symbols() -> list[str]:
            return [f"SYM{i}" for i in range(14)]

        _override_sac_bucket(
            monkeypatch,
            temp_storage,
            _fourteen_symbols,
            universe="halal_filtered",
            drop_validator=False,
        )

        client = TestClient(app)
        try:
            response = client.post(
                "/train/sac/full", json={"universe": "halal_filtered"}
            )
            assert response.status_code == 422
            assert "exactly 15" in response.text
            assert "got 14" in response.text
        finally:
            app.dependency_overrides.clear()

    def test_full_training_halal_universe_returns_202(self, temp_storage, monkeypatch):
        """Variable-size halal universe (yfinance ETF top-holdings) trains.

        The legacy halal universe has variable size month-to-month
        (typical 12-15 names). The endpoint resolves SAC's ``n_stocks``
        and ``target_entropy`` per-bucket via
        ``make_sac_config_for_n_stocks`` so any count >=5 is accepted;
        no equality check rejects the slate.
        """
        _patch_sac_full_training_internals(monkeypatch)

        app.dependency_overrides.clear()
        _override_sac_bucket(
            monkeypatch,
            temp_storage,
            mock_symbols,
            universe="halal",
        )

        client = TestClient(app)
        try:
            response = client.post("/train/sac/full", json={"universe": "halal"})
            assert response.status_code == 202
        finally:
            app.dependency_overrides.clear()

    def test_full_training_halal_too_few_returns_422(self, temp_storage, monkeypatch):
        """halal bucket lower-bound: <5 symbols must 422 at the validator."""
        app.dependency_overrides.clear()

        def _too_few_symbols() -> list[str]:
            return ["AAPL", "MSFT"]  # 2 < 5

        _override_sac_bucket(
            monkeypatch,
            temp_storage,
            _too_few_symbols,
            universe="halal",
            drop_validator=False,
        )

        client = TestClient(app)
        try:
            response = client.post("/train/sac/full", json={"universe": "halal"})
            assert response.status_code == 422
            assert "at least 5" in response.text
        finally:
            app.dependency_overrides.clear()

    def test_full_training_eval_cagr_below_floor_rejects(
        self, temp_storage, monkeypatch
    ):
        """eval_cagr=0.10 is below the 0.12 floor -> promoted=False.

        Under the old gate, an inaugural model with sub-floor CAGR
        auto-promoted (cold-start fallback). The new policy keeps the
        absolute floor as a guardrail and rejects regardless of prior.
        """
        from brain_api.routes.training.sac import full as sac_full_route

        # Override the SAC bucket BEFORE patching the in-process train_sac
        # mock so the resolved bucket symbols match the test slate.
        app.dependency_overrides.clear()
        _override_sac_bucket(monkeypatch, temp_storage, mock_symbols)

        # Patch the heavy SAC compute helpers, then override train_sac
        # to return a sub-floor result so we exercise the rejection path.
        _patch_sac_full_training_internals(monkeypatch)
        monkeypatch.setattr(
            sac_full_route,
            "train_sac",
            lambda training_data, config, shutdown_event=None: (
                create_mock_training_result(config, eval_cagr=0.10)
            ),
        )

        client = TestClient(app)
        try:
            r1 = client.post("/train/sac/full")
            assert r1.status_code == 202
            status = client.get(f"/train/status/{r1.json()['job_id']}")
            assert status.status_code == 200
            assert status.json()["status"] == "completed"
            data = status.json()["result"]
            assert data["promoted"] is False
            assert any(
                "eval_cagr" in r and "below floor" in r for r in data["failure_reasons"]
            )
            assert temp_storage.read_current_version() is None
        finally:
            app.dependency_overrides.clear()

    def test_force_retrain_rejection_preserves_same_version_active_artifact(
        self, temp_storage, monkeypatch
    ):
        """A rejected force-retrain must leave the prior promoted current intact."""
        from brain_api.routes.training.sac import full as sac_full_route

        app.dependency_overrides.clear()
        _override_sac_bucket(monkeypatch, temp_storage, mock_symbols)
        _patch_sac_full_training_internals(monkeypatch)
        client = TestClient(app)
        try:
            initial = client.post("/train/sac/full")
            assert initial.status_code == 202
            first_status = client.get(f"/train/status/{initial.json()['job_id']}")
            version = first_status.json()["result"]["version"]
            assert first_status.json()["result"]["promoted"] is True
            assert temp_storage.read_current_version() == version
            actor_path = (
                temp_storage.base_path
                / "models"
                / temp_storage.bucket_name
                / version
                / "actor.pt"
            )
            assert actor_path.is_file()

            def _rejected_result(training_data, config, shutdown_event=None):
                del training_data, shutdown_event
                result = create_mock_training_result(config, eval_cagr=0.10)
                with torch.no_grad():
                    for parameter in result.actor.parameters():
                        parameter.fill_(9.0)
                return result

            monkeypatch.setattr(sac_full_route, "train_sac", _rejected_result)
            rejected = client.post("/train/sac/full", json={"force": True})
            assert rejected.status_code == 202
            status = client.get(f"/train/status/{rejected.json()['job_id']}")
            assert status.json()["status"] == "completed"
            assert status.json()["result"]["promoted"] is False
            assert temp_storage.read_current_version() == version
        finally:
            app.dependency_overrides.clear()

    def test_full_training_missing_symbol_returns_fails_readiness(
        self, temp_storage, monkeypatch
    ):
        """A missing slate member's returns fail instead of becoming zeros.

        Halal membership overrides money, so training cannot silently shrink
        the slate when a required symbol has no price history.
        """
        from brain_api.routes.training.sac import full as sac_full_route

        app.dependency_overrides.clear()

        def _resolver_extra() -> list[str]:
            # 6 symbols; the price loader below only covers 5.
            return [*mock_symbols(), "EXTRA1"]

        _override_sac_bucket(monkeypatch, temp_storage, _resolver_extra)
        _patch_sac_full_training_internals(monkeypatch)

        # Drop "EXTRA1" from the price loader output so available_symbols
        # = 5 < expected_symbol_count = 6.
        def _price_loader_drops_extra(symbols, start_date, end_date):
            kept = [s for s in symbols if s != "EXTRA1"]
            return mock_price_loader(kept, start_date, end_date)

        monkeypatch.setattr(
            sac_full_route, "load_prices_yfinance", _price_loader_drops_extra
        )

        client = TestClient(app)
        try:
            r1 = client.post("/train/sac/full")
            assert r1.status_code == 202
            status = client.get(f"/train/status/{r1.json()['job_id']}")
            assert status.status_code == 200
            assert status.json()["status"] == "failed"
            assert "missing price histories: ['EXTRA1']" in status.json()["error"]
            assert temp_storage.read_current_version() is None
        finally:
            app.dependency_overrides.clear()

    def test_full_training_new_window_writes_new_version_and_promotes(
        self, temp_storage, monkeypatch
    ):
        """A forced retrain on a new data window writes+promotes a new version.

        Without ``force=True``, a healthy first run leaves ``current``
        pointing at a matching-symbol v3 artifact, so the symbol-equality
        short-circuit would return 200 instead of retraining.
        """
        from datetime import date as dt_date

        from brain_api.routes.training.sac import full as sac_full_route

        _patch_sac_full_training_internals(monkeypatch)

        app.dependency_overrides.clear()
        _override_sac_bucket(monkeypatch, temp_storage, mock_symbols)

        client = TestClient(app)

        try:
            r1 = client.post("/train/sac/full")
            assert r1.status_code == 202

            first_status = client.get(f"/train/status/{r1.json()['job_id']}")
            v1 = first_status.json()["result"]["version"]
            assert first_status.json()["result"]["promoted"] is True
            assert temp_storage.read_current_version() == v1

            monkeypatch.setattr(
                sac_full_route,
                "resolve_training_window",
                lambda: (dt_date(2015, 1, 2), dt_date(2025, 12, 26)),
            )

            r2 = client.post("/train/sac/full", json={"force": True})
            assert r2.status_code == 202
            second_status = client.get(f"/train/status/{r2.json()['job_id']}")
            v2 = second_status.json()["result"]["version"]
            assert v2 != v1
            assert second_status.json()["result"]["promoted"] is True
            assert temp_storage.read_current_version() == v2
        finally:
            app.dependency_overrides.clear()

    def test_full_training_force_true_bypasses_short_circuit_when_symbols_match(
        self, temp_storage, monkeypatch
    ):
        """force=True bypasses the new short-circuit and re-enters training.

        Same setup as the force=False test, but with ``{"force": true}``
        on the second POST. Because the patched window also produces a
        new deterministic version that doesn't yet exist on disk, the
        existing version-equality short-circuit also misses, so the
        endpoint creates a new background job and returns 202.
        """
        from datetime import date as dt_date

        from brain_api.routes.training.sac import full as sac_full_route

        _patch_sac_full_training_internals(monkeypatch)

        app.dependency_overrides.clear()
        _override_sac_bucket(monkeypatch, temp_storage, mock_symbols)

        client = TestClient(app)

        try:
            r1 = client.post("/train/sac/full")
            assert r1.status_code == 202
            first_status = client.get(f"/train/status/{r1.json()['job_id']}")
            v1 = first_status.json()["result"]["version"]
            assert temp_storage.read_current_version() == v1

            monkeypatch.setattr(
                sac_full_route,
                "resolve_training_window",
                lambda: (dt_date(2015, 1, 2), dt_date(2025, 12, 26)),
            )

            r2 = client.post("/train/sac/full", json={"force": True})
            assert r2.status_code == 202, (
                f"Expected 202 (force=True bypasses symbol-equality and "
                f"new window produces a fresh version), got {r2.status_code}"
            )
            second_status = client.get(f"/train/status/{r2.json()['job_id']}")
            v2 = second_status.json()["result"]["version"]
            assert v2 != v1
            assert second_status.json()["result"]["promoted"] is True
            assert temp_storage.read_current_version() == v2
        finally:
            app.dependency_overrides.clear()

    def test_full_training_force_false_proceeds_when_symbols_differ(
        self, temp_storage, monkeypatch
    ):
        """force=False with a different slate -> training proceeds.

        Trains once with ``mock_symbols`` (5 names), then swaps the
        bucket resolver to a disjoint 5-name slate. The new
        symbol-equality short-circuit must NOT fire (set difference),
        so training restarts and the endpoint returns 202.
        """
        _patch_sac_full_training_internals(monkeypatch)

        app.dependency_overrides.clear()
        _override_sac_bucket(monkeypatch, temp_storage, mock_symbols)

        client = TestClient(app)

        try:
            r1 = client.post("/train/sac/full")
            assert r1.status_code == 202
            first_status = client.get(f"/train/status/{r1.json()['job_id']}")
            v1 = first_status.json()["result"]["version"]
            assert temp_storage.read_current_version() == v1

            def _different_symbols() -> list[str]:
                return [f"ZZZ{i:02d}" for i in range(10)]

            assert set(_different_symbols()).isdisjoint(set(mock_symbols())), (
                "Test precondition: replacement slate must be disjoint "
                "from mock_symbols so symbol-equality cannot fire."
            )

            _override_sac_bucket(monkeypatch, temp_storage, _different_symbols)

            r2 = client.post("/train/sac/full")
            assert r2.status_code == 202, (
                f"Expected 202 (different symbols -> short-circuit misses, "
                f"training restarts), got {r2.status_code}"
            )
            second_status = client.get(f"/train/status/{r2.json()['job_id']}")
            assert second_status.json()["result"]["promoted"] is True
            assert (
                temp_storage.read_current_version()
                == second_status.json()["result"]["version"]
            )
        finally:
            app.dependency_overrides.clear()


# ============================================================================
# Experience endpoint tests
# ============================================================================


class TestExperienceStore:
    """Tests for /experience/store endpoint."""

    def test_store_experience_returns_200(self):
        """Test that storing experience returns 200."""
        client = TestClient(app)

        response = client.post(
            "/experience/store",
            json={
                "run_id": "paper:2025-01-27",
                "week_start": "2025-01-27",
                "week_end": "2025-01-31",
                "model_type": "sac",
                "model_version": "v2025-01-01_test123",
                "state": {
                    "current_weights": {"AAPL": 0.10, "MSFT": 0.08, "CASH": 0.82},
                    "signals": {"AAPL": {"news_sentiment": 0.5}},
                },
                "action": {"AAPL": 0.12, "MSFT": 0.10, "CASH": 0.78},
                "turnover": 0.04,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "record_id" in data
        assert data["stored"] is True

    def test_store_experience_any_model_type_accepted(self):
        """Test that any model_type string is accepted (no strict enum validation)."""
        client = TestClient(app)

        response = client.post(
            "/experience/store",
            json={
                "run_id": "paper:2025-01-27",
                "week_start": "2025-01-27",
                "week_end": "2025-01-31",
                "model_type": "custom_type",  # Any string is accepted
                "model_version": "v2025-01-01_test123",
                "state": {},
                "action": {"CASH": 1.0},
                "turnover": 0.0,
            },
        )

        # API accepts any model_type string (no strict enum validation)
        assert response.status_code == 200


class TestExperienceLabel:
    """Tests for /experience/label endpoint."""

    def test_label_experience_returns_200(self):
        """Test that labeling experience returns 200."""
        client = TestClient(app)

        # First store some experience
        client.post(
            "/experience/store",
            json={
                "run_id": "paper:2025-01-20",
                "week_start": "2025-01-20",
                "week_end": "2025-01-24",
                "model_type": "sac",
                "model_version": "v2025-01-01_test123",
                "state": {"current_weights": {"CASH": 1.0}},
                "action": {"CASH": 1.0},
                "turnover": 0.0,
            },
        )

        # Then try to label it
        response = client.post(
            "/experience/label",
            json={"run_id": "paper:2025-01-20"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "records_labeled" in data
        assert "records_skipped" in data


class TestExperienceList:
    """Tests for /experience/list endpoint."""

    def test_list_experience_returns_200(self):
        """Test that listing experience returns 200."""
        client = TestClient(app)

        response = client.get("/experience/list")

        assert response.status_code == 200
        data = response.json()
        # API returns list directly, not wrapped in "experiences" key
        assert isinstance(data, list)


# ============================================================================
# State dimension and PatchTST forecast validation tests
# ============================================================================


class TestStateDimensionValidation:
    """Tests for state dimension handling in inference."""

    def test_inference_with_all_cash_portfolio(self, inference_client):
        """Test inference with all-cash portfolio (no positions)."""
        response = inference_client.post(
            "/inference/sac?universe=halal_filtered",
            json={
                "portfolio": {
                    "cash": 10000.0,
                    "positions": [],  # All cash, no positions
                },
                "feature_bundle": _mock_feature_bundle(),
            },
        )

        assert response.status_code == 200
        data = response.json()
        weights = data["target_weights"]

        # Should still return valid weights
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01

    def test_inference_with_symbols_not_in_model(self, inference_client):
        """Test inference with portfolio containing symbols not in model."""
        feature_bundle = _mock_feature_bundle()
        feature_bundle["execution_prices"]["UNKNOWN_SYM"] = 25.0
        response = inference_client.post(
            "/inference/sac?universe=halal_filtered",
            json={
                "portfolio": {
                    "cash": 5000.0,
                    "positions": [
                        {"symbol": "AAPL", "market_value": 2500.0},  # In model
                        {
                            "symbol": "UNKNOWN_SYM",
                            "market_value": 2500.0,
                        },  # NOT in model
                    ],
                },
                "feature_bundle": feature_bundle,
            },
        )

        # Off-slate liquidation is safe only because its raw execution price
        # survived validation before the active-only feature bundle was built.
        assert response.status_code == 200


# ============================================================================
# Storage and artifacts tests
# ============================================================================


class TestSACLSTMStorage:
    """Tests for SAC storage functionality."""

    def test_write_and_load_artifacts(self, temp_storage):
        """Test that artifacts can be written and loaded correctly."""
        config = mock_config()
        result = create_mock_training_result(config)

        version = "v2025-01-01_storage_test"
        metadata = create_sac_metadata(
            version=version,
            data_window_start="2020-01-01",
            data_window_end="2025-01-01",
            symbols=result.symbol_order,
            config=config,
            promoted=True,
            prior_version=None,
            actor_loss=result.final_actor_loss,
            critic_loss=result.final_critic_loss,
            avg_episode_return=result.avg_episode_return,
            avg_episode_sharpe=result.avg_episode_sharpe,
            eval_sharpe=result.eval_sharpe,
            eval_cagr=result.eval_cagr,
            eval_max_drawdown=result.eval_max_drawdown,
        )

        # Write artifacts
        temp_storage.write_artifacts(
            version=version,
            actor=result.actor,
            critic=result.critic,
            critic_target=result.critic_target,
            log_alpha=result.log_alpha,
            scaler=result.scaler,
            config=config,
            symbol_order=result.symbol_order,
            metadata=metadata,
            v3_auxiliary=_v3_auxiliary(result),
        )

        # Verify version exists
        assert temp_storage.version_exists(version)

        # Load and verify artifacts
        loaded = temp_storage.load_artifacts(version)
        assert loaded.version == version
        assert loaded.symbol_order == result.symbol_order
        assert isinstance(loaded.actor, GaussianActor)
        assert isinstance(loaded.critic, TwinCritic)
        assert isinstance(loaded.critic_target, TwinCritic)

    def test_promote_version(self, temp_storage):
        """Test version promotion."""
        config = mock_config()
        result = create_mock_training_result(config)

        version = "v2025-01-01_promote_test"
        metadata = create_sac_metadata(
            version=version,
            data_window_start="2020-01-01",
            data_window_end="2025-01-01",
            symbols=result.symbol_order,
            config=config,
            promoted=True,
            prior_version=None,
            actor_loss=0.1,
            critic_loss=0.05,
            avg_episode_return=0.02,
            avg_episode_sharpe=0.5,
            eval_sharpe=0.6,
            eval_cagr=0.10,
            eval_max_drawdown=0.15,
        )

        temp_storage.write_artifacts(
            version=version,
            actor=result.actor,
            critic=result.critic,
            critic_target=result.critic_target,
            log_alpha=result.log_alpha,
            scaler=result.scaler,
            config=config,
            symbol_order=result.symbol_order,
            metadata=metadata,
            v3_auxiliary=_v3_auxiliary(result),
        )

        # Promote version
        temp_storage.promote_version(version)

        # Verify current version
        assert temp_storage.read_current_version() == version

        # Verify can load current artifacts
        loaded = temp_storage.load_current_artifacts()
        assert loaded.version == version

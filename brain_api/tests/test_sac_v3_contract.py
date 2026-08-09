"""Deterministic business-contract tests for the SAC v3 production rebuild."""

from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pytest
import torch

from brain_api.core.portfolio_rl.constraints import compute_turnover_from_allocations
from brain_api.core.portfolio_rl.env import PortfolioEnv
from brain_api.core.portfolio_rl.sac_networks import GaussianActor, TwinCritic
from brain_api.core.portfolio_rl.scaler import MEDIAN_GLOBAL_INDEX, PortfolioScaler
from brain_api.core.portfolio_rl.state import (
    ACTION_DIM,
    ASSET_FEATURES,
    MAX_ASSETS,
    STATE_DIM,
    build_state_vector,
    cross_sectional_rank,
    pack_state,
    unpack_state,
)
from brain_api.core.sac.config import SACConfig
from brain_api.core.sac.decision_context import (
    SACDecisionContextError,
    SACFeatureBundle,
)
from brain_api.core.sac.momentum_signals import compute_realized_vol_20d
from brain_api.core.sac.regime_hmm import (
    RegimeHMMArtifact,
    causal_filter,
    fit_regime_hmm,
    live_market_observations,
    market_observations,
)


def _state(n_valid: int = 12) -> np.ndarray:
    rng = np.random.default_rng(7)
    assets = rng.normal(size=(MAX_ASSETS, ASSET_FEATURES))
    mask = np.zeros(MAX_ASSETS, dtype=bool)
    mask[:n_valid] = True
    globals_ = np.asarray([0.01, 0.6, 0.7, 0.1, 0.2])
    return pack_state(assets, globals_, mask)


def test_state_ranks_ties_and_preserves_raw_patch_globals() -> None:
    assert STATE_DIM == 245
    np.testing.assert_array_equal(cross_sectional_rank(np.asarray([4.0])), [0.0])
    np.testing.assert_allclose(
        cross_sectional_rank(np.asarray([1.0, 2.0, 2.0, 4.0])),
        [-1.0, 0.0, 0.0, 1.0],
    )
    symbols = ["A", "B", "C", "D"]
    signals = {
        symbol: {
            "momentum_1w": value,
            "momentum_4w": value,
            "momentum_12_1": value,
            "news_sentiment": value,
            "realized_vol_20d": value,
        }
        for symbol, value in zip(symbols, (1.0, 2.0, 2.0, 4.0), strict=True)
    }
    forecasts = dict(zip(symbols, (-0.02, 0.01, 0.03, 0.04), strict=True))
    weights = np.zeros(ACTION_DIM)
    weights[-1] = 1.0
    unpacked = unpack_state(build_state_vector(signals, forecasts, weights, symbols))
    np.testing.assert_allclose(unpacked.globals[:2], [0.02, 0.75])
    assert np.all(unpacked.asset_features[4:] == 0.0)


def test_ineligible_held_weight_folds_into_cash_in_observation() -> None:
    symbols = [f"S{index:02d}" for index in range(12)]
    signals = {
        symbol: {
            "momentum_1w": 0.01 + index * 0.001,
            "momentum_4w": 0.02,
            "momentum_12_1": 0.03,
            "news_sentiment": 0.0,
            "realized_vol_20d": 0.2,
        }
        for index, symbol in enumerate(symbols)
    }
    forecasts = dict.fromkeys(symbols, 0.01)
    weights = np.zeros(ACTION_DIM)
    weights[0] = 0.25
    weights[1:12] = 0.05
    weights[-1] = 0.20
    mask = np.zeros(MAX_ASSETS, dtype=bool)
    mask[1:12] = True
    unpacked = unpack_state(
        build_state_vector(
            signals,
            forecasts,
            weights,
            symbols,
            asset_mask=mask,
            regime_probabilities=(0.4, 0.3),
        )
    )
    visible = float(
        unpacked.asset_features[unpacked.asset_mask, 6].sum() + unpacked.globals[-1]
    )
    assert visible == pytest.approx(1.0)
    assert unpacked.globals[-1] == pytest.approx(0.45)
    assert unpacked.asset_features[0, 6] == 0.0


def test_turnover_from_allocations_includes_forced_sells() -> None:
    current = {"AAPL": 0.5, "NFLX": 0.3, "CASH": 0.2}
    target = {"AAPL": 0.6, "CASH": 0.4, "NFLX": 0.0}
    # 0.5 * (|0.6-0.5| + |0-0.3| + |0.4-0.2|) = 0.5 * 0.6 = 0.3
    assert compute_turnover_from_allocations(current, target) == pytest.approx(0.3)


def test_scaler_changes_only_raw_patchtst_median() -> None:
    states = np.vstack((_state(), _state()))
    states[0, MEDIAN_GLOBAL_INDEX] = 0.01
    states[1, MEDIAN_GLOBAL_INDEX] = 0.03
    scaler = PortfolioScaler.create().fit(states)
    transformed = scaler.transform(states)
    changed = np.flatnonzero(np.any(transformed != states, axis=0))
    np.testing.assert_array_equal(changed, [MEDIAN_GLOBAL_INDEX])
    np.testing.assert_allclose(scaler.inverse_transform(transformed), states)


def test_scaler_fits_known_raw_weekly_medians_once() -> None:
    medians = np.asarray([-0.03, 0.01, 0.02, 0.04])
    scaler = PortfolioScaler.create().fit_patchtst_medians(medians)
    assert scaler.median_mean == pytest.approx(0.01)
    assert scaler.median_scale == pytest.approx(np.std(medians, ddof=0))


def test_actor_and_critic_ignore_pads_and_respect_ticker_permutations() -> None:
    torch.manual_seed(11)
    actor = GaussianActor()
    critic = TwinCritic()
    state = torch.tensor(_state(), dtype=torch.float32).unsqueeze(0)
    action, log_prob = actor(state)
    unpacked = unpack_state(state.numpy()[0])
    assert action.shape == (1, ACTION_DIM)
    assert torch.equal(
        action[0, :MAX_ASSETS][~torch.tensor(unpacked.asset_mask)],
        torch.zeros(MAX_ASSETS - int(unpacked.asset_mask.sum())),
    )
    assert torch.isfinite(log_prob).all()

    permutation = torch.randperm(MAX_ASSETS)
    assets = state[:, : MAX_ASSETS * ASSET_FEATURES].reshape(
        1, MAX_ASSETS, ASSET_FEATURES
    )
    globals_ = state[:, MAX_ASSETS * ASSET_FEATURES : MAX_ASSETS * ASSET_FEATURES + 5]
    mask = state[:, -MAX_ASSETS:]
    permuted = torch.cat(
        (assets[:, permutation].reshape(1, -1), globals_, mask[:, permutation]), dim=1
    )
    deterministic, _ = actor(state, deterministic=True)
    permuted_deterministic, _ = actor(permuted, deterministic=True)
    inverse = torch.argsort(permutation)
    torch.testing.assert_close(
        permuted_deterministic[:, :MAX_ASSETS][:, inverse],
        deterministic[:, :MAX_ASSETS],
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(permuted_deterministic[:, -1], deterministic[:, -1])
    q1, q2 = critic(state, action)
    permuted_action = torch.cat(
        (action[:, :MAX_ASSETS][:, permutation], action[:, -1:]), dim=1
    )
    permuted_q1, permuted_q2 = critic(permuted, permuted_action)
    torch.testing.assert_close(q1, permuted_q1, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(q2, permuted_q2, atol=1e-6, rtol=1e-6)


def test_realized_vol_uses_20_log_returns_ddof_one() -> None:
    closes = np.exp(np.linspace(0, 0.2, 21) + np.sin(np.arange(21)) * 0.01)
    expected = np.std(np.diff(np.log(closes)), ddof=1) * np.sqrt(252)
    assert compute_realized_vol_20d(closes, as_of_index=20) == pytest.approx(expected)


def _feature_bundle(
    n_symbols: int = 12, *, missing_held_price: bool = False
) -> SACFeatureBundle:
    symbols = [f"S{index:02d}" for index in range(n_symbols)]
    closes = np.linspace(80.0, 120.0, 274).tolist()
    execution = dict.fromkeys(symbols, 100.0)
    if missing_held_price:
        execution[symbols[0]] = None
    return SACFeatureBundle.create(
        symbols=symbols,
        adjusted_closes=dict.fromkeys(symbols, closes),
        news_sentiment=dict.fromkeys(symbols, 0.0),
        news_article_counts=dict.fromkeys(symbols, 0),
        patchtst_forecasts=dict.fromkeys(symbols, 0.01),
        execution_prices=execution,
        market_history=[],
    )


def test_production_eligibility_allows_12_to_11_but_rejects_below_10() -> None:
    bundle = _feature_bundle()
    weights = dict.fromkeys(bundle.symbols, 0.0)
    weights["CASH"] = 1.0
    mask, _, _ = bundle.eligible_inputs(weights)
    assert mask.sum() == 12
    reduced = SACFeatureBundle.create(
        **{
            **bundle.to_dict(),
            "patchtst_forecasts": {
                **bundle.patchtst_forecasts,
                bundle.symbols[0]: None,
                bundle.symbols[1]: None,
                bundle.symbols[2]: None,
            },
        }
    )
    with pytest.raises(SACDecisionContextError, match="at least 10"):
        reduced.eligible_inputs(weights)


def test_missing_held_execution_price_aborts_rebalance() -> None:
    bundle = _feature_bundle(missing_held_price=True)
    weights = dict.fromkeys(bundle.symbols, 0.0)
    weights[bundle.symbols[0]] = 0.2
    weights["CASH"] = 0.8
    with pytest.raises(SACDecisionContextError, match="held asset"):
        bundle.eligible_inputs(weights)


def test_hmm_fit_is_causal_and_persists_market_tail() -> None:
    rng = np.random.default_rng(42)
    raw = rng.normal(size=(180, 4))
    raw[:, 1] = np.exp(raw[:, 1] - 2)
    raw[:, 2] = np.exp(raw[:, 2] + 3)
    dates = [date(2025, 1, 1) + timedelta(days=index) for index in range(len(raw))]
    tail_spy = np.linspace(90, 110, 21)
    tail_vix = np.linspace(18, 22, 21)
    artifact = fit_regime_hmm(
        raw,
        dates,
        spy_tail=tail_spy,
        vix_tail=tail_vix,
        tail_dates=dates[-21:],
    )
    prefix = causal_filter(raw[:40], artifact)
    extended = causal_filter(raw[:80], artifact)
    np.testing.assert_allclose(prefix, extended[:40], atol=1e-12, rtol=1e-12)
    restored = type(artifact).from_dict(artifact.to_dict())
    np.testing.assert_allclose(restored.terminal_posterior, artifact.terminal_posterior)
    assert len(restored.spy_tail) == 21


def test_hmm_first_emission_and_terminal_continuation_are_distinct() -> None:
    start = np.asarray([0.8, 0.1, 0.1])
    transition = np.asarray([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    terminal = np.asarray([0.2, 0.3, 0.5])
    means = np.asarray([[0.0] * 4, [1.0] * 4, [2.0] * 4])
    artifact = RegimeHMMArtifact(
        start_probability=start,
        transition=transition,
        means=means,
        variances=np.ones((3, 4)),
        scaler_mean=np.zeros(4),
        scaler_scale=np.ones(4),
        label_map={"calm": 0, "transition": 1, "stress": 2},
        terminal_posterior=terminal,
        training_cutoff_date=date(2026, 8, 7),
        fit_start_date=date(2025, 1, 1),
        iterations=2,
        log_likelihood=-1.0,
        spy_tail=np.ones(21),
        vix_tail=np.full(21, 20.0),
        tail_dates=tuple(date(2026, 7, 10) + timedelta(days=i) for i in range(21)),
    )
    observation = np.asarray([[0.0, 0.0, 1.0, 0.0]])
    # Common Gaussian constants cancel during normalization.
    likelihood = np.exp(-0.5 * np.sum((observation[0] - means) ** 2, axis=1))
    expected_initial = start * likelihood
    expected_initial /= expected_initial.sum()
    expected_continuation = (terminal @ transition) * likelihood
    expected_continuation /= expected_continuation.sum()

    np.testing.assert_allclose(
        causal_filter(observation, artifact)[0], expected_initial
    )
    np.testing.assert_allclose(
        causal_filter(observation, artifact, terminal)[0], expected_continuation
    )
    assert not np.allclose(expected_initial, expected_continuation)


def test_live_market_history_requires_exact_completed_xnys_range() -> None:
    artifact = RegimeHMMArtifact(
        start_probability=np.full(3, 1 / 3),
        transition=np.eye(3),
        means=np.zeros((3, 4)),
        variances=np.ones((3, 4)),
        scaler_mean=np.zeros(4),
        scaler_scale=np.ones(4),
        label_map={"calm": 0, "transition": 1, "stress": 2},
        terminal_posterior=np.full(3, 1 / 3),
        training_cutoff_date=date(2026, 8, 7),
        fit_start_date=date(2025, 1, 1),
        iterations=2,
        log_likelihood=-1.0,
        spy_tail=np.linspace(100.0, 120.0, 21),
        vix_tail=np.linspace(15.0, 17.0, 21),
        tail_dates=tuple(date(2026, 7, 10) + timedelta(days=i) for i in range(21)),
    )
    observations, dates = live_market_observations(artifact, [], date(2026, 8, 10))
    assert observations.shape == (0, 4)
    assert dates == []

    monday_partial = [
        {"date": "2026-08-10", "spy_adjusted_close": 121.0, "vix_close": 18.0}
    ]
    with pytest.raises(ValueError, match=r"extra=\['2026-08-10'\]"):
        live_market_observations(artifact, monday_partial, date(2026, 8, 10))

    stale_artifact = replace(artifact, training_cutoff_date=date(2026, 8, 3))
    with pytest.raises(ValueError, match="2026-08-04"):
        live_market_observations(stale_artifact, [], date(2026, 8, 10))


def test_market_observation_formula_uses_adjusted_spy_and_positive_vix() -> None:
    spy = np.linspace(100, 130, 25)
    vix = np.linspace(15, 20, 25)
    rows = market_observations(spy, vix)
    assert rows.shape == (5, 4)
    assert rows[0, 0] == pytest.approx(spy[20] / spy[0] - 1)
    assert rows[0, 3] == pytest.approx(vix[20] / vix[15] - 1)


def test_variable_slate_environment_uses_fixed_actions_without_costing_pads() -> None:
    n_stocks = 10
    env = PortfolioEnv(
        symbol_returns=np.full((2, n_stocks), 0.001),
        signals=np.zeros((2, n_stocks, 5)),
        patchtst_forecasts=np.zeros((2, n_stocks)),
        prices=np.full((2, n_stocks), 100.0),
        symbol_order=[f"S{index:02d}" for index in range(n_stocks)],
        config=SACConfig(n_stocks=n_stocks),
        asset_masks=np.ones((2, n_stocks), dtype=bool),
    )
    state = env.reset(start_week=0)
    assert state.shape == (STATE_DIM,)
    result = env.step(np.zeros(ACTION_DIM))
    assert result.info["target_weights"][10:30] == [0.0] * 20
    assert np.isfinite(result.reward)

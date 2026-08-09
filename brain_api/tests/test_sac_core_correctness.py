"""Deterministic business-logic tests for SAC accounting and actor state."""

from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from brain_api.core.portfolio_rl.constraints import apply_softmax_to_weights
from brain_api.core.portfolio_rl.rewards import (
    RebalanceTransition,
    compute_net_log_reward,
)
from brain_api.core.portfolio_rl.sac_networks import GaussianActor
from brain_api.core.portfolio_rl.sac_trainer import SACTrainer
from brain_api.core.portfolio_rl.state import (
    ACTION_DIM,
    STATE_DIM,
    StateSchema,
    build_state_vector,
)
from brain_api.core.sac.config import SACConfig
from brain_api.core.sac.training import (
    TrainingData,
    build_training_data,
    evaluate_policy,
    train_sac,
)


def _patchtst_inputs() -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    signals = {
        "AAA": {
            "news_sentiment": 0.1,
            "momentum_1w": 0.01,
            "momentum_4w": 0.02,
            "momentum_12_1": 0.10,
            "realized_vol_20d": 0.2,
        }
    }
    return signals, {"AAA": 0.03}


def test_state_schema_dimension_and_strict_construction() -> None:
    signals, patchtst = _patchtst_inputs()
    state = build_state_vector(signals, patchtst, np.array([0.8, 0.2]), ["AAA"])

    assert StateSchema(n_stocks=15).n_forecasts_per_stock == 1
    assert StateSchema(n_stocks=15).state_dim == STATE_DIM
    assert state.shape == (245,)
    assert state[-30] == 1.0


@pytest.mark.parametrize(
    ("missing_from", "expected"),
    [
        ("signal", "realized_vol_20d"),
        ("patchtst", "observations"),
    ],
)
def test_state_vector_never_zero_fills_missing_actor_inputs(
    missing_from: str, expected: str
) -> None:
    signals, patchtst = _patchtst_inputs()
    if missing_from == "signal":
        signals["AAA"].pop("realized_vol_20d")
    else:
        patchtst.clear()

    with pytest.raises(ValueError, match=expected):
        build_state_vector(signals, patchtst, np.array([0.8, 0.2]), ["AAA"])


def test_state_vector_rejects_empty_signals() -> None:
    with pytest.raises(ValueError, match="observations"):
        build_state_vector(
            {}, {}, np.array([0.0, 1.0]), ["AAA"], StateSchema(n_stocks=1)
        )


def test_rebalance_transition_uses_exact_net_growth_and_drift() -> None:
    target = np.array([0.6, 0.3, 0.1])
    returns = np.array([0.10, -0.05])
    transition = RebalanceTransition.calculate(target, returns, 0.002)

    gross = 0.6 * 0.10 + 0.3 * -0.05
    net_growth = 1 + gross - 0.002
    expected = np.array(
        [
            0.6 * 1.10 / net_growth,
            0.3 * 0.95 / net_growth,
            (0.1 - 0.002) / net_growth,
        ]
    )
    assert transition.gross_return == pytest.approx(gross)
    assert transition.net_log_return == pytest.approx(np.log(net_growth))
    assert transition.post_weights == pytest.approx(expected)
    assert transition.post_weights.sum() == pytest.approx(1.0)


def test_exact_reward_is_log_one_plus_gross_minus_cost() -> None:
    config = SACConfig(n_stocks=2, reward_scale=100.0, hhi_penalty_scale=0.0)
    reward = compute_net_log_reward(0.04, 0.01, config)
    assert reward == pytest.approx(np.log(1.03) * 100.0)
    assert reward != pytest.approx((np.log(1.04) - np.log(1.01)) * 100.0)


def test_training_data_requires_complete_inputs_and_uses_trade_time_price() -> None:
    signals, patchtst = _patchtst_inputs()
    symbols = [f"S{index:02d}" for index in range(10)]
    price_inputs = {symbol: np.array([100.0, 110.0, 99.0]) for symbol in symbols}
    signal_arrays = {
        symbol: {
            name: np.array([value, value]) for name, value in signals["AAA"].items()
        }
        for symbol in symbols
    }
    data = build_training_data(
        prices=price_inputs,
        signals=signal_arrays,
        patchtst_predictions={
            symbol: np.array([patchtst["AAA"], patchtst["AAA"]]) for symbol in symbols
        },
        symbol_order=symbols,
    )
    assert data.prices[:, 0] == pytest.approx([100.0, 110.0])
    assert data.symbol_returns[:, 0] == pytest.approx([0.10, -0.10])

    broken = dict(signal_arrays[symbols[0]])
    broken.pop("momentum_4w")
    with pytest.raises(ValueError, match="momentum_4w"):
        build_training_data(
            price_inputs,
            {**signal_arrays, symbols[0]: broken},
            {symbol: np.array([0.1, 0.1]) for symbol in symbols},
            symbols,
        )


def test_tanh_softmax_has_bounded_concentration() -> None:
    logits = np.array([1.0, *([-1.0] * 15)])
    weights = apply_softmax_to_weights(logits)
    theoretical_max = np.e**2 / (np.e**2 + 15)
    assert weights.max() == pytest.approx(theoretical_max)
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(weights > 0)


def test_gaussian_actor_log_prob_includes_tanh_jacobian() -> None:
    actor = GaussianActor(hidden_sizes=(16, 16))
    signals, patchtst = _patchtst_inputs()
    state = build_state_vector(signals, patchtst, np.array([0.8, 0.2]), ["AAA"])
    torch.manual_seed(42)
    action, log_prob = actor(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
    assert torch.isfinite(log_prob).all()
    assert action.shape == (1, ACTION_DIM)
    assert torch.count_nonzero(action[0, 1:30]) == 0


def test_policy_evaluation_is_chronological_and_uses_104_net_returns() -> None:
    class Actor:
        def get_action(self, state: np.ndarray, deterministic: bool) -> np.ndarray:
            assert deterministic
            return np.array([0.0])

    class EvalEnv:
        def __init__(self) -> None:
            self.index = 0
            self.reset_args: list[int | None] = []

        def reset(self, start_week: int | None = None) -> np.ndarray:
            self.reset_args.append(start_week)
            self.index = 0
            return np.array([0.0])

        def step(self, action: np.ndarray) -> SimpleNamespace:
            del action
            self.index += 1
            return SimpleNamespace(
                next_state=np.array([float(self.index)]),
                done=self.index == 104,
                portfolio_return=0.50,
                info={"net_portfolio_return": 0.001},
            )

    env = EvalEnv()
    _sharpe, cagr, _drawdown = evaluate_policy(
        Actor(), env, SACConfig(), expected_periods=104
    )
    assert env.reset_args == [0]
    assert env.index == 104
    assert cagr == pytest.approx((1.001**104) ** 0.5 - 1)


def test_trainer_result_reports_tracked_optimizer_losses() -> None:
    trainer = object.__new__(SACTrainer)
    trainer.actor = object()
    trainer.critic = object()
    trainer.critic_target = object()
    trainer.log_alpha = torch.tensor(0.0)
    trainer.final_actor_loss = 1.25
    trainer.final_critic_loss = 2.5
    trainer.episode_returns = [0.1]
    trainer.episode_sharpes = [0.2]

    result = trainer.get_result()

    assert result.final_actor_loss == pytest.approx(1.25)
    assert result.final_critic_loss == pytest.approx(2.5)


def test_training_scaler_uses_every_training_week_median_once(monkeypatch) -> None:
    from brain_api.core.sac import training as training_module

    fitted_samples: list[np.ndarray] = []

    class FakeEnv:
        action_dim = ACTION_DIM
        state_dim = STATE_DIM

        def reset(self, start_week=None):
            del start_week
            return np.zeros(STATE_DIM)

        def step(self, action):
            next_state = np.zeros(STATE_DIM)
            next_state[0] = float(np.sum(action))
            return SimpleNamespace(next_state=next_state, done=False)

    class FakeScaler:
        def fit_patchtst_medians(self, medians):
            fitted_samples.append(medians.copy())
            return self

        def transform(self, states):
            return states

    class FakeTrainer:
        def __init__(self, env, config, shutdown_event=None):
            del env, config, shutdown_event

        def train(self, total_timesteps):
            del total_timesteps

        def get_result(self):
            return SimpleNamespace(
                actor=object(),
                critic=object(),
                critic_target=object(),
                log_alpha=torch.tensor(0.0),
                final_actor_loss=0.1,
                final_critic_loss=0.2,
                avg_episode_return=0.3,
                avg_episode_sharpe=0.4,
            )

    monkeypatch.setattr(
        training_module,
        "create_env_from_training_data",
        lambda *args, **kwargs: FakeEnv(),
    )
    monkeypatch.setattr(
        training_module.PortfolioScaler,
        "create",
        lambda **kwargs: FakeScaler(),
    )
    monkeypatch.setattr(training_module, "SACTrainer", FakeTrainer)
    monkeypatch.setattr(
        training_module,
        "evaluate_policy",
        lambda *args, **kwargs: (0.5, 0.2, -0.1),
    )
    fake_hmm = SimpleNamespace(terminal_posterior=np.full(3, 1 / 3))
    monkeypatch.setattr(
        training_module,
        "market_observations",
        lambda spy, vix: np.zeros((len(spy) - 20, 4)),
    )
    monkeypatch.setattr(
        training_module, "fit_regime_hmm", lambda *args, **kwargs: fake_hmm
    )
    monkeypatch.setattr(
        training_module,
        "causal_filter",
        lambda observations, *args, **kwargs: np.full((len(observations), 3), 1 / 3),
    )
    monkeypatch.setattr(
        training_module, "regime_probabilities", lambda *args: (1 / 3, 1 / 3)
    )
    market_dates = [date(2020, 1, 1) + timedelta(days=index) for index in range(228)]
    forecasts = np.arange(208, dtype=float).reshape(-1, 1)
    data = TrainingData(
        symbol_returns=np.zeros((208, 1)),
        signals=np.zeros((208, 1, 5)),
        patchtst_forecasts=forecasts,
        prices=np.ones((208, 1)),
        asset_masks=np.ones((208, 1), dtype=bool),
        symbol_order=["AAA"],
        n_weeks=208,
        n_stocks=1,
        weekly_dates=market_dates[20:],
        market_dates=market_dates,
        spy_adjusted_closes=np.ones(228),
        vix_closes=np.ones(228),
    )
    config = SACConfig(n_stocks=1, seed=123, total_timesteps=1)

    train_sac(data, config)

    assert len(fitted_samples) == 1
    train_weeks = 104
    np.testing.assert_array_equal(fitted_samples[0], forecasts[:train_weeks, 0])

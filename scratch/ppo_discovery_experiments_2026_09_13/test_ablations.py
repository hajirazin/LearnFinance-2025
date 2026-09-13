"""Ablation metrics must use economic weekly log return, not shaped reward."""

from __future__ import annotations

from types import SimpleNamespace

import ablations as ablation_module
import pytest
import torch

from brain_api.core.ppo_discovery.config import PPODiscoveryConfig
from brain_api.core.ppo_discovery.diagnostics import (
    compute_portfolio_transition_diagnostics,
)
from brain_api.core.ppo_discovery.policy import PPODiscoveryActorCritic


def test_ablation_metrics_use_realized_net_return(monkeypatch) -> None:
    diagnostics = compute_portfolio_transition_diagnostics(
        {"CASH": 1.0},
        {"CASH": 1.0},
        transaction_cost_fraction=0.0,
    )
    steps = [
        SimpleNamespace(
            reward=99.0,
            realized_net_return=0.01,
            diagnostics=diagnostics,
            state=SimpleNamespace(as_of="2026-01-05"),
        ),
        SimpleNamespace(
            reward=-50.0,
            realized_net_return=0.02,
            diagnostics=diagnostics,
            state=SimpleNamespace(as_of="2026-01-12"),
        ),
    ]
    captured: dict[str, list[float]] = {}

    def fake_eval(logs):
        captured["logs"] = [float(value) for value in logs]
        return {
            "cagr": 0.1,
            "vol": 0.1,
            "sharpe": 1.0,
            "max_drawdown": 0.0,
            "n_weeks": 2.0,
        }

    monkeypatch.setattr(
        ablation_module, "collect_closed_loop_rollout", lambda *args, **kwargs: steps
    )
    monkeypatch.setattr(ablation_module, "evaluate_policy_weeks", fake_eval)
    payload = ablation_module._metrics(None, [], None, None, None, None, None)
    assert captured["logs"] == [0.01, 0.02]
    assert payload["status"] == "ok"
    assert payload["portfolio_diagnostics"]["summary"]["n_weeks"] == 2.0
    assert payload["portfolio_diagnostics"]["weekly"][0]["as_of"] == "2026-01-05"


def _tiny_ablation_config() -> PPODiscoveryConfig:
    return PPODiscoveryConfig(
        dropout=0.0,
        total_timesteps=4,
        ppo_epochs=1,
        minibatch_size=2,
        ppo_microbatch_size=2,
        seeds=(42,),
    )


def test_retrain_ablation_uses_candidate_device(monkeypatch) -> None:
    device = torch.device("cpu")
    config = _tiny_ablation_config()
    start = PPODiscoveryActorCritic(config).to(device)
    captured: dict[str, object] = {}

    def fake_train(policy, episode_fn, *, config, seed, **kwargs):
        del policy, episode_fn, config, seed
        captured["device"] = kwargs.get("device")
        return {"ppo_loss": 0.0}

    monkeypatch.setattr(ablation_module, "train_ppo_discovery", fake_train)
    monkeypatch.setattr(
        ablation_module,
        "_metrics",
        lambda *args, **kwargs: {"status": "ok", "cagr": 0.1},
    )
    payload = ablation_module._retrain_ablation(
        start,
        [],
        [],
        {"snapshot": None, "ohlcv": None, "spy": None, "scalers": None},
        config,
        freeze_encoder_updates=1,
        device=device,
    )
    assert captured["device"] == device
    assert payload["status"] == "ok"


def test_retrain_ablation_reraises_accelerator_oom(monkeypatch) -> None:
    device = torch.device("cpu")
    config = _tiny_ablation_config()
    start = PPODiscoveryActorCritic(config).to(device)

    def boom(*args, **kwargs):
        raise MemoryError("mps allocator oom")

    monkeypatch.setattr(ablation_module, "train_ppo_discovery", boom)
    with pytest.raises(MemoryError, match="mps allocator oom"):
        ablation_module._retrain_ablation(
            start,
            [],
            [],
            {"snapshot": None, "ohlcv": None, "spy": None, "scalers": None},
            config,
            freeze_encoder_updates=1,
            device=device,
        )


def test_eval_ablation_oom_is_not_converted_to_failed(monkeypatch) -> None:
    config = _tiny_ablation_config()
    candidate = PPODiscoveryActorCritic(config)

    def boom(*args, **kwargs):
        raise MemoryError("cuda oom")

    monkeypatch.setattr(ablation_module, "_metrics", boom)
    with pytest.raises(MemoryError, match="cuda oom"):
        ablation_module.run_required_ablations(
            candidate,
            train_weeks=[],
            test_weeks=[],
            snapshot=None,
            ohlcv=None,
            spy=None,
            scalers=None,
            config=config,
            pretrained=candidate,
        )


def test_no_transaction_cost_ablation_retrains_with_costs_off(monkeypatch) -> None:
    device = torch.device("cpu")
    config = _tiny_ablation_config()
    start = PPODiscoveryActorCritic(config).to(device)
    recorded: list[object] = []

    def fake_rollout(*args, **kwargs):
        recorded.append(kwargs.get("include_transaction_cost"))
        return []

    def fake_train(policy, episode_fn, *, config, seed, **kwargs):
        del policy, config, seed, kwargs
        episode_fn(start, None)
        return {"ppo_loss": 0.0}

    monkeypatch.setattr(ablation_module, "collect_closed_loop_rollout", fake_rollout)
    monkeypatch.setattr(ablation_module, "train_ppo_discovery", fake_train)
    monkeypatch.setattr(
        ablation_module,
        "_metrics",
        lambda *args, **kwargs: {"status": "ok", "cagr": 0.1},
    )
    ablation_module._retrain_ablation(
        start,
        [],
        [],
        {"snapshot": None, "ohlcv": None, "spy": None, "scalers": None},
        config,
        freeze_encoder_updates=1,
        device=device,
        train_include_transaction_cost=False,
        eval_include_transaction_cost=True,
    )
    assert recorded == [False]


def test_no_transaction_cost_eval_uses_locked_costs(monkeypatch) -> None:
    device = torch.device("cpu")
    config = _tiny_ablation_config()
    start = PPODiscoveryActorCritic(config).to(device)
    eval_flags: list[object] = []

    def fake_metrics(*args, **kwargs):
        eval_flags.append(kwargs.get("include_transaction_cost"))
        return {"status": "ok", "cagr": 0.1}

    monkeypatch.setattr(
        ablation_module, "train_ppo_discovery", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(ablation_module, "_metrics", fake_metrics)
    ablation_module._retrain_ablation(
        start,
        [],
        [],
        {"snapshot": None, "ohlcv": None, "spy": None, "scalers": None},
        config,
        freeze_encoder_updates=1,
        device=device,
        train_include_transaction_cost=False,
        eval_include_transaction_cost=True,
    )
    assert eval_flags == [True]

"""Ablation metrics must use economic weekly log return, not shaped reward."""

from __future__ import annotations

from types import SimpleNamespace

from brain_api.core.ppo_discovery import ablations as ablation_module


def test_ablation_metrics_use_realized_net_return(monkeypatch) -> None:
    steps = [
        SimpleNamespace(reward=99.0, realized_net_return=0.01),
        SimpleNamespace(reward=-50.0, realized_net_return=0.02),
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

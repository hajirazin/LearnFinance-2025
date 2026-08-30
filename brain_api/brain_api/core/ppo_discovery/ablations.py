"""Required ppo_discovery ablations. Unavailable is not a passing status."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from brain_api.core.ppo_discovery.config import PPODiscoveryConfig
from brain_api.core.ppo_discovery.environment import collect_closed_loop_rollout
from brain_api.core.ppo_discovery.evaluator import evaluate_policy_weeks, mark_ablations
from brain_api.core.ppo_discovery.policy import PPODiscoveryActorCritic
from brain_api.core.ppo_discovery.trainer import train_ppo_discovery
from brain_api.core.training_utils import is_accelerator_out_of_memory


def _metrics(
    policy, weeks, snapshot, ohlcv, spy, scalers, config, **kwargs
) -> dict[str, Any]:
    steps = collect_closed_loop_rollout(
        policy,
        weeks,
        snapshot=snapshot,
        ohlcv_by_symbol=ohlcv,
        spy=spy,
        feature_scalers=scalers,
        config=config,
        deterministic=True,
        **kwargs,
    )
    payload = evaluate_policy_weeks([step.realized_net_return for step in steps])
    payload["status"] = "ok"
    return payload


def run_required_ablations(
    candidate: PPODiscoveryActorCritic,
    *,
    train_weeks,
    test_weeks,
    snapshot,
    ohlcv,
    spy,
    scalers,
    config: PPODiscoveryConfig,
    pretrained: PPODiscoveryActorCritic,
) -> dict[str, Any]:
    """Evaluate every required ablation. Failures are status=failed, never unavailable."""
    device = next(candidate.parameters()).device
    available: dict[str, Any] = {}
    eval_kw = {
        "snapshot": snapshot,
        "ohlcv": ohlcv,
        "spy": spy,
        "scalers": scalers,
        "config": config,
    }
    try:
        available["full_ppo"] = _metrics(candidate, test_weeks, **eval_kw)
    except Exception as exc:
        _reraise_oom(exc, device)
        available["full_ppo"] = {"status": "failed", "error": str(exc)}
    try:
        available["no_news_features"] = _metrics(
            candidate, test_weeks, **eval_kw, zero_news_features=True
        )
    except Exception as exc:
        _reraise_oom(exc, device)
        available["no_news_features"] = {"status": "failed", "error": str(exc)}
    try:
        shuffled = _shuffle_news(test_weeks)
        available["news_time_shuffled"] = _metrics(candidate, shuffled, **eval_kw)
    except Exception as exc:
        _reraise_oom(exc, device)
        available["news_time_shuffled"] = {"status": "failed", "error": str(exc)}
    try:
        available["no_temporal_encoder"] = _metrics(
            candidate, test_weeks, **eval_kw, zero_history=True
        )
    except Exception as exc:
        _reraise_oom(exc, device)
        available["no_temporal_encoder"] = {"status": "failed", "error": str(exc)}
    try:
        available["fixed_k_15"] = _metrics(candidate, test_weeks, **eval_kw, force_k=15)
    except Exception as exc:
        _reraise_oom(exc, device)
        available["fixed_k_15"] = {"status": "failed", "error": str(exc)}
    try:
        available["equal_weight_selected"] = _metrics(
            candidate, test_weeks, **eval_kw, equal_weight_selected=True
        )
    except Exception as exc:
        _reraise_oom(exc, device)
        available["equal_weight_selected"] = {"status": "failed", "error": str(exc)}
    try:
        available["no_hmm_globals"] = _metrics(
            candidate, test_weeks, **eval_kw, zero_hmm=True
        )
    except Exception as exc:
        _reraise_oom(exc, device)
        available["no_hmm_globals"] = {"status": "failed", "error": str(exc)}
    try:
        available["no_transaction_cost_term"] = _metrics(
            candidate, test_weeks, **eval_kw, include_transaction_cost=False
        )
    except Exception as exc:
        _reraise_oom(exc, device)
        available["no_transaction_cost_term"] = {"status": "failed", "error": str(exc)}
    available["frozen_pretrained_encoder"] = _retrain_ablation(
        pretrained,
        train_weeks,
        test_weeks,
        eval_kw,
        config,
        freeze_encoder_updates=10**9,
        device=device,
    )
    available["no_supervised_pretraining"] = _retrain_ablation(
        PPODiscoveryActorCritic(config).to(device),
        train_weeks,
        test_weeks,
        eval_kw,
        config,
        freeze_encoder_updates=config.freeze_encoder_updates,
        device=device,
    )
    return mark_ablations(available)


def _shuffle_news(weeks):
    if len(weeks) < 2:
        return list(weeks)
    rotated = [weeks[-1].news_by_symbol, *[week.news_by_symbol for week in weeks[:-1]]]
    shuffled = []
    for week, news in zip(weeks, rotated, strict=True):
        shuffled.append(
            week.__class__(
                cutoff=week.cutoff,
                rebalance_session=week.rebalance_session,
                next_rebalance_session=week.next_rebalance_session,
                news_by_symbol=news,
                p_calm=week.p_calm,
                p_stress=week.p_stress,
            )
        )
    return shuffled


def _reraise_oom(exc: Exception, device) -> None:
    if is_accelerator_out_of_memory(exc, device):
        raise


def _retrain_ablation(
    start_policy: PPODiscoveryActorCritic,
    train_weeks,
    test_weeks,
    eval_kw,
    config: PPODiscoveryConfig,
    *,
    freeze_encoder_updates: int,
    device,
) -> dict[str, Any]:
    from dataclasses import replace

    try:
        policy = PPODiscoveryActorCritic(config).to(device)
        policy.load_state_dict(deepcopy(start_policy.state_dict()))
        policy.to(device)
        train_cfg = replace(config, freeze_encoder_updates=freeze_encoder_updates)
        train_ppo_discovery(
            policy,
            lambda current, cache: collect_closed_loop_rollout(
                current,
                train_weeks,
                snapshot=eval_kw["snapshot"],
                ohlcv_by_symbol=eval_kw["ohlcv"],
                spy=eval_kw["spy"],
                feature_scalers=eval_kw["scalers"],
                config=train_cfg,
                temporal_cache=cache,
            ),
            config=train_cfg,
            seed=int(config.seeds[0]),
            device=device,
        )
        return _metrics(policy, test_weeks, **eval_kw)
    except Exception as exc:
        _reraise_oom(exc, device)
        return {"status": "failed", "error": str(exc)}


__all__ = ["run_required_ablations"]

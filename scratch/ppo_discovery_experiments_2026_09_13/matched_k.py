"""Matched-K closed loops with average-rank percentiles (informational)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from brain_api.core.ppo_discovery.config import MAX_SELECTED, PPODiscoveryConfig
from brain_api.core.ppo_discovery.environment import (
    WeeklyTransition,
    collect_closed_loop_rollout,
)
from brain_api.core.ppo_discovery.evaluator import evaluate_policy_weeks
from brain_api.core.ppo_discovery.policy import PPODiscoveryActorCritic
from brain_api.core.ppo_discovery.pretraining import _average_ranks
from brain_api.core.ppo_discovery.schemas import UniverseSnapshot


def matched_k_average_rank(
    policy: PPODiscoveryActorCritic,
    *,
    test_weeks: Sequence[WeeklyTransition],
    snapshot: UniverseSnapshot,
    ohlcv: Mapping,
    spy,
    scalers: Mapping[str, Mapping[str, float]],
    config: PPODiscoveryConfig,
) -> dict[str, object]:
    """Independent force-K loops. Percentile uses average ranks on CAGR ties."""
    cagrs: dict[int, float] = {}
    for k in range(MAX_SELECTED + 1):
        steps_logs = _week_logs_forced_k(
            policy,
            test_weeks,
            snapshot,
            ohlcv,
            spy,
            scalers,
            config,
            force_k=k,
        )
        cagrs[k] = evaluate_policy_weeks(steps_logs)["cagr"]
    values = np.asarray([cagrs[k] for k in range(MAX_SELECTED + 1)], dtype=np.float64)
    ranks = _average_ranks(values)
    n = len(values)
    percentiles = {int(k): float((ranks[k] - 1.0) / max(n - 1, 1)) for k in range(n)}
    return {"cagr_by_k": cagrs, "average_rank_percentile": percentiles}


def _week_logs_forced_k(
    policy, weeks, snapshot, ohlcv, spy, scalers, config, *, force_k: int
) -> list[float]:
    steps = collect_closed_loop_rollout(
        policy,
        weeks,
        snapshot=snapshot,
        ohlcv_by_symbol=ohlcv,
        spy=spy,
        feature_scalers=scalers,
        config=config,
        deterministic=True,
        force_k=force_k,
    )
    return [step.realized_net_return for step in steps]


__all__ = ["matched_k_average_rank"]

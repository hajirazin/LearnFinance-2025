"""Locked random allocator used as the ppo_discovery paper baseline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from brain_api.core.ppo_discovery.config import (
    CASH_FLOOR,
    MAX_SELECTED,
    PPODiscoveryConfig,
)
from brain_api.core.ppo_discovery.environment import (
    WeeklyTransition,
    collect_closed_loop_rollout,
)
from brain_api.core.ppo_discovery.evaluator import evaluate_policy_weeks
from brain_api.core.ppo_discovery.schemas import CanonicalPPOState, UniverseSnapshot

LOCKED_RANDOM_SEED = 2026


class LockedRandomAllocator:
    """Coin-toss K in ``0..min(15, n_eligible)``, equal stock weights, 2% cash."""

    def __init__(self, seed: int = LOCKED_RANDOM_SEED) -> None:
        self._rng = np.random.default_rng(seed)

    def eval(self) -> LockedRandomAllocator:
        return self

    def infer_weights(self, state: CanonicalPPOState) -> dict[str, float]:
        eligible = [
            symbol
            for symbol, flag in zip(
                state.symbols, state.asset_mask.tolist(), strict=True
            )
            if flag and symbol
        ]
        n_eligible = len(eligible)
        max_k = min(MAX_SELECTED, n_eligible)
        k = int(self._rng.integers(0, max_k + 1)) if max_k else 0
        if k == 0:
            return {"CASH": 1.0}
        chosen = sorted(self._rng.choice(eligible, size=k, replace=False).tolist())
        stock_mass = 1.0 - CASH_FLOOR
        weight = stock_mass / k
        return {**dict.fromkeys(chosen, weight), "CASH": CASH_FLOOR}

    def infer_decision_value(
        self, state: CanonicalPPOState, force_k: int | None = None
    ) -> tuple[dict[str, float], tuple[str, ...], float]:
        del force_k
        weights = self.infer_weights(state)
        order = tuple(symbol for symbol in weights if symbol != "CASH")
        return weights, order, 0.0

    def value(self, state: CanonicalPPOState) -> torch.Tensor:
        del state
        return torch.zeros((), dtype=torch.float32)


def locked_random_test_metrics(
    test_weeks: Sequence[WeeklyTransition],
    *,
    snapshot: UniverseSnapshot,
    ohlcv: Mapping[str, Any],
    spy: Any,
    scalers: Mapping[str, Any],
    config: PPODiscoveryConfig,
) -> dict[str, float]:
    logs = collect_closed_loop_rollout(
        LockedRandomAllocator(),  # type: ignore[arg-type]
        test_weeks,
        snapshot=snapshot,
        ohlcv_by_symbol=ohlcv,
        spy=spy,
        feature_scalers=scalers,
        config=config,
        deterministic=True,
    )
    metrics = evaluate_policy_weeks([step.realized_net_return for step in logs])
    metrics["seed"] = float(LOCKED_RANDOM_SEED)
    return metrics


__all__ = [
    "LOCKED_RANDOM_SEED",
    "LockedRandomAllocator",
    "locked_random_test_metrics",
]

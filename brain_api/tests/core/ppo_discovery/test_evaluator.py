"""CAGR and drawdown evaluation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from brain_api.core.ppo_discovery.config import PPODiscoveryConfig
from brain_api.core.ppo_discovery.evaluator import (
    evaluate_policy_weeks,
    weekly_net_cagr,
)
from brain_api.core.ppo_discovery.schemas import PPODiscoveryError


def test_weekly_net_cagr_accepts_ndarray() -> None:
    logs = np.asarray([0.01, 0.02, -0.005], dtype=np.float64)
    cagr = weekly_net_cagr(logs)
    assert np.isfinite(cagr)
    metrics = evaluate_policy_weeks(logs)
    assert metrics["cagr"] == pytest.approx(cagr)


def test_weekly_net_cagr_rejects_empty() -> None:
    with pytest.raises(PPODiscoveryError, match="at least one week"):
        weekly_net_cagr([])


def test_select_candidate_seed_is_closest_to_median_not_max() -> None:
    from brain_api.core.ppo_discovery.evaluator import select_candidate_seed

    chosen = select_candidate_seed(
        {42: 0.10, 123: 0.20, 2026: 0.50},
        {42: 0.1, 123: 0.1, 2026: 1.0},
    )
    assert chosen == 123


def test_old_ten_seed_tuple_is_diagnostic() -> None:
    from brain_api.core.ppo_discovery.splits import resolve_experiment_variant

    diagnostic = PPODiscoveryConfig(seeds=(42, 123, 2026, 7, 19, 31, 73, 101, 211, 509))
    assert resolve_experiment_variant(diagnostic) == "diagnostic"
    assert resolve_experiment_variant(PPODiscoveryConfig(seeds=(42,))) == "diagnostic"

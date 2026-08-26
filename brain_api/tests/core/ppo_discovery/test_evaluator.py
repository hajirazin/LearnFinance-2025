"""CAGR and drawdown evaluation helpers."""

from __future__ import annotations

import numpy as np
import pytest

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

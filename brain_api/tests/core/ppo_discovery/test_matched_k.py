"""Matched-K informational closed loops use average ranks on CAGR ties."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from brain_api.core.ppo_discovery.config import MAX_SELECTED, PPODiscoveryConfig
from brain_api.core.ppo_discovery.matched_k import matched_k_average_rank
from brain_api.core.ppo_discovery.synthetic import make_synthetic_state


def test_matched_k_average_rank_percentile_on_cagr_ties(monkeypatch) -> None:
    config = PPODiscoveryConfig(dropout=0.0)
    cagrs = {k: 0.10 if k < 8 else 0.20 for k in range(MAX_SELECTED + 1)}

    def _fake_logs(*_args, force_k: int, **_kwargs):
        return [float(cagrs[force_k])]

    monkeypatch.setattr(
        "brain_api.core.ppo_discovery.matched_k._week_logs_forced_k",
        _fake_logs,
    )
    monkeypatch.setattr(
        "brain_api.core.ppo_discovery.matched_k.evaluate_policy_weeks",
        lambda logs: {"cagr": logs[0]},
    )
    result = matched_k_average_rank(
        MagicMock(),
        test_weeks=[],
        snapshot=make_synthetic_state().universe_snapshot,
        ohlcv={},
        spy=None,
        scalers={},
        config=config,
    )
    low = [result["average_rank_percentile"][k] for k in range(8)]
    high = [result["average_rank_percentile"][k] for k in range(8, MAX_SELECTED + 1)]
    assert np.allclose(low, low[0])
    assert np.allclose(high, high[0])
    assert high[0] > low[0]
    assert result["cagr_by_k"][0] == 0.10
    assert result["cagr_by_k"][15] == 0.20

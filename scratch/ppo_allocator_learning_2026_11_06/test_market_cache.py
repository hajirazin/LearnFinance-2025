"""Behavioral parity for the fast research replay and the outer time boundary."""

import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from market_rollout import rollout
from research_support import ROOT

from brain_api.core.ppo_discovery.config import PPODiscoveryConfig
from brain_api.core.ppo_discovery.environment import (
    WeeklyTransition,
    collect_closed_loop_rollout,
)
from brain_api.core.ppo_discovery.policy import PPODiscoveryActorCritic
from brain_api.core.ppo_discovery.weeks import open_to_open_return
from brain_api.core.prices import repair_ohlc_envelope


@pytest.fixture(scope="module")
def evidence():
    if not (ROOT / "market_dataset.pkl").exists():
        pytest.skip("Requires the locally generated, unversioned market dataset")
    with (ROOT / "market_dataset.pkl").open("rb") as handle:
        return pickle.load(handle)


def test_outer_cutoff_excludes_2026_training_returns(evidence):
    train = [r for r in evidence["templates"] if r["split"] == "train"]
    test = [r for r in evidence["templates"] if r["split"] == "test"]
    assert all(r["end"].year <= 2025 for r in train)
    assert all(r["start"].year == 2026 for r in test)
    assert evidence["hmm"].training_cutoff_date.year <= 2025
    assert len(train) == 312 and len(test) == 36


def test_cached_book_feedback_matches_production_replay(evidence):
    torch.set_num_threads(4)
    torch.manual_seed(42)
    config = PPODiscoveryConfig()
    policy = PPODiscoveryActorCritic(config).eval()
    with torch.no_grad():
        policy.count_head.weight.zero_()
        policy.count_head.bias.zero_()
        policy.count_head.bias[10] = 10.0
    rows = [r for r in evidence["templates"] if r["split"] == "train"][:3]
    prices = {
        s: repair_ohlc_envelope(f)
        for s, f in pd.read_pickle(ROOT / "market_prices.pkl").items()
    }
    weeks = [
        WeeklyTransition(
            cutoff=r["cutoff"],
            rebalance_session=r["start"],
            next_rebalance_session=r["end"],
            news_by_symbol=r["state"].news_by_symbol,
            p_calm=float(r["state"].globals[0]),
            p_stress=float(r["state"].globals[1]),
        )
        for r in rows
    ]
    native = collect_closed_loop_rollout(
        policy,
        weeks,
        snapshot=evidence["snapshot"],
        ohlcv_by_symbol=prices,
        spy=prices["SPY"],
        feature_scalers=evidence["scalers"],
        config=config,
        deterministic=True,
    )
    cached = rollout(policy, rows, config, deterministic=True)
    for a, b in zip(native, cached, strict=True):
        assert set(a.action.percentage_weights) == set(b.action.percentage_weights)
        for symbol in a.action.percentage_weights:
            assert a.action.percentage_weights[symbol] == pytest.approx(
                b.action.percentage_weights[symbol], abs=2e-7
            )
        assert a.realized_net_return == pytest.approx(b.realized_net_return, abs=2e-7)
        np.testing.assert_allclose(
            a.state.asset_features, b.state.asset_features, rtol=1e-7, atol=1e-7
        )
    first = rows[0]
    for symbol in ["AAPL", "MSFT", "SPY"]:
        price, ret = open_to_open_return(
            prices[symbol], first["start"], first["end"], symbol=symbol
        )
        assert first["opens"][symbol] == price
        assert first["returns"][symbol] == ret

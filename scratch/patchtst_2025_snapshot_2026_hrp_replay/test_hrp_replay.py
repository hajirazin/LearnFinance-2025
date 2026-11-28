"""Observable portfolio, covariance, and information-cutoff invariants."""

import numpy as np
import pandas as pd
import pytest
from hrp_replay import hrp_target, simulate_hrp, weighted_rebalance


def price_history():
    sessions = pd.bdate_range("2025-12-29", periods=7)
    moves = np.array([[-1, -1, -1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]]) * np.array(
        [0.01, 0.02, 0.03]
    )
    closes = np.vstack([np.ones(3) * 100, 100 * np.cumprod(1 + moves, axis=0)])
    closes = np.vstack([closes, [[900, 800, 700], [600, 500, 400]]])
    prices = {
        symbol: pd.DataFrame({"close": closes[:, i]}, index=sessions)
        for i, symbol in enumerate(["A", "B", "C"])
    }
    return prices, sessions


def test_hrp_uses_daily_simple_returns_and_recovers_inverse_variance_case():
    prices, sessions = price_history()
    weights, evidence = hrp_target(prices, ["A", "B", "C"], sessions, sessions[4], 4)
    inverse_variances = 1 / np.array([0.01, 0.02, 0.03]) ** 2
    expected = inverse_variances / inverse_variances.sum()
    assert [weights[s] for s in ["A", "B", "C"]] == pytest.approx(expected)
    assert evidence["daily_returns"] == 4
    assert evidence["history_end"] == sessions[4]


def test_current_week_and_future_prices_cannot_change_hrp_weights():
    prices, sessions = price_history()
    before, _ = hrp_target(prices, ["A", "B", "C"], sessions, sessions[4], 4)
    for frame in prices.values():
        frame.loc[frame.index > sessions[4], "close"] = np.nan
    after, _ = hrp_target(prices, ["A", "B", "C"], sessions, sessions[4], 4)
    assert after == before


@pytest.mark.parametrize("bad", [np.nan, np.inf, 0, -1])
def test_incomplete_or_invalid_history_aborts_without_dropping_selected_stock(bad):
    prices, sessions = price_history()
    prices["B"].loc[sessions[2], "close"] = bad
    with pytest.raises(ValueError, match="history"):
        hrp_target(prices, ["A", "B", "C"], sessions, sessions[4], 4)


def test_missing_exchange_session_is_not_skipped_or_filled():
    prices, sessions = price_history()
    prices["B"] = prices["B"].drop(sessions[2])
    with pytest.raises(ValueError, match="history"):
        hrp_target(prices, ["A", "B", "C"], sessions, sessions[4], 4)


def test_insufficient_history_aborts():
    prices, sessions = price_history()
    with pytest.raises(ValueError, match="history"):
        hrp_target(prices, ["A", "B", "C"], sessions, sessions[3], 4)


def test_weighted_rebalance_preserves_weekend_gap_and_finances_entry_from_exit():
    shares, book = weighted_rebalance(
        {"A": 10, "B": 20}, {"A": 0.75, "C": 0.25}, {"A": 121, "B": 45, "C": 10}, 0
    )
    assert book["nav_before"] == 2110
    assert shares == pytest.approx({"A": 2110 * 0.75 / 121, "C": 52.75})
    assert book["nav_after"] == pytest.approx(2110)
    assert sum(
        t["quantity_delta"] * t["price"] for t in book["trades"]
    ) == pytest.approx(0)
    assert book["cash_after"] == 0


@pytest.mark.parametrize("weights", [{"A": 0.9}, {"A": -0.2, "B": 1.2}, {"A": np.nan}])
def test_invalid_target_weights_abort(weights):
    with pytest.raises(ValueError, match="weights"):
        weighted_rebalance({}, weights, {"A": 10, "B": 20}, 10000)


def test_missing_held_execution_price_aborts_even_when_stock_exits():
    with pytest.raises(ValueError, match="price"):
        weighted_rebalance({"OLD": 3}, {"A": 1.0}, {"A": 20}, 0)


def test_two_week_replay_keeps_stock_selection_and_carries_holdings():
    prices, sessions = price_history()
    for frame in prices.values():
        frame["open"] = frame["close"]
    prices["A"].loc[sessions[5], ["open", "close"]] = [200, 220]
    prices["B"].loc[sessions[5], ["open", "close"]] = [200, 220]
    prices["C"].loc[sessions[5], ["open", "close"]] = [200, 220]
    for frame in prices.values():
        frame.loc[sessions[6], ["open", "close"]] = [440, 440]
    plan = [
        {
            "monday": sessions[5],
            "anchor": sessions[4],
            "execution": sessions[5],
            "valuation": sessions[5],
            "selected": ["A", "B", "C"],
        },
        {
            "monday": sessions[6],
            "anchor": sessions[5],
            "execution": sessions[6],
            "valuation": sessions[6],
            "selected": ["A", "B", "C"],
        },
    ]
    nav, ledger, holdings, _ = simulate_hrp(prices, plan, sessions, lookback=4)
    assert nav.tolist() == pytest.approx([10000, 11000, 22000])
    assert ledger[1]["rebalance"]["nav_before"] == pytest.approx(22000)
    assert holdings.groupby("monday").symbol.apply(set).tolist() == [
        {"A", "B", "C"},
        {"A", "B", "C"},
    ]


def test_completed_experiment_skips_recalculation_and_rejects_changed_inputs(
    tmp_path, monkeypatch, capsys
):
    import json
    from types import SimpleNamespace

    import run as runner

    root = tmp_path / "scratch/new"
    source = tmp_path / "scratch/original"
    root.mkdir(parents=True)
    source.mkdir()
    source.joinpath("evidence.txt").write_text("frozen")
    for name in ["run.py", "hrp_replay.py"]:
        root.joinpath(name).write_text("test-owned code identity")
    core = tmp_path / "brain_api/brain_api/core"
    core.mkdir(parents=True)
    core.joinpath("hrp.py").write_text("test-owned HRP identity")
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "SOURCE", source)
    monkeypatch.setattr(runner, "REPO", tmp_path)
    sessions = pd.bdate_range("2024-10-01", "2026-01-16")
    rng = np.random.default_rng(42)
    prices = {
        s: pd.DataFrame({"open": values, "close": values}, index=sessions)
        for s, values in zip(
            ["A", "B", "C"],
            100 * np.exp(np.cumsum(rng.normal(0, 0.01, (3, len(sessions))), axis=1)),
            strict=True,
        )
    }
    plan = [
        {
            "monday": pd.Timestamp(monday),
            "anchor": pd.Timestamp(anchor),
            "execution": pd.Timestamp(monday),
            "valuation": pd.Timestamp(valuation),
            "selected": ["A", "B", "C"],
            "week_complete": True,
        }
        for monday, anchor, valuation in [
            ("2026-01-05", "2026-01-02", "2026-01-09"),
            ("2026-01-12", "2026-01-09", "2026-01-16"),
        ]
    ]
    original = pd.DataFrame(
        {"PatchTST": [10000, 10100, 10200]},
        index=pd.DatetimeIndex(
            [plan[0]["execution"], *[w["valuation"] for w in plan]], name="date"
        ),
    )
    metadata = {
        "snapshot": "test-only",
        "completed_weeks": 2,
        "final_week_complete": True,
        "limitations": [],
    }
    monkeypatch.setattr(
        runner,
        "load_evidence",
        lambda _: (
            metadata,
            {"retrieved_at": "2026-01-17T00:00Z"},
            prices,
            plan,
            original,
        ),
    )
    monkeypatch.setattr(
        runner,
        "load_patchtst_calendar",
        lambda *args: SimpleNamespace(sessions=sessions),
    )
    runner.main()
    summary = root / "summary.json"
    initial = summary.read_bytes()
    saved = json.loads(initial)
    assert saved["weeks"] == 2
    assert saved["new_model_inference_runs"] == 0

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed evidence must not be replayed")

    monkeypatch.setattr(runner, "load_evidence", forbidden)
    monkeypatch.setattr(runner, "simulate_hrp", forbidden)
    runner.main()
    assert "already complete; skipped" in capsys.readouterr().out
    assert summary.read_bytes() == initial
    source.joinpath("evidence.txt").write_text("changed")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        runner.main()

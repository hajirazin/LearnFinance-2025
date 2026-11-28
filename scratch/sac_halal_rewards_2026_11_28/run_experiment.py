"""Matched full-budget SAC reward experiment; resumable per completed seed."""

import argparse
import hashlib
import json
import os
import pickle
import socket
import time
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from experiment_settings import BENCHMARK, ROOT, SETTINGS, UNIVERSE
from reward_environment import RewardOffsetEnv, benchmark_log_hurdle

from brain_api.core.portfolio_rl.broker_costs import compute_ibkr_rebalance_cost
from brain_api.core.portfolio_rl.eval import compute_cagr, compute_sharpe_ratio
from brain_api.core.sac import training
from brain_api.core.sac.config import DEFAULT_SAC_CONFIG, make_sac_config_for_n_stocks

VARIANTS = SETTINGS.variants
SEEDS = (42, 123, 2026)
FACTORY = training.create_env_from_training_data


def metrics(returns):
    values = np.asarray(returns, dtype=float)
    wealth = np.r_[1.0, np.cumprod(1 + values)]
    return {
        "cagr": compute_cagr(values),
        "sharpe": compute_sharpe_ratio(values),
        "max_drawdown": float(np.max(1 - wealth / np.maximum.accumulate(wealth))),
        "total_return": float(wealth[-1] - 1),
        "weeks": len(values),
    }


def equal_weight_buy_hold_returns(stock_returns):
    relatives = np.vstack(
        (np.ones(stock_returns.shape[1]), np.cumprod(1 + stock_returns, axis=0))
    )
    wealth = relatives.mean(axis=1)
    return wealth[1:] / wealth[:-1] - 1


def net_entry_returns(gross_returns, symbols, prices, config):
    n_stocks = len(symbols)
    current = np.r_[np.zeros(n_stocks), 1.0]
    target = np.r_[np.full(n_stocks, 1.0 / n_stocks), 0.0]
    cost = compute_ibkr_rebalance_cost(
        symbol_order=symbols,
        current_weights=current,
        target_weights=target,
        prices=np.asarray(prices),
        cfg=config.cost_config,
    )
    result = np.asarray(gross_returns).copy()
    # Buy-and-hold benchmarks pay entry from capital before investment.
    result[0] = (1 - cost.total_fraction) * (1 + result[0]) - 1
    return result, cost.total_usd


def trace_policy(result, data, panel):
    start = data.n_weeks - 104
    env = FACTORY(data, result.config, start_week=start, end_week=data.n_weeks)
    env.max_episode_weeks = None
    normalized = training.NormalizedEnv(env, result.scaler)
    state = normalized.reset(start_week=0)
    rows = []
    done = False
    while not done:
        action = result.actor.get_action(state, deterministic=True)
        step = normalized.step(action)
        info = step.info
        row = {
            "session": panel["sessions"][start + len(rows)],
            "net_return": info["net_portfolio_return"],
            "gross_return": info["portfolio_return"],
            "cost_usd": info["transaction_cost_usd"],
            "turnover": info["turnover"],
            "cash": info["target_weights"][-1],
        }
        row.update(
            {
                f"weight_{symbol}": info["target_weights"][i]
                for i, symbol in enumerate(data.symbol_order)
            }
        )
        rows.append(row)
        state, done = step.next_state, step.done
    if len(rows) != 104:
        raise ValueError(f"Expected 104 evaluation weeks, found {len(rows)}")
    frame = pd.DataFrame(rows)
    measured = metrics(frame.net_return)
    if not np.isclose(measured["cagr"], result.eval_cagr) or not np.isclose(
        measured["sharpe"], result.eval_sharpe
    ):
        raise ValueError("Independent trace disagrees with production evaluator")
    return frame, measured


def forbid_network(*args, **kwargs):
    raise RuntimeError("Training experiment forbids all network access")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    panel_bytes = (ROOT / "evidence/panel.pkl").read_bytes()
    panel_digest = hashlib.sha256(panel_bytes).hexdigest()
    panel = pickle.loads(panel_bytes)
    data = panel["data"]
    config = replace(
        make_sac_config_for_n_stocks(DEFAULT_SAC_CONFIG, data.n_stocks),
        total_timesteps=args.steps,
    )
    output = ROOT / f"results_{args.steps}"
    output.mkdir(exist_ok=True)
    start = data.n_weeks - 104
    ew_train = equal_weight_buy_hold_returns(data.symbol_returns[:start])
    ew_test = equal_weight_buy_hold_returns(data.symbol_returns[start:])
    equal_weight = np.r_[ew_train, ew_test]
    baselines = {}
    benchmark_frame = pd.DataFrame({"session": panel["sessions"][start:-1]})
    for name, gross, symbols, entry_prices in (
        (
            f"{BENCHMARK}_buy_hold",
            panel[f"{BENCHMARK.lower()}_returns"][start:],
            [BENCHMARK],
            [panel[f"{BENCHMARK.lower()}_prices"][start]],
        ),
        (
            f"{UNIVERSE}_equal_weight_buy_hold",
            ew_test,
            data.symbol_order,
            data.prices[start],
        ),
    ):
        returns, entry_cost = net_entry_returns(gross, symbols, entry_prices, config)
        baselines[name] = {**metrics(returns), "entry_cost_usd": entry_cost}
        benchmark_frame[name] = returns
    benchmark_frame.to_csv(output / "benchmarks.csv", index=False)
    (output / "benchmarks.json").write_text(json.dumps(baselines, indent=2))
    for seed in args.seeds:
        for variant in args.variants:
            target = output / variant / f"seed_{seed}"
            result_path = target / "metrics.json"
            if result_path.exists():
                saved = json.loads(result_path.read_text())
                if (
                    saved["panel_sha256"] != panel_digest
                    or saved["config"] != replace(config, seed=seed).to_dict()
                ):
                    raise ValueError(
                        "Existing result has incompatible evidence/configuration"
                    )
                print("Reusing completed", variant, seed, flush=True)
                continue
            hurdles = benchmark_log_hurdle(
                variant,
                panel[f"{BENCHMARK.lower()}_returns"],
                equal_weight,
                benchmark=BENCHMARK,
            )

            def factory(
                training_data,
                seed_config,
                start_week=0,
                end_week=None,
                _hurdles=hurdles,
            ):
                env = FACTORY(training_data, seed_config, start_week, end_week)
                return RewardOffsetEnv(env, _hurdles[start_week:end_week])

            print(f"START variant={variant} seed={seed} steps={args.steps}", flush=True)
            started = time.monotonic()
            with (
                patch.object(socket.socket, "connect", forbid_network),
                patch.object(training, "create_env_from_training_data", factory),
                # Tiny attention networks run faster on one CPU thread than MPS.
                patch(
                    "brain_api.core.portfolio_rl.sac_trainer.get_device",
                    lambda: torch.device("cpu"),
                ),
            ):
                result = training.train_sac(data, replace(config, seed=seed))
            frame, measured = trace_policy(result, data, panel)
            target.mkdir(parents=True, exist_ok=True)
            frame.to_csv(target / "test_returns.csv", index=False)
            torch.save(
                {
                    "actor": result.actor.state_dict(),
                    "critic": result.critic.state_dict(),
                },
                target / "model.pt",
            )
            (target / "scaler.pkl").write_bytes(pickle.dumps(result.scaler))
            (target / "regime_hmm.pkl").write_bytes(pickle.dumps(result.regime_hmm))
            report = {
                "run_id": f"scratch:sac-{UNIVERSE}-rewards:{BENCHMARK}:2026-11-28",
                "universe": UNIVERSE,
                "benchmark": BENCHMARK,
                "attempt": 1,
                "as_of": "2026-11-28",
                "variant": variant,
                "seed": seed,
                "config": result.config.to_dict(),
                "panel_sha256": panel_digest,
                "training_seconds": time.monotonic() - started,
                "test": measured,
                "avg_weekly_turnover": float(frame.turnover.mean()),
                "avg_cash_weight": float(frame.cash.mean()),
                "test_cost_usd_fixed_nav": float(frame.cost_usd.sum()),
                "audit": result.audit_metadata,
                f"beats_{BENCHMARK.lower()}_both": measured["cagr"]
                > baselines[f"{BENCHMARK}_buy_hold"]["cagr"]
                and measured["sharpe"] > baselines[f"{BENCHMARK}_buy_hold"]["sharpe"],
            }
            result_path.write_text(json.dumps(report, indent=2))
            print(
                "COMPLETE",
                variant,
                seed,
                json.dumps(measured),
                f"seconds={report['training_seconds']:.1f}",
                flush=True,
            )
    print("ALL REQUESTED RUNS COMPLETE", flush=True)


if __name__ == "__main__":
    main()

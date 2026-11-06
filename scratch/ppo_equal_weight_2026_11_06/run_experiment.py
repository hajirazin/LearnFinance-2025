"""Read-only PPO behavioral replay and same-decision allocation ablation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from statistics_helpers import (
    bootstrap,
    dispersion,
    dispersion_summary,
    rank_correlation,
)

from brain_api.core.ppo_discovery.distributions import (
    clamp_concentration,
    deterministic_weights,
)
from brain_api.core.ppo_discovery.environment import (
    _next_open_market,
    _post_rebalance_weights,
)
from brain_api.core.ppo_discovery.evaluator import evaluate_policy_weeks
from brain_api.core.ppo_discovery.news_adapter import load_historical_ppo_news_features
from brain_api.core.ppo_discovery.policy import (
    PPODiscoveryActorCritic,
    tensors_from_state,
)
from brain_api.core.ppo_discovery.regime import weekly_regime_probabilities
from brain_api.core.ppo_discovery.rewards import ppo_discovery_reward
from brain_api.core.ppo_discovery.schemas import UniverseSnapshot
from brain_api.core.ppo_discovery.state_builder import (
    StateBuildRequest,
    build_ppo_discovery_state,
)
from brain_api.core.ppo_discovery.weeks import (
    actor_cutoff_datetimes,
    open_to_open_return,
    prices_as_of,
    weekly_trade_clock,
)
from brain_api.core.prices import repair_ohlc_envelope
from brain_api.core.sac.regime_hmm import RegimeHMMArtifact
from brain_api.news.store import NewsStore, news_db_path, news_lock_path
from brain_api.storage.ppo_discovery.local import PPODiscoveryHalalNewModelStorage

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
DATA = REPO / "brain_api/data"
BUCKET = DATA / "models/ppo_discovery_halal_new"


def write_json(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, allow_nan=False, default=str))


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def protected_hashes():
    paths = list((REPO / "brain_api/brain_api").rglob("*.py"))
    paths += [p for p in BUCKET.rglob("*") if p.is_file()]
    paths += [DATA / "news/news_v1.duckdb"]
    paths += list((REPO / "temporal").glob("**/*.py"))
    paths = [p for p in paths if ".venv" not in p.parts]
    return {str(p.relative_to(REPO)): file_hash(p) for p in sorted(paths)}


class ReadOnlyNewsStore(NewsStore):
    def __init__(self):
        self.base_path = DATA
        self.db_path = news_db_path(DATA)
        self.lock_path = news_lock_path(DATA)

    def _connect(self):
        return duckdb.connect(str(self.db_path), read_only=True)


def summarize_saved():
    result = {}
    for version in sorted(BUCKET.glob("v*")):
        evaluation = json.loads((version / "evaluation.json").read_text())
        rows = (evaluation.get("portfolio_diagnostics") or {}).get("weekly", [])
        if not rows:
            continue
        cvs = [r["stock_weight_cv"] for r in rows]
        result[version.name] = {
            "weeks": len(rows),
            "mean_cv": float(np.mean(cvs)),
            "max_cv": max(cvs),
            "fraction_cv_below_5pct": float(np.mean(np.array(cvs) <= 0.05)),
            "mean_cv_bootstrap": bootstrap(cvs),
        }
    return result


def model_details(policy, state, force_k=None):
    with torch.no_grad():
        history, features, globals_, mask = tensors_from_state(
            state, torch.device("cpu")
        )
        encoded, pooled = policy.encode(
            history[None], features[None], globals_[None], mask[None]
        )
        counts, scores, _ = policy.heads(encoded, pooled)
        valid = np.flatnonzero(state.asset_mask).tolist()
        ranking = sorted(valid, key=lambda i: (-float(scores[0, i]), state.symbols[i]))
        k = int(counts[0].argmax()) if force_k is None else force_k
        selected = ranking[:k]
        if not selected:
            return {"CASH": 1.0}, [], scores[0], None, encoded[0], pooled[0], counts[0]
        cash, raw = policy.cash_and_allocation_raw(encoded[0], pooled[0], selected)
        weights = deterministic_weights(
            count_logits=counts[0],
            selection_logits=scores[0],
            cash_raw=cash,
            allocation_raw=raw,
            asset_mask=mask,
            symbols=state.symbols,
            cash_floor=policy.config.cash_floor,
            force_k=force_k,
        )
        alpha = clamp_concentration(raw[selected]).detach()
    return (
        weights,
        selected,
        scores[0].detach(),
        alpha,
        encoded[0].detach(),
        pooled[0].detach(),
        counts[0].detach(),
    )


def equalize(weights):
    stocks = [s for s in weights if s != "CASH"]
    if not stocks:
        return {"CASH": 1.0}
    cash = weights["CASH"]
    return {**dict.fromkeys(stocks, (1 - cash) / len(stocks)), "CASH": cash}


def transition(prior, target, prices, start, end, config):
    returns, opens = _next_open_market(prior, target, prices, start, end)
    _, gross, cost, net = ppo_discovery_reward(
        prior_weights=prior,
        target_weights=target,
        symbol_returns=returns,
        symbol_prices=opens,
        nav_usd=config.training_nav_usd,
        config=config,
    )
    return _post_rebalance_weights(target, returns, opens, cost), {
        "net_log": net,
        "gross": gross,
        "cost": cost,
    }


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    before = protected_hashes()
    write_json("protected_before.json", before)
    storage = PPODiscoveryHalalNewModelStorage(DATA)
    # Replays keep the model version bound to the downloaded evidence, even
    # after a future production promotion changes the current pointer.
    version = json.loads((ROOT / "download_manifest.json").read_text())["version"]
    artifact = storage.load_artifacts(version)
    policy = PPODiscoveryActorCritic(artifact.config).eval()
    policy.load_state_dict(artifact.policy_state_dict)
    manifest = artifact.universe_manifest.copy()
    manifest["sorted_symbols"] = tuple(manifest["sorted_symbols"])
    snapshot = UniverseSnapshot(**manifest)
    saved_eval = json.loads((artifact.artifact_dir / "evaluation.json").read_text())
    cutoffs = [
        datetime.fromisoformat(r["as_of"])
        for r in saved_eval["portfolio_diagnostics"]["weekly"]
    ]
    prices = {
        s: repair_ohlc_envelope(f)
        for s, f in pd.read_pickle(ROOT / "prices.pkl").items()
    }
    news = load_historical_ppo_news_features(
        cutoffs, list(snapshot.sorted_symbols), store=ReadOnlyNewsStore()
    )
    end_date = date.fromisoformat(artifact.metadata["end_date"])
    clock = weekly_trade_clock(date(2019, 1, 1), end_date)
    clocks = {
        c: (clock.rebalance_sessions[i], clock.rebalance_sessions[i + 1])
        for i, c in enumerate(actor_cutoff_datetimes(clock)[:-1])
    }
    regimes = weekly_regime_probabilities(
        RegimeHMMArtifact.from_dict(artifact.regime_hmm),
        prices,
        start_date=date(2019, 1, 1),
        completed_through=max(cutoffs).date(),
        weekly_cutoffs=[c.date() for c in cutoffs],
    )
    write_json(
        "input_manifest.json",
        {
            "version": version,
            "seed": artifact.metadata["training_seed"],
            "policy_sha256": file_hash(artifact.artifact_dir / "policy.pt"),
            "prices_sha256": file_hash(ROOT / "prices.pkl"),
            "protocol_sha256": file_hash(ROOT / "PROTOCOL.md"),
            "cutoffs": [c.isoformat() for c in cutoffs],
            "news_identity": "production adapter, read-only DuckDB",
            "original_test_hash": artifact.metadata["evaluation_dataset_hash"],
            "fresh_prices_exact_original_reproduction": False,
            "empty_download_symbols": [s for s, f in prices.items() if f.empty],
            "survivorship_bias": True,
            "retrospective_reused_test_set": True,
        },
    )
    print(
        f"Loaded policy={version}, test_weeks={len(cutoffs)}, news complete", flush=True
    )
    books = {name: {"CASH": 1.0} for name in ["ppo", "equal_fixed", "equal_feedback"]}
    rows, stock_rows, feedback_rows, force_rows = (
        [],
        [],
        [],
        {k: [] for k in [3, 5, 10, 15]},
    )
    all_logs = {name: [] for name in books}
    for index, cutoff in enumerate(cutoffs):
        start, end = clocks[cutoff]
        sliced = {
            s: prices_as_of(f, cutoff.date())
            for s, f in prices.items()
            if not f.empty and f.index.min().date() <= cutoff.date()
        }
        calm, stress = regimes[cutoff.date()]

        def state_for(book):
            return build_ppo_discovery_state(
                StateBuildRequest(
                    as_of=cutoff,
                    universe_snapshot=snapshot,
                    ohlcv_by_symbol=sliced,
                    news_by_symbol=news[cutoff],
                    current_weights=book,
                    p_calm=calm,
                    p_stress=stress,
                    spy_closes=sliced["SPY"]["close"].to_numpy(),
                    feature_scalers=artifact.feature_scalers,
                )
            )

        state = state_for(books["ppo"])
        weights, selected, scores, alpha, encoded, pooled, counts = model_details(
            policy, state
        )
        if index == 0:
            official = policy.infer_weights(state)
            assert set(official) == set(weights)
            assert max(abs(official[s] - weights[s]) for s in weights) < 1e-7
        fixed = equalize(weights)
        assert abs(fixed["CASH"] - weights["CASH"]) < 1e-12
        k = len(selected)
        selected_symbols = [state.symbols[i] for i in selected]
        outcome = {}
        for name, target in [("ppo", weights), ("equal_fixed", fixed)]:
            books[name], outcome[name] = transition(
                books[name], target, prices, start, end, artifact.config
            )
            all_logs[name].append(outcome[name]["net_log"])
        feedback_state = state_for(books["equal_feedback"])
        feedback = equalize(policy.infer_weights(feedback_state))
        books["equal_feedback"], feedback_outcome = transition(
            books["equal_feedback"], feedback, prices, start, end, artifact.config
        )
        all_logs["equal_feedback"].append(feedback_outcome["net_log"])
        feedback_rows.append({"as_of": cutoff.isoformat(), "weights": feedback})
        metrics = dispersion([weights[s] for s in selected_symbols])
        row = {
            "as_of": cutoff.isoformat(),
            "k": k,
            "cash": weights["CASH"],
            "weights": weights,
            "equity_equal_portfolio_weight": (1 - weights["CASH"]) / k if k else None,
            "eligible_count": int(state.asset_mask.sum()),
            **(metrics or {}),
            "ppo": outcome["ppo"],
            "equal_fixed": outcome["equal_fixed"],
        }
        if k >= 2:
            samples = torch.distributions.Dirichlet(alpha).sample((1000,)).numpy()
            row["sampled_conditional_cv_median"] = float(
                np.median(samples.std(axis=1) * k)
            )
            row["sampled_conditional_cv_p05_p95"] = np.quantile(
                samples.std(axis=1) * k, [0.05, 0.95]
            ).tolist()
            row["alpha_min_max_sum"] = [
                float(alpha.min()),
                float(alpha.max()),
                float(alpha.sum()),
            ]
            row["selection_score_weight_spearman"] = rank_correlation(
                scores[selected].numpy(), [weights[s] for s in selected_symbols]
            )
        future, eligible_scores, missing_future = [], [], []
        for slot in np.flatnonzero(state.asset_mask):
            symbol = state.symbols[slot]
            try:
                _, realized = open_to_open_return(
                    prices[symbol], start, end, symbol=symbol
                )
            except ValueError as exc:
                missing_future.append({"symbol": symbol, "error": str(exc)})
                continue
            future.append(realized)
            eligible_scores.append(float(scores[slot]))
        row["selection_score_realized_rank_ic"] = rank_correlation(
            eligible_scores, future
        )
        row["selection_outcome_missing"] = missing_future
        row["selection_gross_edge_vs_matched_cash_universe_mean"] = outcome[
            "equal_fixed"
        ]["gross"] - (1 - weights["CASH"]) * float(np.mean(future))
        row["selection_evaluable_count"] = len(future)
        for rank, slot in enumerate(selected, 1):
            symbol = state.symbols[slot]
            stock_rows.append(
                {
                    "as_of": cutoff.isoformat(),
                    "symbol": symbol,
                    "selection_rank": rank,
                    "selection_logit": float(scores[slot]),
                    "portfolio_weight": weights[symbol],
                    "equity_weight": weights[symbol] / (1 - weights["CASH"]),
                    "relative_deviation": k * weights[symbol] / (1 - weights["CASH"])
                    - 1,
                    "alpha": float(alpha[rank - 1]),
                }
            )
        with torch.no_grad():
            ranking = sorted(
                np.flatnonzero(state.asset_mask),
                key=lambda i: (-float(scores[i]), state.symbols[i]),
            )
            mask = torch.as_tensor(state.asset_mask)
            for forced_k in force_rows:
                forced_selected = ranking[:forced_k]
                cash_raw, raw = policy.cash_and_allocation_raw(
                    encoded, pooled, forced_selected
                )
                forced = deterministic_weights(
                    count_logits=counts,
                    selection_logits=scores,
                    cash_raw=cash_raw,
                    allocation_raw=raw,
                    asset_mask=mask,
                    symbols=state.symbols,
                    cash_floor=artifact.config.cash_floor,
                    force_k=forced_k,
                )
                force_rows[forced_k].append(
                    {
                        "as_of": cutoff.isoformat(),
                        "k": forced_k,
                        **dispersion(
                            [forced[state.symbols[i]] for i in forced_selected]
                        ),
                    }
                )
        rows.append(row)
        write_json("weekly.json", rows)
        print(
            f"week={index + 1}/{len(cutoffs)} date={cutoff.date()} K={k} CV={row.get('cv', 0):.4f} max_dev={row.get('max_relative_deviation', 0):.4f}",
            flush=True,
        )
    summary = {
        "version": version,
        "finished_at": datetime.now(UTC).isoformat(),
        "dispersion": dispersion_summary(rows),
        "k_counts": dict(Counter(r["k"] for r in rows)),
        "mean_cash": float(np.mean([r["cash"] for r in rows])),
        "performance": {
            name: evaluate_policy_weeks(logs) for name, logs in all_logs.items()
        },
        "paired_allocation_net_log_difference": {
            str(b): bootstrap(
                np.array(all_logs["ppo"]) - all_logs["equal_fixed"], block=b
            )
            for b in [4, 8, 13]
        },
        "selection_rank_ic": {
            str(b): bootstrap(
                [
                    r["selection_score_realized_rank_ic"]
                    for r in rows
                    if r["selection_score_realized_rank_ic"] is not None
                ],
                block=b,
            )
            for b in [4, 8, 13]
        },
        "selection_gross_edge": bootstrap(
            [r["selection_gross_edge_vs_matched_cash_universe_mean"] for r in rows]
        ),
        "mean_selection_score_weight_spearman": float(
            np.mean(
                [
                    r["selection_score_weight_spearman"]
                    for r in rows
                    if r.get("selection_score_weight_spearman") is not None
                ]
            )
        ),
        "mean_sampled_conditional_cv_median": float(
            np.mean(
                [
                    r["sampled_conditional_cv_median"]
                    for r in rows
                    if "sampled_conditional_cv_median" in r
                ]
            )
        ),
        "forced_k_sensitivity": {
            str(k): dispersion_summary(v) for k, v in force_rows.items()
        },
        "saved_artifact_audit": summarize_saved(),
        "counterexample_weeks_10pct": [
            r["as_of"] for r in rows if r.get("max_relative_deviation", 0) > 0.10
        ],
        "saved_vs_replay_cagr": {
            "saved": saved_eval["test_cagr"],
            "fresh": evaluate_policy_weeks(all_logs["ppo"])["cagr"],
        },
        "stale_saved_allocation_comparison": saved_eval.get(
            "allocation_head_diagnostics"
        ),
    }
    after = protected_hashes()
    write_json("protected_after.json", after)
    summary["protected_files_unchanged"] = before == after
    summary["protected_file_count"] = len(before)
    assert before == after, "Production evidence changed during experiment"
    write_json("summary.json", summary)
    write_json("feedback_weekly.json", feedback_rows)
    write_json("forced_k_weekly.json", force_rows)
    pd.DataFrame(stock_rows).to_csv(ROOT / "stock_weights.csv", index=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

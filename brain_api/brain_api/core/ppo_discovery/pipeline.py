"""Two-stage historical ppo_discovery training. Writes a candidate only."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from brain_api.core.ppo_discovery.ablations import run_required_ablations
from brain_api.core.ppo_discovery.artifacts import write_candidate_artifact
from brain_api.core.ppo_discovery.config import HISTORY_BARS, PPODiscoveryConfig
from brain_api.core.ppo_discovery.environment import (
    WeeklyTransition,
    collect_closed_loop_rollout,
)
from brain_api.core.ppo_discovery.evaluator import (
    block_bootstrap_mean_ci,
    evaluate_policy_weeks,
    reject_current_patchtst_on_old_weeks,
    select_candidate_seed,
)
from brain_api.core.ppo_discovery.news_store import (
    load_weekly_news_features,
    weekly_news_path,
)
from brain_api.core.ppo_discovery.policy import PPODiscoveryActorCritic
from brain_api.core.ppo_discovery.pretraining import (
    next_week_open_log_return,
    pretrain_temporal_encoder,
)
from brain_api.core.ppo_discovery.price_features import encoder_channels_from_ohlcv
from brain_api.core.ppo_discovery.regime import (
    fit_ppo_regime_hmm,
    weekly_regime_probabilities,
)
from brain_api.core.ppo_discovery.schemas import PPODiscoveryError, UniverseSnapshot
from brain_api.core.ppo_discovery.trainer import train_ppo_discovery
from brain_api.core.ppo_discovery.weeks import (
    actor_cutoff_datetimes,
    prices_as_of,
    weekly_trade_clock,
)
from brain_api.core.prices import load_prices_yfinance
from brain_api.storage.ppo_discovery.local import PPODiscoveryHalalNewModelStorage


def run_ppo_discovery_training(
    snapshot: UniverseSnapshot,
    *,
    config: PPODiscoveryConfig,
    storage: PPODiscoveryHalalNewModelStorage,
    end_date: date,
    experiment_id: str,
    start_date: date | None = None,
    alpha_hrp_weekly_log: Sequence[float] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
    base_path: Path | str | None = None,
    skip_supervised_pretraining: bool = False,
    freeze_encoder: bool = False,
) -> dict[str, Any]:
    """Historical two-stage train. Never writes ``current``."""
    reject_current_patchtst_on_old_weeks(False)
    report = progress or (lambda _payload: None)
    symbols = list(snapshot.sorted_symbols)
    price_start = start_date or date(end_date.year - 7, 1, 1)
    report({"stage": "prices"})
    prices = load_prices_yfinance([*symbols, "SPY", "^VIX"], price_start, end_date)
    missing = [name for name in (*symbols, "SPY", "^VIX") if name not in prices]
    if missing:
        raise PPODiscoveryError(f"missing yfinance frames: {missing}")
    spy = prices["SPY"]
    clock = weekly_trade_clock(price_start, end_date)
    cutoffs = actor_cutoff_datetimes(clock)
    regimes_by_date = {}
    ohlcv = {symbol: prices[symbol] for symbol in symbols}
    transitions: list[WeeklyTransition] = []
    for index in range(len(clock.rebalance_sessions) - 1):
        cutoff = cutoffs[index]
        if not _has_encoder_history(ohlcv, symbols, cutoff.date()):
            continue
        news = load_weekly_news_features(
            cutoff, symbols, base_path=base_path or storage.base_path
        )
        transitions.append(
            WeeklyTransition(
                cutoff=cutoff,
                rebalance_session=clock.rebalance_sessions[index],
                next_rebalance_session=clock.rebalance_sessions[index + 1],
                news_by_symbol=news,
                p_calm=0.0,
                p_stress=0.0,
            )
        )
    if len(transitions) < 2:
        raise PPODiscoveryError(
            "need at least two weekly transitions after history warmup"
        )
    test_n = max(1, len(transitions) // 5)
    train_weeks = transitions[:-test_n]
    test_weeks = transitions[-test_n:]
    if not train_weeks:
        raise PPODiscoveryError("empty train split")
    hmm_cutoff = train_weeks[-1].cutoff.date()
    report({"stage": "hmm", "cutoff": hmm_cutoff.isoformat()})
    hmm = fit_ppo_regime_hmm(prices, start_date=price_start, cutoff=hmm_cutoff)
    regimes_by_date = weekly_regime_probabilities(
        hmm,
        prices,
        start_date=price_start,
        completed_through=end_date,
        weekly_cutoffs=[week.cutoff.date() for week in transitions],
    )
    train_weeks = [_with_regime(week, regimes_by_date) for week in train_weeks]
    test_weeks = [_with_regime(week, regimes_by_date) for week in test_weeks]
    scalers = _fit_count_scaler(train_weeks)
    policy = PPODiscoveryActorCritic(config)
    if not skip_supervised_pretraining:
        report({"stage": "pretrain"})
        histories, targets = _pretrain_arrays(train_weeks, snapshot, ohlcv)
        pretrain_temporal_encoder(
            policy, histories, targets, config=config, seed=config.seeds[0]
        )
    if freeze_encoder:
        from dataclasses import replace

        config = replace(config, freeze_encoder_updates=10**9)
    seed_val: dict[int, float] = {}
    seed_sharpe: dict[int, float] = {}
    seed_policies: dict[int, PPODiscoveryActorCritic] = {}
    failed_seeds: list[int] = []
    last_error: Exception | None = None
    val_weeks = train_weeks[-max(1, len(train_weeks) // 5) :]
    for seed in config.seeds:
        report({"stage": "ppo", "seed": seed})
        seed_policy = PPODiscoveryActorCritic(config)
        seed_policy.load_state_dict(policy.state_dict())
        try:
            train_ppo_discovery(
                seed_policy,
                lambda current: collect_closed_loop_rollout(
                    current,
                    train_weeks,
                    snapshot=snapshot,
                    ohlcv_by_symbol=ohlcv,
                    spy=spy,
                    feature_scalers=scalers,
                    config=config,
                ),
                config=config,
                seed=seed,
            )
            val_metrics = _eval_weeks(
                seed_policy, val_weeks, snapshot, ohlcv, spy, scalers, config
            )
        except (PPODiscoveryError, ValueError) as exc:
            failed_seeds.append(int(seed))
            last_error = exc
            continue
        seed_val[int(seed)] = val_metrics["cagr"]
        seed_sharpe[int(seed)] = val_metrics["sharpe"]
        seed_policies[int(seed)] = seed_policy
    if not seed_val:
        raise PPODiscoveryError(
            f"every ppo_discovery seed failed: {last_error}"
        ) from last_error
    chosen = select_candidate_seed(seed_val, seed_sharpe)
    chosen_policy = seed_policies[chosen]
    test_metrics = _eval_weeks(
        chosen_policy, test_weeks, snapshot, ohlcv, spy, scalers, config
    )
    test_logs = _week_logs(
        chosen_policy, test_weeks, snapshot, ohlcv, spy, scalers, config
    )
    alpha_cagr = alpha_dd = paired = None
    if alpha_hrp_weekly_log is not None:
        alpha_logs = list(alpha_hrp_weekly_log)
        if len(alpha_logs) != len(test_logs):
            raise PPODiscoveryError(
                "Alpha-HRP weekly log length must match the test split"
            )
        alpha_metrics = evaluate_policy_weeks(alpha_logs)
        alpha_cagr = alpha_metrics["cagr"]
        alpha_dd = alpha_metrics["max_drawdown"]
        paired, _lo, _hi = block_bootstrap_mean_ci(
            [float(p) - float(a) for p, a in zip(test_logs, alpha_logs, strict=True)]
        )
    report({"stage": "ablations"})
    ablations = run_required_ablations(
        chosen_policy,
        train_weeks=train_weeks,
        test_weeks=test_weeks,
        snapshot=snapshot,
        ohlcv=ohlcv,
        spy=spy,
        scalers=scalers,
        config=config,
        pretrained=policy,
    )
    evaluation = {
        "test_cagr": test_metrics["cagr"],
        "test_max_drawdown": test_metrics["max_drawdown"],
        "alpha_hrp_test_cagr": alpha_cagr,
        "alpha_hrp_test_max_drawdown": alpha_dd,
        "paired_vs_alpha_hrp_point": paired,
        "ablations": ablations,
        "failed_seeds": failed_seeds,
        "candidate": True,
        "survivorship_bias": True,
        "selected_seed": chosen,
    }
    news_manifest = {
        "complete": True,
        "parquet": str(weekly_news_path(base_path or storage.base_path)),
        "cutoffs": [week.cutoff.isoformat() for week in transitions],
    }
    price_manifest = {
        "complete": True,
        "source": "yfinance",
        "symbols": symbols,
        "start": price_start.isoformat(),
        "end": end_date.isoformat(),
        "session_count": len(spy),
    }
    report({"stage": "write_candidate"})
    version = write_candidate_artifact(
        storage,
        chosen_policy,
        config=config,
        evaluation=evaluation,
        universe_manifest=snapshot.to_dict(),
        feature_scalers=scalers,
        regime_hmm=hmm.to_dict(),
        news_manifest=news_manifest,
        price_manifest=price_manifest,
        experiment_id=experiment_id,
        end_date=end_date.isoformat(),
    )
    return {
        "version": version,
        "promoted": False,
        "universe": snapshot.universe,
        "snapshot_sha256": snapshot.snapshot_sha256,
        "evaluation": evaluation,
        "news_manifest": news_manifest,
        "price_manifest": price_manifest,
    }


def _with_regime(week: WeeklyTransition, regimes: dict) -> WeeklyTransition:
    calm, stress = regimes[week.cutoff.date()]
    return WeeklyTransition(
        cutoff=week.cutoff,
        rebalance_session=week.rebalance_session,
        next_rebalance_session=week.next_rebalance_session,
        news_by_symbol=week.news_by_symbol,
        p_calm=float(calm),
        p_stress=float(stress),
    )


def _has_encoder_history(ohlcv, symbols, cutoff: date) -> bool:
    for symbol in symbols:
        frame = ohlcv.get(symbol)
        if frame is None:
            return False
        try:
            sliced = prices_as_of(frame, cutoff)
        except PPODiscoveryError:
            return False
        if len(sliced) < HISTORY_BARS:
            return False
    return True


def _fit_count_scaler(weeks: Sequence[WeeklyTransition]) -> dict[str, dict[str, float]]:
    values = [
        float(row.log1p_article_count)
        for week in weeks
        for row in week.news_by_symbol.values()
    ]
    if not values:
        raise PPODiscoveryError("cannot fit log1p_article_count scaler on empty news")
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    scale = float(array.std(ddof=0))
    if scale < 1e-12:
        scale = 1.0
    return {"log1p_article_count": {"mean": mean, "scale": scale}}


def _pretrain_arrays(weeks, snapshot, ohlcv):
    from brain_api.core.ppo_discovery.weeks import open_to_open_return

    histories = []
    targets = []
    symbols = list(snapshot.sorted_symbols)
    for week in weeks:
        cutoff = week.cutoff.date()
        history_rows = []
        target_rows = []
        for symbol in symbols:
            sliced = prices_as_of(ohlcv[symbol], cutoff)
            history_rows.append(encoder_channels_from_ohlcv(sliced))
            start_open, simple = open_to_open_return(
                ohlcv[symbol],
                week.rebalance_session,
                week.next_rebalance_session,
                symbol=symbol,
            )
            target_rows.append(
                next_week_open_log_return(start_open, start_open * (1.0 + simple))
            )
        histories.append(np.stack(history_rows, axis=0))
        targets.append(np.asarray(target_rows, dtype=np.float64))
    return histories, targets


def _week_logs(policy, weeks, snapshot, ohlcv, spy, scalers, config) -> list[float]:
    steps = collect_closed_loop_rollout(
        policy,
        weeks,
        snapshot=snapshot,
        ohlcv_by_symbol=ohlcv,
        spy=spy,
        feature_scalers=scalers,
        config=config,
        deterministic=True,
    )
    return [step.reward for step in steps]


def _eval_weeks(
    policy, weeks, snapshot, ohlcv, spy, scalers, config
) -> dict[str, float]:
    return evaluate_policy_weeks(
        _week_logs(policy, weeks, snapshot, ohlcv, spy, scalers, config)
    )

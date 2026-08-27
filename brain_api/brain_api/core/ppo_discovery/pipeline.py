"""Two-stage historical ppo_discovery training. Writes a candidate only."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from brain_api.core.ppo_discovery.ablations import run_required_ablations
from brain_api.core.ppo_discovery.artifacts import write_candidate_artifact
from brain_api.core.ppo_discovery.baselines import locked_random_test_metrics
from brain_api.core.ppo_discovery.checkpoints import (
    hash_state_dict,
    load_seed_checkpoint,
    model_config_hash,
    save_seed_checkpoint,
    seed_checkpoint_dir,
)
from brain_api.core.ppo_discovery.config import (
    ENCODER_CHANNELS,
    HISTORY_BARS,
    MIN_ELIGIBLE_ASSETS,
    PPODiscoveryConfig,
)
from brain_api.core.ppo_discovery.dataset_identity import build_dataset_identity
from brain_api.core.ppo_discovery.environment import (
    WeeklyTransition,
    collect_closed_loop_rollout,
)
from brain_api.core.ppo_discovery.evaluator import (
    aggregate_seed_metrics,
    block_bootstrap_mean_ci,
    evaluate_policy_weeks,
    reject_current_patchtst_on_old_weeks,
    select_candidate_seed,
)
from brain_api.core.ppo_discovery.matched_k import matched_k_average_rank
from brain_api.core.ppo_discovery.news_adapter import load_weekly_ppo_news_features
from brain_api.core.ppo_discovery.policy import PPODiscoveryActorCritic
from brain_api.core.ppo_discovery.pretraining import (
    next_week_open_log_return,
    pretrain_temporal_encoder,
)
from brain_api.core.ppo_discovery.price_features import (
    apply_encoder_channel_scaler,
    encoder_channels_from_ohlcv,
)
from brain_api.core.ppo_discovery.promotion import (
    ppo_discovery_source_digest,
    protocol_file_digest,
)
from brain_api.core.ppo_discovery.regime import (
    fit_ppo_regime_hmm,
    weekly_regime_probabilities,
)
from brain_api.core.ppo_discovery.schemas import PPODiscoveryError, UniverseSnapshot
from brain_api.core.ppo_discovery.splits import (
    FULL_VARIANT,
    is_locked_full_training,
    split_walk_forward,
)
from brain_api.core.ppo_discovery.trainer import train_ppo_discovery
from brain_api.core.ppo_discovery.weeks import (
    actor_cutoff_datetimes,
    news_window_starts_at_or_after_archive,
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
    experiment_variant: str = FULL_VARIANT,
) -> dict[str, Any]:
    """Historical two-stage train. Never writes ``current``."""
    reject_current_patchtst_on_old_weeks(False)
    if experiment_variant == FULL_VARIANT and not is_locked_full_training(
        config,
        skip_supervised_pretraining=skip_supervised_pretraining,
        freeze_encoder=freeze_encoder,
    ):
        raise PPODiscoveryError(
            "experiment_variant='full' requires locked default seeds and "
            "10_000 timesteps with both training stages"
        )
    report = progress or (lambda _payload: None)
    symbols = list(snapshot.sorted_symbols)
    price_start = start_date or date(end_date.year - 7, 1, 1)
    report({"stage": "prices"})
    prices = load_prices_yfinance([*symbols, "SPY", "^VIX"], price_start, end_date)
    ohlcv = _ohlcv_for_training(prices, symbols)
    spy = prices["SPY"]
    clock = weekly_trade_clock(price_start, end_date)
    cutoffs = actor_cutoff_datetimes(clock)
    regimes_by_date = {}
    transitions: list[WeeklyTransition] = []
    for index in range(len(clock.rebalance_sessions) - 1):
        cutoff = cutoffs[index]
        if not news_window_starts_at_or_after_archive(cutoff):
            continue
        if _eligible_count(ohlcv, symbols, cutoff.date()) < MIN_ELIGIBLE_ASSETS:
            continue
        news = load_weekly_ppo_news_features(cutoff, symbols)
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
    if len(transitions) < 5:
        raise PPODiscoveryError(
            "need at least five weekly transitions after history warmup"
        )
    train_weeks, val_weeks, test_weeks = split_walk_forward(
        transitions, experiment_variant=experiment_variant
    )
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
    val_weeks = [_with_regime(week, regimes_by_date) for week in val_weeks]
    test_weeks = [_with_regime(week, regimes_by_date) for week in test_weeks]
    identity = build_dataset_identity(
        train_weeks,
        val_weeks,
        test_weeks,
        snapshot=snapshot,
        ohlcv=ohlcv,
        spy=spy,
    )
    scalers = _fit_feature_scalers(train_weeks, snapshot, ohlcv)
    policy = PPODiscoveryActorCritic(config)
    if not skip_supervised_pretraining:
        report({"stage": "pretrain"})
        histories, targets = _pretrain_arrays(
            train_weeks, snapshot, ohlcv, feature_scalers=scalers
        )
        pretrain_temporal_encoder(
            policy, histories, targets, config=config, seed=config.seeds[0]
        )
    pretrained_encoder_state = {
        key: tensor.detach().cpu().clone()
        for key, tensor in policy.temporal.state_dict().items()
    }
    if freeze_encoder:
        from dataclasses import replace

        config = replace(config, freeze_encoder_updates=10**9)
    if base_path is None:
        from brain_api.storage.base import DEFAULT_DATA_PATH

        checkpoint_root = Path(DEFAULT_DATA_PATH)
    else:
        checkpoint_root = Path(base_path)
    ckpt_dir = seed_checkpoint_dir(
        checkpoint_root,
        experiment_id=experiment_id,
        snapshot_hash=snapshot.snapshot_sha256,
        config_hash=model_config_hash(config),
    )
    checkpoint_expected = {
        "protocol_digest": protocol_file_digest(),
        "training_dataset_hash": identity.training_dataset_hash,
        "snapshot_sha256": snapshot.snapshot_sha256,
        "model_config_hash": model_config_hash(config),
        "code_revision": ppo_discovery_source_digest(),
        "pretrained_encoder_sha256": hash_state_dict(pretrained_encoder_state),
    }
    seed_val: dict[int, float] = {}
    seed_sharpe: dict[int, float] = {}
    seed_policies: dict[int, PPODiscoveryActorCritic] = {}
    failed_seeds: list[int] = []
    last_error: Exception | None = None
    for seed in config.seeds:
        report({"stage": "ppo", "seed": seed})
        seed_policy = PPODiscoveryActorCritic(config)
        seed_policy.load_state_dict(policy.state_dict())
        loaded = load_seed_checkpoint(
            ckpt_dir, seed=int(seed), expected=checkpoint_expected
        )
        try:
            if loaded is not None:
                seed_policy.load_state_dict(loaded["state_dict"])
                report({"stage": "ppo_resume", "seed": seed})
            else:
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
                save_seed_checkpoint(
                    ckpt_dir,
                    seed=int(seed),
                    policy=seed_policy,
                    metadata={
                        "experiment_id": experiment_id,
                        **checkpoint_expected,
                    },
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
    report({"stage": "matched_k"})
    matched_k = matched_k_average_rank(
        chosen_policy,
        test_weeks=test_weeks,
        snapshot=snapshot,
        ohlcv=ohlcv,
        spy=spy,
        scalers=scalers,
        config=config,
    )
    random_baseline = locked_random_test_metrics(
        test_weeks,
        snapshot=snapshot,
        ohlcv=ohlcv,
        spy=spy,
        scalers=scalers,
        config=config,
    )
    seed_metrics = {
        str(seed): {
            "val_cagr": seed_val[seed],
            "val_sharpe": seed_sharpe[seed],
        }
        for seed in seed_val
    }
    evaluation = {
        "test_cagr": test_metrics["cagr"],
        "test_max_drawdown": test_metrics["max_drawdown"],
        "test_weekly_net_log": test_logs,
        "alpha_hrp_test_cagr": alpha_cagr,
        "alpha_hrp_test_max_drawdown": alpha_dd,
        "paired_vs_alpha_hrp_point": paired,
        "ablations": ablations,
        "matched_k": matched_k,
        "failed_seeds": failed_seeds,
        "candidate": True,
        "survivorship_bias": True,
        "selected_seed": chosen,
        "seed_metrics": seed_metrics,
        "seed_aggregates": {
            "val_cagr": aggregate_seed_metrics(seed_val),
            "val_sharpe": aggregate_seed_metrics(seed_sharpe),
            "n_seeds": len(seed_val),
        },
        "locked_random_baseline": random_baseline,
        "training_dataset_hash": identity.training_dataset_hash,
        "validation_dataset_hash": identity.validation_dataset_hash,
        "evaluation_dataset_hash": identity.evaluation_dataset_hash,
    }
    news_manifest = {
        "complete": True,
        "store": "duckdb",
        "cutoffs": [week.cutoff.isoformat() for week in transitions],
        "training_dataset_hash": identity.training_dataset_hash,
        "validation_dataset_hash": identity.validation_dataset_hash,
        "evaluation_dataset_hash": identity.evaluation_dataset_hash,
        "weeks": identity.news_weeks,
    }
    session_dates = ",".join(pd.Timestamp(ts).strftime("%Y-%m-%d") for ts in spy.index)
    price_manifest = {
        "complete": True,
        "source": "yfinance",
        "symbols": symbols,
        "start": price_start.isoformat(),
        "end": end_date.isoformat(),
        "session_count": len(spy),
        "session_dates_sha256": hashlib.sha256(
            session_dates.encode("utf-8")
        ).hexdigest(),
        "symbol_session_hashes": identity.price_sessions,
        "training_dataset_hash": identity.training_dataset_hash,
        "validation_dataset_hash": identity.validation_dataset_hash,
        "evaluation_dataset_hash": identity.evaluation_dataset_hash,
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
        experiment_variant=experiment_variant,
        end_date=end_date.isoformat(),
        pretrained_encoder_state_dict=pretrained_encoder_state,
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


def _ohlcv_for_training(
    prices: dict[str, pd.DataFrame], symbols: Sequence[str]
) -> dict[str, pd.DataFrame]:
    """Require SPY and VIX; omit missing stock frames so they mask later."""
    missing_index = [name for name in ("SPY", "^VIX") if name not in prices]
    if missing_index:
        raise PPODiscoveryError(f"missing yfinance frames: {missing_index}")
    return {symbol: prices[symbol] for symbol in symbols if symbol in prices}


def _eligible_count(ohlcv, symbols, cutoff: date) -> int:
    count = 0
    for symbol in symbols:
        frame = ohlcv.get(symbol)
        if frame is None:
            continue
        try:
            sliced = prices_as_of(frame, cutoff)
        except PPODiscoveryError:
            continue
        if len(sliced) >= HISTORY_BARS:
            count += 1
    return count


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


def _fit_feature_scalers(
    weeks: Sequence[WeeklyTransition], snapshot: UniverseSnapshot, ohlcv
) -> dict[str, Any]:
    """Fit news-count and per-channel encoder scalers on the train fold only."""
    scalers: dict[str, Any] = dict(_fit_count_scaler(weeks))
    channel_rows: list[np.ndarray] = []
    for week in weeks:
        cutoff = week.cutoff.date()
        for symbol in snapshot.sorted_symbols:
            frame = ohlcv.get(symbol)
            if frame is None:
                continue
            try:
                tensor = encoder_channels_from_ohlcv(prices_as_of(frame, cutoff))
            except PPODiscoveryError:
                continue
            channel_rows.append(tensor.reshape(-1, ENCODER_CHANNELS))
    if not channel_rows:
        raise PPODiscoveryError(
            "cannot fit encoder_channels scaler on an empty train fold"
        )
    stacked = np.concatenate(channel_rows, axis=0)
    mean = stacked.mean(axis=0)
    scale = stacked.std(axis=0, ddof=0)
    scale = np.maximum(scale, 1e-12)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
        raise PPODiscoveryError("encoder_channels scaler is non-finite")
    scalers["encoder_channels"] = {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
    }
    return scalers


def _pretrain_arrays(weeks, snapshot, ohlcv, feature_scalers=None):
    from brain_api.core.ppo_discovery.weeks import open_to_open_return

    histories = []
    targets = []
    symbols = list(snapshot.sorted_symbols)
    for week in weeks:
        cutoff = week.cutoff.date()
        history_rows = []
        target_rows = []
        for symbol in symbols:
            frame = ohlcv.get(symbol)
            if frame is None:
                continue
            try:
                sliced = prices_as_of(frame, cutoff)
                history = apply_encoder_channel_scaler(
                    encoder_channels_from_ohlcv(sliced), feature_scalers
                )
                start_open, simple = open_to_open_return(
                    frame,
                    week.rebalance_session,
                    week.next_rebalance_session,
                    symbol=symbol,
                )
                target = next_week_open_log_return(
                    start_open, start_open * (1.0 + simple)
                )
            except PPODiscoveryError:
                continue
            history_rows.append(history)
            target_rows.append(target)
        if not history_rows:
            continue
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
    return [step.realized_net_return for step in steps]


def _eval_weeks(
    policy, weeks, snapshot, ohlcv, spy, scalers, config
) -> dict[str, float]:
    return evaluate_policy_weeks(
        _week_logs(policy, weeks, snapshot, ohlcv, spy, scalers, config)
    )

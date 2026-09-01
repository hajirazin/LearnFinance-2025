"""Two-stage historical ppo_discovery training. Writes a candidate only."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from brain_api.core.ppo_discovery.ablations import run_required_ablations
from brain_api.core.ppo_discovery.artifacts import write_candidate_artifact
from brain_api.core.ppo_discovery.baselines import locked_random_test_metrics
from brain_api.core.ppo_discovery.checkpoints import (
    hash_state_dict,
    model_config_hash,
    seed_checkpoint_dir,
    train_recipe_hash,
)
from brain_api.core.ppo_discovery.config import (
    MIN_ELIGIBLE_ASSETS,
    PPODiscoveryConfig,
)
from brain_api.core.ppo_discovery.dataset_identity import build_dataset_identity
from brain_api.core.ppo_discovery.environment import WeeklyTransition
from brain_api.core.ppo_discovery.evaluator import (
    block_bootstrap_mean_ci,
    evaluate_policy_weeks,
    reject_current_patchtst_on_old_weeks,
)
from brain_api.core.ppo_discovery.matched_k import matched_k_average_rank
from brain_api.core.ppo_discovery.news_adapter import (
    load_historical_ppo_news_features,
)
from brain_api.core.ppo_discovery.policy import PPODiscoveryActorCritic
from brain_api.core.ppo_discovery.pretraining import pretrain_temporal_encoder
from brain_api.core.ppo_discovery.promotion import (
    ppo_discovery_source_digest,
    protocol_file_digest,
)
from brain_api.core.ppo_discovery.regime import (
    fit_ppo_regime_hmm,
    weekly_regime_probabilities,
)
from brain_api.core.ppo_discovery.schemas import PPODiscoveryError, UniverseSnapshot
from brain_api.core.ppo_discovery.seed_ledger import (
    empty_seeds_ledger,
    fail_job_on_accelerator_oom,
)
from brain_api.core.ppo_discovery.seed_training import (
    train_ppo_discovery_seeds,
    week_logs,
)
from brain_api.core.ppo_discovery.splits import (
    FULL_VARIANT,
    is_locked_full_training,
    split_walk_forward,
)
from brain_api.core.ppo_discovery.training_features import (
    eligible_count,
    fit_feature_scalers,
    ohlcv_for_training,
    pretrain_arrays,
    with_regime,
)
from brain_api.core.ppo_discovery.weeks import (
    actor_cutoff_datetimes,
    news_window_starts_at_or_after_archive,
    weekly_trade_clock,
)
from brain_api.core.prices import load_prices_yfinance
from brain_api.core.sac.market_sessions import xnys_session_dates
from brain_api.core.training_utils import get_device, is_accelerator_out_of_memory
from brain_api.core.vix_fallback import (
    VixFallbackAudit,
    VixFallbackResult,
    apply_cboe_vix_fallback,
)
from brain_api.storage.ppo_discovery.local import PPODiscoveryHalalNewModelStorage


def repair_ppo_training_vix(
    prices: dict[str, pd.DataFrame],
    *,
    price_start: date,
    hmm_weeks: Sequence[date],
) -> VixFallbackResult:
    """Repair only index sessions consumed by PPO's weekly HMM states."""
    return apply_cboe_vix_fallback(
        prices,
        required_dates=xnys_session_dates(price_start, max(hmm_weeks)),
    )


def price_manifest_with_vix_audit(
    manifest: dict[str, Any], audit: VixFallbackAudit
) -> dict[str, Any]:
    """Add fallback provenance without changing the primary source contract."""
    return {**manifest, "vix_provenance": audit.to_dict()}


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
    device = get_device()
    line = (
        f"[PPO] device={device.type} pytorch={torch.__version__} "
        f"mps_built={torch.backends.mps.is_built()} "
        f"mps_available={torch.backends.mps.is_available()} "
        f"cuda_available={torch.cuda.is_available()}"
    )
    print(line, flush=True)
    report({"stage": "device", "device": device.type})
    symbols = list(snapshot.sorted_symbols)
    price_start = start_date or date(end_date.year - 7, 1, 1)
    report({"stage": "prices"})
    prices = load_prices_yfinance([*symbols, "SPY", "^VIX"], price_start, end_date)
    clock = weekly_trade_clock(price_start, end_date)
    cutoffs = actor_cutoff_datetimes(clock)
    ohlcv = ohlcv_for_training(prices, symbols)
    regimes_by_date = {}
    eligible_rows: list[tuple[int, datetime]] = []
    for index in range(len(clock.rebalance_sessions) - 1):
        cutoff = cutoffs[index]
        if not news_window_starts_at_or_after_archive(cutoff):
            continue
        if eligible_count(ohlcv, symbols, cutoff.date()) < MIN_ELIGIBLE_ASSETS:
            continue
        eligible_rows.append((index, cutoff))
    report(
        {
            "stage": "news",
            "weeks_total": len(eligible_rows),
            "symbols_total": len(symbols),
        }
    )
    historical_news = load_historical_ppo_news_features(
        [cutoff for _index, cutoff in eligible_rows], symbols
    )
    transitions: list[WeeklyTransition] = []
    for completed, (index, cutoff) in enumerate(eligible_rows, start=1):
        transitions.append(
            WeeklyTransition(
                cutoff=cutoff,
                rebalance_session=clock.rebalance_sessions[index],
                next_rebalance_session=clock.rebalance_sessions[index + 1],
                news_by_symbol=historical_news[cutoff],
                p_calm=0.0,
                p_stress=0.0,
            )
        )
        if completed % 25 == 0 or completed == len(eligible_rows):
            report(
                {
                    "stage": "transitions",
                    "weeks_completed": completed,
                    "weeks_total": len(eligible_rows),
                }
            )
    if len(transitions) < 5:
        raise PPODiscoveryError(
            "need at least five weekly transitions after history warmup"
        )
    train_weeks, val_weeks, test_weeks = split_walk_forward(
        transitions, experiment_variant=experiment_variant
    )
    hmm_cutoff = train_weeks[-1].cutoff.date()
    hmm_weeks = [week.cutoff.date() for week in transitions]
    # Training consumes index evidence only through its final usable actor
    # cutoff. Trailing provider rows after this date cannot block training.
    vix_result = repair_ppo_training_vix(
        prices,
        price_start=price_start,
        hmm_weeks=hmm_weeks,
    )
    prices = vix_result.prices
    spy = prices["SPY"]
    n_obs = max(0, len(spy) - 20)
    report({"stage": "hmm", "cutoff": hmm_cutoff.isoformat()})
    print(
        f"[PPO] hmm start cutoff={hmm_cutoff.isoformat()} "
        f"observations={n_obs} weeks={len(hmm_weeks)} device=cpu",
        flush=True,
    )
    hmm = fit_ppo_regime_hmm(prices, start_date=price_start, cutoff=hmm_cutoff)
    regimes_by_date = weekly_regime_probabilities(
        hmm,
        prices,
        start_date=price_start,
        completed_through=max(hmm_weeks),
        weekly_cutoffs=hmm_weeks,
    )
    print(
        f"[PPO] hmm complete cutoff={hmm_cutoff.isoformat()} "
        f"observations={n_obs} weeks={len(regimes_by_date)} device=cpu",
        flush=True,
    )
    train_weeks = [with_regime(week, regimes_by_date) for week in train_weeks]
    val_weeks = [with_regime(week, regimes_by_date) for week in val_weeks]
    test_weeks = [with_regime(week, regimes_by_date) for week in test_weeks]
    identity = build_dataset_identity(
        train_weeks,
        val_weeks,
        test_weeks,
        snapshot=snapshot,
        ohlcv=ohlcv,
        spy=spy,
    )
    if base_path is None:
        from brain_api.storage.base import DEFAULT_DATA_PATH

        checkpoint_root = Path(DEFAULT_DATA_PATH)
    else:
        checkpoint_root = Path(base_path)
    if freeze_encoder:
        from dataclasses import replace

        config = replace(config, freeze_encoder_updates=10**9)
    recipe = train_recipe_hash(config)
    ckpt_dir = seed_checkpoint_dir(
        checkpoint_root,
        experiment_id=experiment_id,
        snapshot_hash=snapshot.snapshot_sha256,
        recipe_hash=recipe,
    )
    scalers = fit_feature_scalers(train_weeks, snapshot, ohlcv)
    torch.manual_seed(config.seeds[0])
    np.random.seed(config.seeds[0])
    policy = PPODiscoveryActorCritic(config).to(device)
    if not skip_supervised_pretraining:
        report({"stage": "pretrain", "device": device.type})
        histories, targets = pretrain_arrays(
            train_weeks, snapshot, ohlcv, feature_scalers=scalers
        )
        try:
            pretrain_temporal_encoder(
                policy,
                histories,
                targets,
                config=config,
                seed=config.seeds[0],
                device=device,
            )
        except Exception as exc:
            if is_accelerator_out_of_memory(exc, device):
                fail_job_on_accelerator_oom(
                    exc,
                    seed=int(config.seeds[0]),
                    device=device,
                    directory=ckpt_dir,
                    ledger=empty_seeds_ledger(),
                    checkpoint_expected={},
                    progress=report,
                )
            raise
    pretrained_encoder_state = {
        key: tensor.detach().cpu().clone()
        for key, tensor in policy.temporal.state_dict().items()
    }
    checkpoint_expected = {
        "train_recipe_hash": recipe,
        "protocol_digest": protocol_file_digest(),
        "training_dataset_hash": identity.training_dataset_hash,
        "snapshot_sha256": snapshot.snapshot_sha256,
        "code_revision": ppo_discovery_source_digest(),
        "pretrained_encoder_sha256": hash_state_dict(pretrained_encoder_state),
    }
    pretrained_full_state = {
        key: tensor.detach().cpu().clone()
        for key, tensor in policy.state_dict().items()
    }
    seed_result = train_ppo_discovery_seeds(
        pretrained_state=pretrained_full_state,
        train_weeks=train_weeks,
        val_weeks=val_weeks,
        snapshot=snapshot,
        ohlcv=ohlcv,
        spy=spy,
        scalers=scalers,
        config=config,
        ckpt_dir=ckpt_dir,
        checkpoint_expected=checkpoint_expected,
        experiment_id=experiment_id,
        device=device,
        progress=report,
    )
    chosen = seed_result.selected_seed
    chosen_policy = seed_result.selected_policy
    failed_seeds = seed_result.failed_seeds
    try:
        test_logs = week_logs(
            chosen_policy, test_weeks, snapshot, ohlcv, spy, scalers, config
        )
        test_metrics = evaluate_policy_weeks(test_logs)
    except Exception as exc:
        if is_accelerator_out_of_memory(exc, device):
            fail_job_on_accelerator_oom(
                exc,
                seed=int(chosen),
                device=device,
                directory=ckpt_dir,
                ledger=seed_result.ledger,
                checkpoint_expected=dict(checkpoint_expected),
                progress=report,
            )
        raise
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
    try:
        report({"stage": "ablations", "device": device.type})
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
        report({"stage": "matched_k", "device": device.type})
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
    except Exception as exc:
        if is_accelerator_out_of_memory(exc, device):
            fail_job_on_accelerator_oom(
                exc,
                seed=int(chosen),
                device=device,
                directory=ckpt_dir,
                ledger=seed_result.ledger,
                checkpoint_expected=dict(checkpoint_expected),
                progress=report,
            )
        raise
    evaluation = {
        "test_cagr": test_metrics["cagr"],
        "test_sharpe": test_metrics["sharpe"],
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
        "seed_metrics": seed_result.seed_metrics,
        "seed_aggregates": seed_result.seed_aggregates,
        "locked_random_baseline": random_baseline,
        "training_dataset_hash": identity.training_dataset_hash,
        "validation_dataset_hash": identity.validation_dataset_hash,
        "evaluation_dataset_hash": identity.evaluation_dataset_hash,
        "device": device.type,
        "experiment_seeds": list(config.seeds),
        "model_config_hash": model_config_hash(config),
        "train_recipe_hash": recipe,
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
    price_manifest = price_manifest_with_vix_audit(
        {
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
        },
        vix_result.audit,
    )
    report({"stage": "write_candidate", "device": device.type})
    print("[PPO] writing candidate artifact", flush=True)
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
        seeds_ledger=seed_result.ledger,
    )
    print(f"[PPO] wrote candidate {version}", flush=True)
    failure_reasons = list(storage.load_artifacts(version).metadata["failure_reasons"])
    return {
        "version": version,
        "promoted": False,
        "universe": snapshot.universe,
        "snapshot_sha256": snapshot.snapshot_sha256,
        "evaluation": evaluation,
        "news_manifest": news_manifest,
        "price_manifest": price_manifest,
        "failure_reasons": failure_reasons,
        "selected_seed": chosen,
        "experiment_seeds": list(config.seeds),
        "seed_metrics": seed_result.seed_metrics,
        "failed_seeds": failed_seeds,
        "device": device.type,
    }


_ohlcv_for_training = ohlcv_for_training

"""Build the annual inventory for the weekly PatchTST regressor."""

import logging
from datetime import date

from brain_api.core.forecaster_snapshot_identity import (
    annual_snapshot_cutoffs,
    annual_snapshot_parity,
)
from brain_api.core.patchtst.training_evidence import validate_label_evidence
from brain_api.core.patchtst.weekly_data import (
    InsufficientWeeklyHistory,
    load_weekly_training_inputs,
)
from brain_api.core.patchtst.weekly_training import fit_weekly_patchtst
from brain_api.core.version import compute_snapshot_identity_hash
from brain_api.news.store import NewsStore
from brain_api.news.sync import sync_news
from brain_api.routes.training.snapshot_persist import persist_forecaster_snapshot
from brain_api.storage.forecaster_snapshots import create_snapshot_metadata

logger = logging.getLogger(__name__)
EARLIEST_US_SNAPSHOT = date(2017, 12, 31)


def backfill_weekly_patchtst_snapshots(
    storage,
    symbols,
    config,
    start_date,
    end_date,
    price_loader,
    *,
    log_prefix,
) -> date:
    cutoffs = annual_snapshot_cutoffs(start_date, end_date)
    if config.market == "us":
        cutoffs = [cutoff for cutoff in cutoffs if cutoff >= EARLIEST_US_SNAPSHOT]
    if not cutoffs:
        raise ValueError("No supported annual PatchTST snapshot cutoff")
    store = NewsStore(storage.base_path) if config.market == "us" else None
    news_commit = sync_news(store) if store is not None else None
    history_start = date(cutoffs[0].year - 4, 1, 1)
    if config.market == "us":
        history_start = max(history_start, date(2015, 1, 5))
    inputs = None
    first = None
    hf_repo = storage._get_hf_repo()
    for cutoff in cutoffs:
        digest = compute_snapshot_identity_hash(
            storage.forecaster_type, cutoff, config.to_dict()
        )
        action = annual_snapshot_parity(storage, cutoff, digest)
        if action in ("ready", "upload"):
            metadata = storage.read_metadata(cutoff) or {}
            validate_label_evidence(
                metadata.get("training_evidence", {}),
                config.market,
                cutoff,
                len(symbols),
            )
        if action == "upload" and not storage.upload_snapshot_to_hf(cutoff, digest):
            raise ValueError(f"Weekly PatchTST snapshot upload failed for {cutoff}")
        if action == "download" and not storage.download_snapshot_from_hf(
            cutoff, digest
        ):
            raise ValueError(f"Weekly PatchTST snapshot download failed for {cutoff}")
        if action == "download":
            validate_label_evidence(
                (storage.read_metadata(cutoff) or {}).get("training_evidence", {}),
                config.market,
                cutoff,
                len(symbols),
            )
        if action == "train":
            if inputs is None:
                inputs = load_weekly_training_inputs(
                    symbols,
                    history_start,
                    end_date,
                    config,
                    price_loader,
                    store=store,
                )
            try:
                dataset = inputs.labeled(history_start, cutoff)
                result = fit_weekly_patchtst(dataset, config)
            except InsufficientWeeklyHistory:
                if first is not None or config.market != "us":
                    raise
                logger.warning("%s Unsupported starting cutoff %s", log_prefix, cutoff)
                continue
            del dataset
            metadata = create_snapshot_metadata(
                forecaster_type=storage.forecaster_type,
                cutoff_date=cutoff,
                data_window_start=history_start.isoformat(),
                data_window_end=cutoff.isoformat(),
                symbols=symbols,
                config=config,
                train_loss=result.train_loss,
                val_loss=result.val_loss,
                best_epoch=result.best_epoch,
                stopped_epoch=result.stopped_epoch,
                config_symbols_hash=digest,
            )
            metadata["training_evidence"] = {
                **result.evidence,
                "news_commit": news_commit,
            }
            metadata["metrics"]["weekly_rmse_pp"] = result.evidence["weekly_rmse_pp"]
            saved = persist_forecaster_snapshot(
                snapshot_storage=storage,
                cutoff_date=cutoff,
                snapshot_digest=digest,
                model=result.model,
                feature_scaler=result.feature_scaler,
                config=config,
                metadata=metadata,
                train_loss=result.train_loss,
                val_loss=result.val_loss,
                snapshot_hf_repo=hf_repo,
                log_prefix=log_prefix,
            )
            if not saved.is_canonical or (
                hf_repo and not storage.snapshot_digest_exists_on_hf(cutoff, digest)
            ):
                raise ValueError(f"Weekly PatchTST snapshot is not ready for {cutoff}")
            del result
        first = first or cutoff
    if first is None:
        raise InsufficientWeeklyHistory(
            "No qualifying annual US PatchTST snapshot was produced"
        )
    return first

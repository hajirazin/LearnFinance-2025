"""Validate the staged weekly PatchTST and SAC release and publish its artifacts."""

import argparse
import json
import math
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from brain_api.core.forecaster_snapshot_identity import annual_snapshot_cutoffs
from brain_api.core.model_buckets import ModelType, get_bucket
from brain_api.core.patchtst.config import (
    INDIA_WEEKLY_CONFIG,
    US_WEEKLY_CONFIG,
    patchtst_recipe,
)
from brain_api.core.patchtst.training_evidence import validate_label_evidence
from brain_api.core.sac.promotion import SAC_PROMOTION_CAGR_FLOOR
from brain_api.core.version import compute_snapshot_identity_hash
from brain_api.storage.forecaster_snapshots import SnapshotLocalStorage


def validate_release(data_dir: Path):
    forecasters = []
    for universe, config in (
        ("halal_new", US_WEEKLY_CONFIG),
        ("nifty_shariah_500", INDIA_WEEKLY_CONFIG),
    ):
        bucket = get_bucket(ModelType.PATCHTST, universe)
        storage = bucket.local_storage_class(data_dir)
        version = storage.read_current_version()
        if not version:
            raise ValueError(f"Missing staged {bucket.bucket_name} checkpoint")
        model = storage.load_current_artifacts()
        metadata = storage.read_metadata(version) or {}
        if metadata.get("promoted") is not True or patchtst_recipe(
            model.config
        ) != patchtst_recipe(config):
            raise ValueError(
                f"Staged {bucket.bucket_name} is not a healthy weekly regressor"
            )
        if universe == "halal_new" and not metadata["training_evidence"].get(
            "news_commit"
        ):
            raise ValueError("Staged US PatchTST lacks a shared news dataset commit")
        first = date.fromisoformat(
            metadata["training_evidence"]["first_snapshot_cutoff"]
        )
        end = date.fromisoformat(metadata["data_window"]["end"])
        validate_label_evidence(
            metadata["training_evidence"], config.market, end, len(metadata["symbols"])
        )
        snapshot_store = SnapshotLocalStorage(bucket.bucket_name, base_path=data_dir)
        snapshots = []
        for cutoff in annual_snapshot_cutoffs(date(first.year + 1, 1, 1), end):
            digest = compute_snapshot_identity_hash(
                bucket.bucket_name, cutoff, config.to_dict()
            )
            if not snapshot_store.snapshot_exists(cutoff, digest):
                raise ValueError(
                    f"{bucket.bucket_name} lacks {cutoff} snapshot {digest}"
                )
            snapshot = snapshot_store.load_snapshot(cutoff)
            snapshot_metadata = snapshot_store.read_metadata(cutoff) or {}
            validate_label_evidence(
                snapshot_metadata.get("training_evidence", {}),
                config.market,
                cutoff,
                len(metadata["symbols"]),
            )
            if patchtst_recipe(snapshot.config) != patchtst_recipe(config):
                raise ValueError(
                    f"{bucket.bucket_name} has an incompatible {cutoff} snapshot"
                )
            snapshots.append((cutoff, digest))
        if not snapshots:
            raise ValueError(f"{bucket.bucket_name} has no staged annual snapshots")
        forecasters.append(
            (bucket, storage, model, metadata, snapshot_store, snapshots)
        )

    sac_models = []
    for universe in ("halal_filtered", "halal"):
        bucket = get_bucket(ModelType.SAC, universe)
        storage = bucket.local_storage_class(data_dir)
        version = storage.read_current_version()
        if not version:
            raise ValueError(f"Missing staged {bucket.bucket_name} policy")
        model = storage.load_artifacts(version)
        metadata = model.metadata
        cagr = float(metadata.get("metrics", {}).get("eval_cagr", math.nan))
        if (
            metadata.get("promoted") is not True
            or metadata.get("patchtst_recipe") != patchtst_recipe(US_WEEKLY_CONFIG)
            or not math.isfinite(cagr)
            or cagr < SAC_PROMOTION_CAGR_FLOOR
        ):
            raise ValueError(
                f"Staged {bucket.bucket_name} failed weekly release checks"
            )
        sac_models.append((bucket, storage, model))

    cache = data_dir / "cache" / "universe"
    month = date.today().strftime("%Y-%m")
    filtered = json.loads((cache / f"halal_filtered_{month}.json").read_text())
    india = json.loads((cache / f"halal_india_{month}.json").read_text())
    if (
        filtered["model_version"] != forecasters[0][2].version
        or sorted(item["symbol"] for item in filtered["stocks"])
        != sac_models[0][2].symbol_order
        or india["model_version"] != forecasters[1][2].version
    ):
        raise ValueError("Staged monthly screens do not match the release artifacts")
    return forecasters, sac_models


def publish_release(forecasters, sac_models, *, make_current: bool) -> None:
    # Resolve all four destinations before any upload.
    buckets = [item[0] for item in [*forecasters, *sac_models]]
    repo_ids = {bucket.bucket_name: bucket.hf_repo_getter() for bucket in buckets}
    if any(not repo_id for repo_id in repo_ids.values()):
        raise ValueError("Configure all four model repositories before publishing")
    for bucket, storage, model, metadata, snapshot_store, snapshots in forecasters:
        hf = bucket.hf_storage_class(
            repo_id=repo_ids[bucket.bucket_name], local_cache=storage
        )
        for cutoff, digest in snapshots:
            if not snapshot_store.upload_snapshot_to_hf(cutoff, digest):
                raise ValueError(
                    f"Failed to upload {bucket.bucket_name} snapshot for {cutoff}"
                )
        hf.upload_model(
            model.version,
            model.model,
            model.feature_scaler,
            model.config,
            metadata,
            make_current=make_current,
        )
    for bucket, storage, model in sac_models:
        hf = bucket.hf_storage_class(
            repo_id=repo_ids[bucket.bucket_name], local_cache=storage
        )
        hf.upload_model(
            version=model.version,
            actor=model.actor,
            critic=model.critic,
            critic_target=model.critic_target,
            log_alpha=model.log_alpha,
            scaler=model.scaler,
            config=model.config,
            symbol_order=model.symbol_order,
            metadata=model.metadata,
            v3_auxiliary=model.v3_auxiliary,
            make_current=make_current,
        )


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="Isolated staging data directory"
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--publish", action="store_true", help="Publish versioned models and snapshots"
    )
    actions.add_argument(
        "--activate-hf",
        action="store_true",
        help="Move the HF main pointers during the paused paper maintenance window",
    )
    args = parser.parse_args()
    try:
        forecasters, sac_models = validate_release(args.data_dir)
        if args.publish or args.activate_hf:
            publish_release(forecasters, sac_models, make_current=args.activate_hf)
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Release check failed: {exc}\n")
    print(
        json.dumps(
            {
                item[0].bucket_name: item[2].version
                for item in [*forecasters, *sac_models]
            }
        )
    )


if __name__ == "__main__":
    main()

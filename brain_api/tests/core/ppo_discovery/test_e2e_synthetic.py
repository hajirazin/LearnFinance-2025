"""Synthetic e2e: candidate write, promote gates, inference, no PatchTST I/O."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from brain_api.core.ppo_discovery.artifacts import write_candidate_artifact
from brain_api.core.ppo_discovery.config import (
    ASSET_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    PPO_DISCOVERY_BROKER_COST_MODEL,
    PPO_DISCOVERY_TRAINING_NAV_USD,
    REQUIRED_ABLATIONS,
    PPODiscoveryConfig,
)
from brain_api.core.ppo_discovery.inference import (
    reject_schema_mismatch,
    run_ppo_discovery_inference,
)
from brain_api.core.ppo_discovery.policy import PPODiscoveryActorCritic
from brain_api.core.ppo_discovery.promotion import (
    evaluate_ppo_discovery_promotion,
    reevaluate_ppo_discovery,
    result_hash,
)
from brain_api.core.ppo_discovery.schemas import PPODiscoveryError
from brain_api.core.ppo_discovery.synthetic import make_synthetic_state
from brain_api.storage.ppo_discovery.local import PPODiscoveryHalalNewModelStorage


def _hashed_manifests() -> tuple[dict, dict]:
    hashes = {
        "training_dataset_hash": "train-a",
        "validation_dataset_hash": "val-a",
        "evaluation_dataset_hash": "eval-a",
    }
    return (
        {"complete": True, "cutoffs": [], **hashes},
        {"complete": True, "source": "synthetic", **hashes},
    )


def test_candidate_write_promote_inference_does_not_touch_patchtst(
    tmp_path: Path,
) -> None:
    storage = PPODiscoveryHalalNewModelStorage(base_path=tmp_path)
    config = PPODiscoveryConfig(dropout=0.0, total_timesteps=8)
    policy = PPODiscoveryActorCritic(config)
    evaluation = {
        "test_cagr": 0.20,
        "alpha_hrp_test_cagr": 0.15,
        "test_max_drawdown": 0.10,
        "alpha_hrp_test_max_drawdown": 0.12,
        "paired_vs_alpha_hrp_point": 0.001,
        "test_weekly_net_log": [0.01] * 52,
        "ablations": {
            name: {"status": "ok", "cagr": 0.18} for name in REQUIRED_ABLATIONS
        },
        "failed_seeds": [],
    }
    with (
        patch(
            "brain_api.storage.patchtst.local.PatchTSTHalalNewModelStorage.load_current_artifacts"
        ) as patchtst_load,
        patch(
            "brain_api.storage.sac.local.SACHalalFilteredModelStorage.load_current_artifacts"
        ) as sac_load,
    ):
        version = write_candidate_artifact(
            storage,
            policy,
            config=config,
            evaluation=evaluation,
            universe_manifest={
                "snapshot_sha256": "sha256:abc",
                "sorted_symbols": ["S00"],
            },
            experiment_id="ci",
            end_date="2026-08-31",
            regime_hmm={"p_calm": 0.4, "p_stress": 0.3, "schema_version": 3},
            news_manifest=_hashed_manifests()[0],
            price_manifest=_hashed_manifests()[1],
            pretrained_encoder_state_dict=policy.temporal.state_dict(),
        )
        assert storage.read_current_version() is None
        artifacts = storage.load_artifacts(version)
        check = evaluate_ppo_discovery_promotion(
            metadata=artifacts.metadata,
            evaluation=evaluation,
            approved_by="razin",
            expected_config_hash=artifacts.metadata["config_hash"],
        )
        assert check.is_healthy is True
        updated = reevaluate_ppo_discovery(storage, version)
        reloaded = storage.load_artifacts(version)
        evaluation_on_disk = json.loads(
            (reloaded.artifact_dir / "evaluation.json").read_text()
        )
        assert reloaded.metadata["result_hash"] == result_hash(evaluation_on_disk)
        assert reloaded.metadata["result_hash"] == updated["result_hash"]
        storage.load_artifacts(version)
        storage.promote_version(version)
        assert storage.read_current_version() == version
        state = make_synthetic_state()
        result = run_ppo_discovery_inference(
            state,
            expected_digest=state.state_digest,
            artifacts=storage.load_current_artifacts(),
        )
        assert result.model_type == "ppo_discovery"
        assert abs(sum(result.percentage_weights.values()) - 1.0) < 1e-6
        patchtst_load.assert_not_called()
        sac_load.assert_not_called()
    assert artifacts.metadata["asset_feature_names"] == list(ASSET_FEATURE_NAMES)
    assert artifacts.metadata["global_feature_names"] == list(GLOBAL_FEATURE_NAMES)
    assert artifacts.metadata["news_required"] is True
    assert artifacts.metadata["news_schema_version"] == 1
    assert artifacts.metadata["finbert_revision"] == (
        "4556d13015211d73dccd3fdd39d39232506f3e43"
    )
    assert artifacts.metadata["news_adapter_revision"]
    assert artifacts.metadata["broker_cost_model"] == (PPO_DISCOVERY_BROKER_COST_MODEL)
    assert artifacts.metadata["training_nav_usd"] == (PPO_DISCOVERY_TRAINING_NAV_USD)
    assert evaluation["broker_cost_config"] == config.to_dict()["broker_cost_config"]


def test_tampered_evaluation_fails_promote_until_reevaluate(tmp_path: Path) -> None:
    storage = PPODiscoveryHalalNewModelStorage(base_path=tmp_path)
    config = PPODiscoveryConfig(dropout=0.0, total_timesteps=8)
    policy = PPODiscoveryActorCritic(config)
    evaluation = {
        "test_cagr": 0.20,
        "alpha_hrp_test_cagr": 0.15,
        "test_max_drawdown": 0.10,
        "alpha_hrp_test_max_drawdown": 0.12,
        "paired_vs_alpha_hrp_point": 0.001,
        "test_weekly_net_log": [0.01] * 52,
        "ablations": {
            name: {"status": "ok", "cagr": 0.18} for name in REQUIRED_ABLATIONS
        },
        "failed_seeds": [],
    }
    version = write_candidate_artifact(
        storage,
        policy,
        config=config,
        evaluation=evaluation,
        universe_manifest={"snapshot_sha256": "sha256:abc", "sorted_symbols": ["S00"]},
        experiment_id="ci",
        end_date="2026-08-31",
        regime_hmm={"p_calm": 0.4, "p_stress": 0.3, "schema_version": 3},
        news_manifest=_hashed_manifests()[0],
        price_manifest=_hashed_manifests()[1],
        pretrained_encoder_state_dict=policy.temporal.state_dict(),
    )
    artifacts = storage.load_artifacts(version)
    evaluation_path = artifacts.artifact_dir / "evaluation.json"
    tampered = json.loads(evaluation_path.read_text())
    tampered["test_cagr"] = 0.99
    evaluation_path.write_text(json.dumps(tampered, indent=2, sort_keys=True))
    storage.write_checksums(version)
    tampered_on_disk = json.loads(evaluation_path.read_text())
    check = evaluate_ppo_discovery_promotion(
        metadata=artifacts.metadata,
        evaluation=tampered_on_disk,
        approved_by="razin",
        expected_config_hash=artifacts.metadata["config_hash"],
    )
    assert check.is_healthy is False
    assert any("result_hash" in reason for reason in check.failure_reasons)
    updated = reevaluate_ppo_discovery(storage, version)
    reloaded = storage.load_artifacts(version)
    assert reloaded.metadata["result_hash"] == result_hash(updated)
    synced = evaluate_ppo_discovery_promotion(
        metadata=reloaded.metadata,
        evaluation=updated,
        approved_by="razin",
        expected_config_hash=reloaded.metadata["config_hash"],
    )
    assert synced.is_healthy is True


def test_pretrained_encoder_file_is_stage_a_not_post_ppo(tmp_path: Path) -> None:
    storage = PPODiscoveryHalalNewModelStorage(base_path=tmp_path)
    config = PPODiscoveryConfig(dropout=0.0, total_timesteps=8)
    policy = PPODiscoveryActorCritic(config)
    stage_a = {
        key: tensor.detach().clone()
        for key, tensor in policy.temporal.state_dict().items()
    }
    with torch.no_grad():
        for parameter in policy.temporal.parameters():
            parameter.add_(3.0)
    version = write_candidate_artifact(
        storage,
        policy,
        config=config,
        evaluation={
            "test_cagr": 0.20,
            "alpha_hrp_test_cagr": 0.15,
            "test_max_drawdown": 0.10,
            "alpha_hrp_test_max_drawdown": 0.12,
            "paired_vs_alpha_hrp_point": 0.001,
            "test_weekly_net_log": [0.01] * 52,
            "ablations": {
                name: {"status": "ok", "cagr": 0.18} for name in REQUIRED_ABLATIONS
            },
            "failed_seeds": [],
        },
        universe_manifest={"snapshot_sha256": "sha256:abc", "sorted_symbols": ["S00"]},
        experiment_id="ci",
        end_date="2026-08-31",
        regime_hmm={"p_calm": 0.4, "p_stress": 0.3, "schema_version": 3},
        news_manifest=_hashed_manifests()[0],
        price_manifest=_hashed_manifests()[1],
        pretrained_encoder_state_dict=stage_a,
    )
    path = (
        tmp_path
        / "models"
        / "ppo_discovery_halal_new"
        / version
        / "pretrained_temporal_encoder.pt"
    )
    loaded = torch.load(path, map_location="cpu")
    for key, tensor in stage_a.items():
        torch.testing.assert_close(loaded[key], tensor)
        assert not torch.allclose(loaded[key], policy.temporal.state_dict()[key])


def test_reject_schema_mismatch_requires_news_pins() -> None:
    metadata = {
        "asset_feature_names": list(ASSET_FEATURE_NAMES),
        "global_feature_names": list(GLOBAL_FEATURE_NAMES),
        "news_required": True,
        "experiment_variant": "full",
    }
    with pytest.raises(PPODiscoveryError, match="ppo_discovery_schema_version"):
        reject_schema_mismatch(metadata)
    metadata["ppo_discovery_schema_version"] = 1
    with pytest.raises(PPODiscoveryError, match="architecture"):
        reject_schema_mismatch(metadata)
    metadata["architecture"] = "temporal_set_factored"
    with pytest.raises(PPODiscoveryError, match="news_schema_version"):
        reject_schema_mismatch(metadata)
    metadata["news_schema_version"] = 1
    with pytest.raises(PPODiscoveryError, match="FinBERT"):
        reject_schema_mismatch(metadata)


def test_incomplete_version_directory_is_rebuilt(tmp_path: Path) -> None:
    storage = PPODiscoveryHalalNewModelStorage(base_path=tmp_path)
    config = PPODiscoveryConfig(dropout=0.0, total_timesteps=8)
    policy = PPODiscoveryActorCritic(config)
    news, price = _hashed_manifests()

    def _write() -> str:
        return write_candidate_artifact(
            storage,
            policy,
            config=config,
            evaluation={
                "test_cagr": 0.20,
                "alpha_hrp_test_cagr": 0.15,
                "test_max_drawdown": 0.10,
                "alpha_hrp_test_max_drawdown": 0.12,
                "paired_vs_alpha_hrp_point": 0.001,
                "test_weekly_net_log": [0.01] * 52,
                "ablations": {
                    name: {"status": "ok", "cagr": 0.18} for name in REQUIRED_ABLATIONS
                },
                "failed_seeds": [],
            },
            universe_manifest={
                "snapshot_sha256": "sha256:abc",
                "sorted_symbols": ["S00"],
            },
            experiment_id="ci",
            end_date="2026-08-31",
            regime_hmm={"p_calm": 0.4, "p_stress": 0.3, "schema_version": 3},
            news_manifest=news,
            price_manifest=price,
            pretrained_encoder_state_dict=policy.temporal.state_dict(),
        )

    version = _write()
    checksum = (
        tmp_path / "models" / "ppo_discovery_halal_new" / version / "checksums.sha256"
    )
    checksum.unlink()
    assert not storage.version_exists(version)
    rebuilt = _write()
    assert rebuilt == version
    assert storage.version_exists(version)


def test_candidate_metadata_window_timestamp_and_sharpe(tmp_path: Path) -> None:
    storage = PPODiscoveryHalalNewModelStorage(base_path=tmp_path)
    config = PPODiscoveryConfig(dropout=0.0, total_timesteps=8)
    policy = PPODiscoveryActorCritic(config)
    news, price = _hashed_manifests()
    news["weeks"] = [{"cutoff": "2020-06-01T20:00:00+00:00", "symbols": {}}]
    price["start"] = "2019-01-01"
    version = write_candidate_artifact(
        storage,
        policy,
        config=config,
        evaluation={
            "test_cagr": 0.05,
            "test_sharpe": 1.25,
            "alpha_hrp_test_cagr": 0.15,
            "test_max_drawdown": 0.10,
            "alpha_hrp_test_max_drawdown": 0.12,
            "paired_vs_alpha_hrp_point": 0.001,
            "test_weekly_net_log": [0.01] * 52,
            "ablations": {
                name: {"status": "ok", "cagr": 0.18} for name in REQUIRED_ABLATIONS
            },
            "failed_seeds": [],
        },
        universe_manifest={"snapshot_sha256": "sha256:abc", "sorted_symbols": ["S00"]},
        experiment_id="ci",
        end_date="2026-08-31",
        regime_hmm={"p_calm": 0.4, "p_stress": 0.3, "schema_version": 3},
        news_manifest=news,
        price_manifest=price,
        pretrained_encoder_state_dict=policy.temporal.state_dict(),
    )
    metadata = storage.load_artifacts(version).metadata
    assert metadata["data_window"]["start"] == "2019-01-01"
    assert isinstance(metadata["data_window"]["start"], str)
    assert metadata["trained_at"] == metadata["training_timestamp"]
    assert metadata["metrics"]["test_sharpe"] == 1.25
    assert metadata["prior_version"] is None
    assert any("12%" in reason for reason in metadata["failure_reasons"])


def test_data_window_start_falls_back_to_first_week_cutoff(tmp_path: Path) -> None:
    storage = PPODiscoveryHalalNewModelStorage(base_path=tmp_path)
    config = PPODiscoveryConfig(dropout=0.0, total_timesteps=8)
    policy = PPODiscoveryActorCritic(config)
    news, price = _hashed_manifests()
    news["weeks"] = [{"cutoff": "2020-06-01T20:00:00+00:00", "symbols": {}}]
    version = write_candidate_artifact(
        storage,
        policy,
        config=config,
        evaluation={
            "test_cagr": 0.20,
            "test_sharpe": 0.5,
            "alpha_hrp_test_cagr": 0.15,
            "test_max_drawdown": 0.10,
            "alpha_hrp_test_max_drawdown": 0.12,
            "paired_vs_alpha_hrp_point": 0.001,
            "test_weekly_net_log": [0.01] * 52,
            "ablations": {
                name: {"status": "ok", "cagr": 0.18} for name in REQUIRED_ABLATIONS
            },
            "failed_seeds": [],
        },
        universe_manifest={"snapshot_sha256": "sha256:abc", "sorted_symbols": ["S00"]},
        experiment_id="ci",
        end_date="2026-08-31",
        regime_hmm={"p_calm": 0.4, "p_stress": 0.3, "schema_version": 3},
        news_manifest=news,
        price_manifest=price,
        pretrained_encoder_state_dict=policy.temporal.state_dict(),
    )
    start = storage.load_artifacts(version).metadata["data_window"]["start"]
    assert start == "2020-06-01T20:00:00+00:00"
    assert not isinstance(start, dict)

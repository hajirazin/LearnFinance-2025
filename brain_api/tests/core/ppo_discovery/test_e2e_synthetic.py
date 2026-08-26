"""Synthetic e2e: candidate write, promote gates, inference, no PatchTST I/O."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from brain_api.core.ppo_discovery.artifacts import write_candidate_artifact
from brain_api.core.ppo_discovery.config import (
    ASSET_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    REQUIRED_ABLATIONS,
    PPODiscoveryConfig,
)
from brain_api.core.ppo_discovery.inference import run_ppo_discovery_inference
from brain_api.core.ppo_discovery.policy import PPODiscoveryActorCritic
from brain_api.core.ppo_discovery.promotion import evaluate_ppo_discovery_promotion
from brain_api.core.ppo_discovery.synthetic import make_synthetic_state
from brain_api.storage.ppo_discovery.local import PPODiscoveryHalalNewModelStorage


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
        "ablations": {name: {"status": "ok"} for name in REQUIRED_ABLATIONS},
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
            regime_hmm={"p_calm": 0.4, "p_stress": 0.3},
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

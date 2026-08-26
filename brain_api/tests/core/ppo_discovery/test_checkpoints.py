"""Atomic seed-checkpoint resume for ppo_discovery."""

from __future__ import annotations

from pathlib import Path

from brain_api.core.ppo_discovery.checkpoints import (
    load_seed_checkpoint,
    model_config_hash,
    save_seed_checkpoint,
    seed_checkpoint_dir,
)
from brain_api.core.ppo_discovery.config import PPODiscoveryConfig
from brain_api.core.ppo_discovery.policy import PPODiscoveryActorCritic


def test_seed_checkpoint_round_trip_and_resume_path(tmp_path: Path) -> None:
    config = PPODiscoveryConfig(dropout=0.0, total_timesteps=4, seeds=(42,))
    policy = PPODiscoveryActorCritic(config)
    directory = seed_checkpoint_dir(
        tmp_path,
        experiment_id="e2e",
        snapshot_hash="sha256:deadbeef",
        config_hash=model_config_hash(config),
    )
    save_seed_checkpoint(directory, seed=42, policy=policy, metadata={"val_cagr": 0.1})
    loaded = load_seed_checkpoint(directory, seed=42)
    assert loaded is not None
    assert loaded["metadata"]["seed"] == 42
    restored = PPODiscoveryActorCritic(config)
    restored.load_state_dict(loaded["state_dict"])
    original = {name: tensor.clone() for name, tensor in policy.state_dict().items()}
    for name, tensor in restored.state_dict().items():
        assert tensor.equal(original[name])
    assert load_seed_checkpoint(directory, seed=7) is None
    assert "sha256_deadbeef" in str(directory)


def test_stale_checkpoint_hashes_are_ignored(tmp_path: Path) -> None:
    config = PPODiscoveryConfig(dropout=0.0, total_timesteps=4, seeds=(42,))
    policy = PPODiscoveryActorCritic(config)
    directory = seed_checkpoint_dir(
        tmp_path,
        experiment_id="e2e",
        snapshot_hash="sha256:deadbeef",
        config_hash=model_config_hash(config),
    )
    save_seed_checkpoint(
        directory,
        seed=42,
        policy=policy,
        metadata={
            "protocol_digest": "old",
            "training_dataset_hash": "train-old",
            "snapshot_sha256": "sha256:deadbeef",
            "model_config_hash": model_config_hash(config),
        },
    )
    loaded = load_seed_checkpoint(
        directory,
        seed=42,
        expected={
            "protocol_digest": "new",
            "training_dataset_hash": "train-old",
            "snapshot_sha256": "sha256:deadbeef",
            "model_config_hash": model_config_hash(config),
        },
    )
    assert loaded is None
    loaded = load_seed_checkpoint(
        directory,
        seed=42,
        expected={
            "protocol_digest": "old",
            "training_dataset_hash": "train-old",
            "snapshot_sha256": "sha256:deadbeef",
            "model_config_hash": model_config_hash(config),
        },
    )
    assert loaded is not None

"""Versioned SAC v3 candidate isolation without activation."""

from pathlib import Path

import torch

from brain_api.storage.sac import SACHalalFilteredModelStorage, create_sac_metadata
from tests.test_sac import _v3_auxiliary, create_mock_training_result, mock_config


def test_candidate_artifacts_are_isolated_and_do_not_mutate_current(tmp_path: Path):
    storage = SACHalalFilteredModelStorage(tmp_path)
    config = mock_config()
    result = create_mock_training_result(config)
    symbols = result.symbol_order

    for seed in (42, 123, 2026):
        metadata = create_sac_metadata(
            version="v-test",
            data_window_start="2020-01-01",
            data_window_end="2026-08-07",
            symbols=symbols,
            config=config,
            promoted=False,
            prior_version=None,
            actor_loss=0.1,
            critic_loss=0.2,
            avg_episode_return=0.0,
            avg_episode_sharpe=0.0,
            eval_sharpe=0.0,
            eval_cagr=0.0,
            eval_max_drawdown=0.0,
            training_seed=seed,
        )
        storage.write_candidate_artifacts(
            "v-test",
            seed,
            result.actor,
            result.critic,
            result.critic_target,
            torch.tensor(0.0),
            result.scaler,
            config,
            symbols,
            metadata,
            _v3_auxiliary(result),
        )

    version_dir = tmp_path / "models" / "sac_halal_filtered" / "v-test"
    assert storage.read_current_version() is None
    assert not (version_dir / "actor.pt").exists()
    for seed in (42, 123, 2026):
        candidate_dir = version_dir / "candidates" / f"seed-{seed}"
        assert (candidate_dir / "actor.pt").is_file()
        assert (candidate_dir / "sac_v3_auxiliary.json").is_file()

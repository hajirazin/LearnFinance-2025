from pathlib import Path

import numpy as np
import torch

from brain_api.core.portfolio_rl.sac_networks import GaussianActor, TwinCritic
from brain_api.core.portfolio_rl.scaler import PortfolioScaler
from brain_api.core.sac import DEFAULT_SAC_CONFIG
from brain_api.storage.sac import SACHalalFilteredModelStorage


def _artifacts(n_stocks: int = 2):
    # signals (8) + PatchTST (1) + weights (n_stocks + 1) => 10 * n_stocks + 1
    state_dim = 10 * n_stocks + 1
    action_dim = n_stocks + 1
    actor = GaussianActor(state_dim, action_dim)
    critic = TwinCritic(state_dim, action_dim)
    critic_target = TwinCritic(state_dim, action_dim)
    scaler = PortfolioScaler.create(n_stocks=n_stocks)
    scaler.fit(np.zeros((2, state_dim)))
    return actor, critic, critic_target, scaler


def test_candidate_artifacts_are_isolated_and_only_selected_seed_is_promoted(
    tmp_path: Path,
):
    storage = SACHalalFilteredModelStorage(tmp_path)
    actor, critic, critic_target, scaler = _artifacts()
    symbols = ["AAA", "BBB"]

    for seed in (42, 123, 2026):
        storage.write_candidate_artifacts(
            "v-test",
            seed,
            actor,
            critic,
            critic_target,
            torch.tensor(0.0),
            scaler,
            DEFAULT_SAC_CONFIG,
            symbols,
            {"training_seed": seed},
        )

    promoted = storage.promote_candidate("v-test", 123)

    assert (promoted / "actor.pt").is_file()
    assert storage.read_metadata("v-test")["training_seed"] == 123
    assert (promoted / "candidates" / "seed-42" / "metadata.json").is_file()
    assert (promoted / "candidates" / "seed-2026" / "metadata.json").is_file()

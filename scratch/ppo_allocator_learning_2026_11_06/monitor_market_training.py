"""Read-only learning-curve diagnostics on fixed training contexts, never 2026."""

import json
import pickle
import time

import numpy as np
import torch

from market_rollout import selected_dispersion, state_with_book
from research_support import ROOT, write_json

from brain_api.core.ppo_discovery.config import PPODiscoveryConfig
from brain_api.core.ppo_discovery.policy import PPODiscoveryActorCritic

torch.set_num_threads(2)
with (ROOT / "market_dataset.pkl").open("rb") as handle:
    data = pickle.load(handle)
train = [r for r in data["templates"] if r["split"] == "train"]
contexts = [train[i] for i in np.linspace(0, len(train) - 1, 12).astype(int)]
del data, train
policy = PPODiscoveryActorCritic(PPODiscoveryConfig()).eval()
output = []
last = 0
while not (ROOT / "market_test_summary.json").exists():
    progress_path = ROOT / "market_training_progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        episode = int(progress[-1]["episode_index"])
        if episode > last:
            payload = torch.load(
                ROOT / "market_checkpoints/seed-42.partial.pt",
                weights_only=False,
                map_location="cpu",
            )
            policy.load_state_dict(payload["policy"])
            cvs = []
            cash = []
            counts = []
            for row in contexts:
                weights = policy.infer_weights(
                    state_with_book(row["state"], {"CASH": 1.0})
                )
                diag = selected_dispersion(weights)
                if diag["cv"] is not None:
                    cvs.append(diag["cv"])
                cash.append(weights["CASH"])
                counts.append(len(weights) - 1)
            record = {
                "episode": payload["episode_index"],
                "decisions": payload["steps_done"],
                "training_contexts": 12,
                "mean_deterministic_cv": None if not cvs else float(np.mean(cvs)),
                "mean_deterministic_cash": float(np.mean(cash)),
                "mean_deterministic_k": float(np.mean(counts)),
            }
            output.append(record)
            write_json("market_allocation_learning_curve.json", output)
            print(record, flush=True)
            last = episode
    log = (ROOT / "market_training.log").read_text()
    if "Traceback (most recent call last)" in log:
        raise RuntimeError(
            "Full market experiment failed; monitor will not keep waiting"
        )
    time.sleep(20)
print("Market evaluation complete; training-only monitor stopped", flush=True)

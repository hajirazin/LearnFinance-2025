"""One faithful full PPO retrain, pre-2026 data only, local scratch checkpoints."""

import gc
import json
import pickle
import sys
import time

import numpy as np
import torch

from market_rollout import (
    equalize,
    execute,
    rollout,
    selected_dispersion,
    state_with_book,
)
from research_support import ROOT, REPO, file_hash, write_json

from brain_api.core.ppo_discovery.checkpoints import (
    load_seed_partial_checkpoint,
    save_seed_partial_checkpoint,
)
from brain_api.core.ppo_discovery.config import PPODiscoveryConfig
from brain_api.core.ppo_discovery.evaluator import evaluate_policy_weeks
from brain_api.core.ppo_discovery.policy import PPODiscoveryActorCritic
from brain_api.core.ppo_discovery.pretraining import pretrain_temporal_encoder
from brain_api.core.ppo_discovery.trainer import train_ppo_discovery

sys.path.insert(0, str(REPO / "scratch/ppo_equal_weight_2026_11_06"))
from statistics_helpers import bootstrap


def pretrain_inputs(rows):
    histories = []
    targets = []
    for row in rows:
        state = row["state"]
        slots = [
            i
            for i, flag in enumerate(state.asset_mask)
            if flag and state.symbols[i] in row["returns"]
        ]
        histories.append(state.price_history[slots])
        targets.append(
            np.array([np.log1p(row["returns"][state.symbols[i]]) for i in slots])
        )
    return histories, targets


def evaluate(policy, rows, config):
    books = {name: {"CASH": 1.0} for name in ["ppo", "equal_fixed", "equal_feedback"]}
    weekly = []
    for i, row in enumerate(rows):
        state = state_with_book(row["state"], books["ppo"])
        weights, order = policy.infer_decision(state)
        targets = {"ppo": weights, "equal_fixed": equalize(weights)}
        feedback = state_with_book(row["state"], books["equal_feedback"])
        targets["equal_feedback"] = equalize(policy.infer_weights(feedback))
        outcomes = {}
        for name, target in targets.items():
            books[name], outcomes[name], _ = execute(books[name], target, row, config)
        weekly.append(
            {
                "as_of": state.as_of,
                "decision_open": str(row["start"]),
                "outcome_open": str(row["end"]),
                "k": len(order),
                "cash": weights["CASH"],
                "weights": weights,
                "selection_order": list(order),
                **selected_dispersion(weights),
                **outcomes,
            }
        )
        if (i + 1) % 12 == 0:
            print(f"2026 evaluation {i + 1}/{len(rows)}", flush=True)
    metrics = {
        name: evaluate_policy_weeks([r[name]["net_log"] for r in weekly])
        for name in books
    }
    diffs = [r["ppo"]["net_log"] - r["equal_fixed"]["net_log"] for r in weekly]
    cvs = [r["cv"] for r in weekly if r["cv"] is not None]
    summary = {
        "performance": metrics,
        "mean_stock_weight_cv": None if not cvs else float(np.mean(cvs)),
        "multi_stock_weeks": len(cvs),
        "mean_cash": float(np.mean([r["cash"] for r in weekly])),
        "k_counts": {
            str(k): sum(r["k"] == k for r in weekly)
            for k in sorted({r["k"] for r in weekly})
        },
        "paired_weekly_net_log_difference": {
            str(b): bootstrap(diffs, block=b) for b in [4, 8, 13]
        },
        "cagr_difference_vs_equal": metrics["ppo"]["cagr"]
        - metrics["equal_fixed"]["cagr"],
    }
    return summary, weekly


def main():
    if not torch.backends.mps.is_available():
        raise RuntimeError("The larger full-policy experiment requires Mac GPU access")
    torch.set_num_threads(4)
    device = torch.device("mps")
    recorded_manifest_path = ROOT / "market_training_manifest.json"
    recorded_manifest = (
        json.loads(recorded_manifest_path.read_text())
        if recorded_manifest_path.exists()
        else None
    )
    config = (
        PPODiscoveryConfig.from_dict(recorded_manifest["config"])
        if recorded_manifest is not None
        else PPODiscoveryConfig()
    )
    torch.manual_seed(42)
    np.random.seed(42)
    with (ROOT / "market_dataset.pkl").open("rb") as handle:
        data = pickle.load(handle)
    train = [r for r in data["templates"] if r["split"] == "train"]
    test = [r for r in data["templates"] if r["split"] == "test"]
    assert max(r["end"].date().isoformat() for r in train) <= "2025-12-31"
    assert min(r["start"].date().isoformat() for r in test) >= "2026-01-01"
    manifest = {
        "config": config.to_dict(),
        "seed": 42,
        "data": data["manifest"],
        "dataset_sha256": file_hash(ROOT / "market_dataset.pkl"),
        "budget_note": "10,000 minimum sampled decisions; native trainer finishes its final episode",
        "selection": "fixed final-budget checkpoint, no 2026-driven checkpoint selection",
    }
    if recorded_manifest is not None:
        if recorded_manifest["dataset_sha256"] != manifest["dataset_sha256"]:
            raise RuntimeError(
                "The prepared dataset changed; do not reuse cached model files"
            )
    write_json("market_training_manifest.json", manifest)
    policy = PPODiscoveryActorCritic(config).to(device)
    pretrained = ROOT / "market_pretrained.pt"
    if pretrained.exists():
        policy.load_state_dict(
            torch.load(pretrained, map_location=device, weights_only=True)
        )
        print("Loaded scratch pre-2026 pretrained policy", flush=True)
    else:
        histories, targets = pretrain_inputs(train)
        print(
            f"Starting pre-2026 encoder pretraining: weeks={len(histories)}", flush=True
        )
        metrics = pretrain_temporal_encoder(
            policy, histories, targets, config=config, seed=42, device=device
        )
        torch.save(
            {k: v.detach().cpu() for k, v in policy.state_dict().items()}, pretrained
        )
        write_json("market_pretraining.json", metrics)
        del histories, targets
        gc.collect()
        torch.mps.empty_cache()
    checkpoint = ROOT / "market_checkpoints"
    expected = {
        "dataset_sha256": manifest["dataset_sha256"],
        "config": config.to_dict(),
    }
    resume = load_seed_partial_checkpoint(checkpoint, seed=42, expected=expected)
    started = time.time()
    progress = []

    def on_episode(**payload):
        save_seed_partial_checkpoint(
            checkpoint,
            seed=42,
            policy=payload["policy"],
            optimizer=payload["optimizer"],
            device=device,
            steps_done=payload["steps_done"],
            episode_index=payload["episode_index"],
            update_index=payload["update_index"],
            metadata=expected,
        )
        row = {**payload["metrics"], "elapsed_seconds": time.time() - started}
        progress.append(row)
        write_json("market_training_progress.json", progress)
        print("Market training episode", row, flush=True)

    final_path = ROOT / "market_seed42_final.pt"
    if final_path.exists():
        policy.load_state_dict(
            torch.load(final_path, map_location=device, weights_only=True)
        )
        training_metrics = json.loads(
            (ROOT / "market_training_metrics.json").read_text()
        )
    else:
        training_metrics = train_ppo_discovery(
            policy,
            lambda current, cache: rollout(current, train, config, cache),
            config=config,
            seed=42,
            device=device,
            resume=resume,
            on_episode_complete=on_episode,
        )
        torch.save(
            {k: v.detach().cpu() for k, v in policy.state_dict().items()}, final_path
        )
        write_json("market_training_metrics.json", training_metrics)
    policy.eval()
    summary, weekly = evaluate(policy, test, config)
    write_json("market_test_weekly.json", weekly)
    write_json(
        "market_test_summary.json",
        {**summary, "training": training_metrics, "manifest": manifest},
    )
    print("MARKET TEST", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

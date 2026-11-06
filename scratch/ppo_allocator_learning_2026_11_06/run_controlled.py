"""Multi-seed conditional-PPO learning test; no production writes."""

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from controlled_policy import (
    AllocationContext,
    ConditionalAllocationPolicy,
    N,
    context_returns,
)
from controlled_rewards import feasible_comparator, reward

from brain_api.core.ppo_discovery.config import PPODiscoveryConfig
from brain_api.core.ppo_discovery.rollout import RolloutStep
from brain_api.core.ppo_discovery.trainer import ppo_update

ROOT = Path(__file__).resolve().parent
SEEDS = (42, 123, 2026)
CASES = (
    ("head_signal", "head", True),
    ("set_signal", "set", True),
    ("set_hidden_signal", "set", False),
)


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, allow_nan=False))


def evaluate(policy, visible, seed, comparator):
    rng = np.random.default_rng(seed + 100000)
    contexts = [AllocationContext(int(x), visible) for x in rng.integers(0, N, 1000)]
    returns = context_returns(contexts, rng, noise=0)
    q, alpha = policy.deterministic_batch(contexts)
    actions, _, _ = policy.sample_batch(contexts)
    mean_rewards, sampled_rewards, equal_rewards, winner_weights, cvs = (
        [],
        [],
        [],
        [],
        [],
    )
    for i, context in enumerate(contexts):
        mean_rewards.append(reward(q[i], returns[i])["reward"])
        sampled_rewards.append(
            reward(actions[i].dirichlet_weights, returns[i])["reward"]
        )
        equal_rewards.append(reward(np.full(N, 1 / N), returns[i])["reward"])
        winner_weights.append(float(q[i, context.signal_slot]))
        cvs.append(float(q[i].std() / q[i].mean()))
    differences = np.array(mean_rewards) - equal_rewards
    mean = float(differences.mean())
    se = float(differences.std(ddof=1) / np.sqrt(len(differences)))
    feasible_gain = comparator["best"]["reward"] - comparator["equal"]["reward"]
    ci = [mean - 1.96 * se, mean + 1.96 * se]
    return {
        "contexts": len(contexts),
        "mean_winner_equity_weight": float(np.mean(winner_weights)),
        "mean_cv": float(np.mean(cvs)),
        "mean_reward": float(np.mean(mean_rewards)),
        "mean_sampled_reward": float(np.mean(sampled_rewards)),
        "mean_equal_reward": float(np.mean(equal_rewards)),
        "mean_paired_reward_advantage": mean,
        "paired_reward_ci95": ci,
        "fraction_feasible_gain": mean / feasible_gain,
        "mean_total_concentration": float(alpha.sum(axis=1).mean()),
        "passes": bool(np.mean(winner_weights) >= 0.30 and ci[0] > 0),
        "confidence_note": "Conditional iid synthetic contexts; across-seed replication is the main evidence.",
    }


def positive_control(config, device, comparator):
    torch.manual_seed(42)
    policy = ConditionalAllocationPolicy(config, "head").to(device).eval()
    contexts = [AllocationContext(i % N) for i in range(100)]
    target = torch.full(
        (len(contexts), N),
        (1 - comparator["best"]["winner_weight"]) / (N - 1),
        device=device,
    )
    for i, c in enumerate(contexts):
        target[i, c.signal_slot] = comparator["best"]["winner_weight"]
    optimizer = torch.optim.Adam(policy.allocation_head.parameters(), lr=0.01)
    for _ in range(500):
        alpha, _ = policy.context_encoding(contexts)
        q = alpha / alpha.sum(dim=-1, keepdim=True)
        loss = (q - target).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return evaluate(policy, True, 42, comparator)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.set_num_threads(4)
    config = replace(
        PPODiscoveryConfig(),
        minibatch_size=64,
        ppo_microbatch_size=64,
        freeze_encoder_updates=10**9,
        total_timesteps=10000,
    )
    comparator = feasible_comparator()
    write("controlled_comparator.json", comparator)
    positive = positive_control(config, device, comparator)
    write("positive_control.json", positive)
    print(f"Positive capacity control {positive}", flush=True)
    results = []
    for case, representation, visible in CASES:
        for seed in SEEDS:
            dest = ROOT / f"controlled_{case}_seed{seed}.json"
            if dest.exists():
                results.append(json.loads(dest.read_text()))
                continue
            torch.manual_seed(seed)
            np.random.seed(seed)
            rng = np.random.default_rng(seed)
            policy = (
                ConditionalAllocationPolicy(config, representation).to(device).eval()
            )
            optimizer = torch.optim.AdamW(
                [p for p in policy.parameters() if p.requires_grad],
                lr=config.actor_lr,
                weight_decay=config.weight_decay,
            )
            initial = evaluate(policy, visible, seed, comparator)
            started = time.time()
            history = []
            for update in range(int(np.ceil(config.total_timesteps / 128))):
                size = min(128, config.total_timesteps - update * 128)
                contexts = [
                    AllocationContext(int(x), visible) for x in rng.integers(0, N, size)
                ]
                returns = context_returns(contexts, rng)
                actions, values, logs = policy.sample_batch(contexts)
                steps = []
                for i, context in enumerate(contexts):
                    economic = reward(actions[i].dirichlet_weights, returns[i])
                    steps.append(
                        RolloutStep(
                            state=context,
                            action=actions[i],
                            reward=economic["reward"],
                            value=float(values[i]),
                            log_p=float(logs[i]),
                            done=True,
                            realized_net_return=economic["net_log"],
                        )
                    )
                ppo_update(policy, steps, optimizer, config=config, update_index=update)
                gradient_norm = float(
                    torch.norm(policy.allocation_head.weight.grad).item()
                )
                if (update + 1) % 8 == 0 or update == 78:
                    q, _ = policy.deterministic_batch(
                        [AllocationContext(i, visible) for i in range(N)]
                    )
                    record = {
                        "decisions": min((update + 1) * 128, config.total_timesteps),
                        "mean_winner_weight": float(np.mean(np.diag(q))),
                        "allocation_gradient_norm": gradient_norm,
                        "seconds": time.time() - started,
                    }
                    history.append(record)
                    print(case, seed, record, flush=True)
                    write(f"progress_{case}_seed{seed}.json", history)
            final = evaluate(policy, visible, seed, comparator)
            result = {
                "case": case,
                "representation": representation,
                "signal_visible": visible,
                "seed": seed,
                "decisions": config.total_timesteps,
                "config": config.to_dict(),
                "device": str(device),
                "initial": initial,
                "final": final,
                "history": history,
                "elapsed_seconds": time.time() - started,
            }
            write(dest.name, result)
            torch.save(
                {k: v.detach().cpu() for k, v in policy.state_dict().items()},
                ROOT / f"controlled_{case}_seed{seed}.pt",
            )
            results.append(result)
            write(
                "controlled_summary.json",
                {
                    "positive_control": positive,
                    "feasible_comparator": comparator,
                    "runs": results,
                },
            )
    print("Controlled matrix complete", flush=True)


if __name__ == "__main__":
    main()

"""Research runner for extra PPO seeds, ablations, matched-K, and random baseline.

Never writes production ``current``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRATCH_SEEDS = (42, 123, 2026)


def _results_dir() -> Path:
    path = ROOT / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(name: str, payload: dict[str, Any]) -> Path:
    dest = _results_dir() / name
    dest.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return dest


def run_extra_seeds(config_seeds: tuple[int, ...] = SCRATCH_SEEDS) -> dict[str, Any]:
    """Record the research seed recipe. Training reuses production seed_training."""
    from brain_api.core.ppo_discovery.config import PPODiscoveryConfig

    config = PPODiscoveryConfig(seeds=config_seeds)
    return {
        "kind": "extra_seeds",
        "seeds": list(config.seeds),
        "note": (
            "Call brain_api.core.ppo_discovery.seed_training.train_ppo_discovery_seeds "
            "with this config from a notebook or a longer historical runner. "
            "This CLI does not launch a full retrain."
        ),
    }


def run_against_candidate(candidate_dir: Path) -> dict[str, Any]:
    """Score research ablations against an already-written production candidate."""
    from ablations import run_required_ablations
    from baselines import locked_random_test_metrics
    from diagnostics import (
        build_allocation_head_diagnostics,
        build_transaction_cost_training_diagnostics,
    )
    from matched_k import matched_k_average_rank

    from brain_api.core.ppo_discovery.policy import PPODiscoveryActorCritic
    from brain_api.storage.ppo_discovery.local import PPODiscoveryHalalNewModelStorage

    storage = PPODiscoveryHalalNewModelStorage(base_path=candidate_dir.parent.parent)
    version = candidate_dir.name
    artifacts = storage.load_artifacts(version)
    policy = PPODiscoveryActorCritic(artifacts.config)
    policy.load_state_dict(artifacts.policy_state_dict)
    evaluation = json.loads((candidate_dir / "evaluation.json").read_text())
    return {
        "kind": "candidate_research",
        "version": version,
        "selected_seed": evaluation.get("selected_seed"),
        "test_cagr": evaluation.get("test_cagr"),
        "test_sharpe": evaluation.get("test_sharpe"),
        "note": (
            "Ablation retrains, matched-K, and locked-random need the original "
            "train/test weeks. Re-run historical feature construction in a "
            "notebook, then call run_required_ablations / matched_k_average_rank / "
            "locked_random_test_metrics. Comparison builders are imported here "
            "so they stay out of production."
        ),
        "helpers": {
            "run_required_ablations": run_required_ablations.__name__,
            "matched_k_average_rank": matched_k_average_rank.__name__,
            "locked_random_test_metrics": locked_random_test_metrics.__name__,
            "build_allocation_head_diagnostics": (
                build_allocation_head_diagnostics.__name__
            ),
            "build_transaction_cost_training_diagnostics": (
                build_transaction_cost_training_diagnostics.__name__
            ),
        },
        "policy_loaded": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PPO discovery research runner. Does not write current."
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=None,
        help="Existing production candidate directory (version folder).",
    )
    args = parser.parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "written_at": stamp,
        "promotes_current": False,
        "extra_seeds": run_extra_seeds(),
    }
    if args.candidate_dir is not None:
        payload["candidate"] = run_against_candidate(args.candidate_dir.resolve())
    path = _write_json(f"research_{stamp}.json", payload)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()

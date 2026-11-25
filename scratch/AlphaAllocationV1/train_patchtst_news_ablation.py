"""Disable news information in the frozen seven-slot PatchTST experiment."""

import argparse
import importlib.metadata
import json
from pathlib import Path

import torch
from compare_patchtst import sha256
from train_v1_xgboost_comparison import (
    OUTPUT as SHARED,
)
from train_v1_xgboost_comparison import (
    SEEDS,
    completed,
    fit_patchtst,
    make_patchtst,
    write_json,
)

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data/patchtst_news_ablation"


def preflight():
    dataset_path = SHARED / "dataset.json"
    dataset = json.loads(dataset_path.read_text())
    for filename, digest in dataset["arrays"].items():
        if sha256(SHARED / filename) != digest:
            raise ValueError(f"Shared dataset changed: {filename}")
    for package, version in dataset["versions"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(
                f"Package version differs from the seven-feature fit: {package}"
            )
    current = json.loads(json.dumps(make_patchtst().config.to_dict()))
    references = {}
    for seed in SEEDS:
        name = f"patchtst_{seed}"
        if not completed(name):
            raise ValueError(f"Missing verified seven-feature reference: {name}")
        path = SHARED / f"{name}.json"
        metadata = json.loads(path.read_text())
        if metadata["config"] != current:
            raise ValueError(f"PatchTST configuration differs from {name}")
        if metadata["target_scale"] != dataset["target_std"]:
            raise ValueError(f"Target scale differs from {name}")
        references[name] = {
            "metadata_sha256": sha256(path),
            "weights_sha256": metadata["artifacts"][f"{name}.pt"],
        }
    manifest = {
        "dataset_sha256": sha256(dataset_path),
        "seven_feature_references": references,
        "disabled_feature": "news_sentiment",
        "informative_features": [
            feature for feature in dataset["features"] if feature != "news_sentiment"
        ],
        "network_input_slots": 7,
        "masking": "Replace normalized sentiment history with zero in every copied batch",
        "preserved": "Original rows, eligibility, other values, target, optimizer, validation and selector",
        "seeds": list(SEEDS),
        "config": current,
        "source_sha256": {
            path.name: sha256(path)
            for path in (Path(__file__), ROOT / "train_v1_xgboost_comparison.py")
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("The frozen ablation specification changed")
    write_json(path, manifest)
    return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    torch.set_num_threads(4)
    dataset = preflight()
    print(
        "Verified identical architecture, source arrays, targets, and reference artifacts.",
        flush=True,
    )
    for seed in SEEDS if args.seed is None else (args.seed,):
        print(
            f"Training six-informative-feature counterpart for seed {seed}", flush=True
        )
        fit_patchtst(
            seed,
            dataset["target_std"],
            torch.device(args.device),
            output=OUTPUT,
            disable_news=True,
        )


if __name__ == "__main__":
    main()

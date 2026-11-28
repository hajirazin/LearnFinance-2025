"""Run the authorized, local-only E2 experiment and record both outcomes."""

import argparse
import fcntl
import importlib.metadata
import logging
import os
import subprocess
import sys
import traceback
from dataclasses import replace
from datetime import UTC, datetime

import torch
from dataset import prepare, split_panel
from evidence import DATA, ROOT, code_hashes, read_json, sha256, write_json
from reporting import evaluate
from training import fit_arm

from brain_api.core.patchtst.config import US_WEEKLY_CONFIG


def run(device, *, prepare_only=False):
    config = replace(US_WEEKLY_CONFIG, early_stopping_patience=15)
    panel, manifest = prepare(config)
    if prepare_only:
        return
    development, evaluation = split_panel(panel)
    identity = {
        "code": code_hashes(),
        "panel": sha256(DATA / "panel/manifest.json"),
        "config": config.to_dict(),
        "device": str(device),
        "seed": 42,
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("torch", "numpy", "pandas", "transformers", "duckdb")
        },
    }
    attempts = DATA / "attempts"
    attempts.mkdir(exist_ok=True)
    compatible = []
    for path in sorted(attempts.iterdir()):
        if (path / "run.json").exists() and read_json(path / "run.json")[
            "identity"
        ] == identity:
            compatible.append(path)
    if compatible and all(
        (compatible[-1] / arm / "result.json").exists() for arm in ("mse", "huber")
    ):
        attempt = compatible[-1]
    else:
        attempt = attempts / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        attempt.mkdir()
        write_json(
            attempt / "run.json",
            {
                "identity": identity,
                "pid": os.getpid(),
                "command": sys.argv,
                "revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "working_tree": subprocess.check_output(
                    ["git", "status", "--short"], text=True
                ).splitlines(),
            },
        )
    write_json(
        DATA / "active.json",
        {"attempt": str(attempt.relative_to(ROOT)), "status": "running"},
    )
    try:
        for arm in ("mse", "huber"):
            destination = attempt / arm
            if not (destination / "result.json").exists():
                fit_arm(development, config, arm, device, destination)
        # Initial weights must match exactly before any comparison is accepted.
        initial = [
            torch.load(attempt / a / "initial.pt", weights_only=True)
            for a in ("mse", "huber")
        ]
        for key in initial[0]:
            torch.testing.assert_close(initial[0][key], initial[1][key], rtol=0, atol=0)
        result = evaluate(evaluation, config, device, attempt, manifest)
        write_json(
            DATA / "active.json",
            {
                "attempt": str(attempt.relative_to(ROOT)),
                "status": "complete",
                "decision": result["decision"],
            },
        )
        print(result, flush=True)
    except BaseException:
        (attempt / "failure.txt").write_text(traceback.format_exc())
        write_json(
            DATA / "active.json",
            {"attempt": str(attempt.relative_to(ROOT)), "status": "failed"},
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is unavailable; run with Mac GPU access or explicitly choose CPU"
        )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / "experiment.lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(device, prepare_only=args.prepare_only)


if __name__ == "__main__":
    main()

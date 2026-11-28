"""Finish the prepared experiment in one uninterrupted background process."""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SHARED = REPO / "scratch/sac_halal_rewards_2026_11_28"


def write_status(root, phase, **details):
    status = {
        "run_id": "scratch:sac-halal_filtered-rewards:SPY:2026-11-28",
        "attempt": 1,
        "as_of": datetime.now(UTC).isoformat(),
        "controller_pid": os.getpid(),
        "phase": phase,
        "completed_training_runs": len(
            list((root / "results_50000").glob("*/seed_*/metrics.json"))
        ),
        **details,
    }
    temporary = root / "status.json.tmp"
    temporary.write_text(json.dumps(status, indent=2))
    temporary.replace(root / "status.json")


def existing_training_alive(pid):
    try:
        command = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "command="], text=True, timeout=5
        )
    except subprocess.CalledProcessError as error:
        if error.returncode == 1:
            return False
        raise
    return (
        "run_experiment.py" in command
        and "sac_halal_filtered_spy_rewards_2026_11_28/scenario.json" in command
    )


def run_stage(stage, root):
    command = [
        sys.executable,
        "-u",
        str(SHARED / f"{stage}.py"),
        "--scenario",
        str(root / "scenario.json"),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO / "brain_api")
    if stage == "run_experiment":
        # The existing trainer already writes here; preserve its earlier output.
        with (root / "run.log").open("a") as stream:
            subprocess.run(
                command,
                cwd=REPO,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=8 * 60 * 60,
            )
    else:
        subprocess.run(command, cwd=REPO, env=environment, check=True, timeout=60 * 60)


def run_pipeline(root=ROOT, existing_pid=None, stage_runner=run_stage):
    try:
        manifest = json.loads((root / "evidence/manifest.json").read_text())
        for filename, field in (
            ("panel.pkl", "panel_sha256"),
            ("adjusted_ohlcv.parquet", "price_sha256"),
        ):
            actual = hashlib.sha256(
                (root / "evidence" / filename).read_bytes()
            ).hexdigest()
            if actual != manifest[field]:
                raise ValueError(f"Frozen input changed: {filename}")
        if existing_pid is not None:
            print(
                f"Waiting for active training PID {existing_pid}; training continues uninterrupted.",
                flush=True,
            )
            deadline = time.monotonic() + 8 * 60 * 60
            while existing_training_alive(existing_pid):
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Existing experiment did not finish within eight hours"
                    )
                write_status(
                    root, "waiting_for_existing_training", training_pid=existing_pid
                )
                time.sleep(15)
        for stage in (
            "run_experiment",
            "execution_sensitivity",
            "summarize",
            "verify_results",
        ):
            print(f"START {stage}", flush=True)
            write_status(root, stage)
            stage_runner(stage, root)
            print(f"COMPLETE {stage}", flush=True)
        report = root / "results_50000/report.md"
        verification = json.loads(
            (root / "results_50000/verification.json").read_text()
        )
        if (
            not report.is_file()
            or verification["matched_training_runs"] != 12
            or verification["execution_sensitivity_evaluations"] != 48
        ):
            raise RuntimeError("Final report or experiment verification is incomplete")
        write_status(root, "complete", report=str(report))
        print(f"ALL DONE: {report}", flush=True)
    except Exception as error:
        write_status(root, "failed", error=f"{type(error).__name__}: {error}")
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing-pid", type=int)
    args = parser.parse_args()
    with (ROOT / "pipeline.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("An experiment controller is already running") from None
        run_pipeline(existing_pid=args.existing_pid)


if __name__ == "__main__":
    main()

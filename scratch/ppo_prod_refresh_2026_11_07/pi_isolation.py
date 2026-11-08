"""Temporarily isolate Pi execution, preserving original schedule states."""

import argparse
import asyncio
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from temporalio.client import Client

ROOT = Path(__file__).resolve().parent
ADDRESS = "razinpi.local:7233"
WORKER = "learnfinance-temporal-worker"


def docker(*args):
    return subprocess.check_output(
        ["docker", "--context", "remote-pi", *args], text=True
    ).strip()


async def main(action):
    client = await Client.connect(ADDRESS)
    path = ROOT / "pi_isolation_before.json"
    if action == "isolate":
        if path.exists():
            raise RuntimeError(
                "Isolation snapshot already exists; use its state rather than overwrite"
            )
        running = [
            {"id": w.id, "type": w.workflow_type}
            async for w in client.list_workflows(query='ExecutionStatus = "Running"')
        ]
        states = []
        async for entry in await client.list_schedules():
            description = await client.get_schedule_handle(entry.id).describe()
            state = description.schedule.state
            states.append({"id": entry.id, "paused": state.paused, "note": state.note})
        snapshot = {
            "as_of": datetime.now(UTC).isoformat(),
            "address": ADDRESS,
            "schedules": states,
            "running_workflows": running,
            "worker_running": docker(
                "inspect", WORKER, "--format", "{{.State.Running}}"
            )
            == "true",
        }
        path.write_text(json.dumps(snapshot, indent=2))
        if running:
            raise RuntimeError(
                f"Existing workflows are running; isolation will not disrupt them: {running}"
            )
        for state in states:
            if not state["paused"]:
                await client.get_schedule_handle(state["id"]).pause(
                    note="Temporary PPO-only validation, 2026-11-07"
                )
        docker("stop", "--time", "30", WORKER)
        after = [
            {"id": w.id, "type": w.workflow_type}
            async for w in client.list_workflows(query='ExecutionStatus = "Running"')
        ]
        if after:
            raise RuntimeError(
                f"Unexpected workflow appeared during isolation: {after}"
            )
        print(
            json.dumps(
                {
                    "schedules": len(states),
                    "temporarily_paused": sum(not s["paused"] for s in states),
                    "shared_worker_running": docker(
                        "inspect", WORKER, "--format", "{{.State.Running}}"
                    ),
                    "running_workflows": after,
                }
            ),
            flush=True,
        )
    elif action == "restore":
        before = json.loads(path.read_text())
        for state in before["schedules"]:
            if not state["paused"]:
                await client.get_schedule_handle(state["id"]).unpause(
                    note=state["note"] or ""
                )
        if before["worker_running"]:
            docker("start", WORKER)
        restored = []
        for state in before["schedules"]:
            current = (
                await client.get_schedule_handle(state["id"]).describe()
            ).schedule.state
            assert current.paused == state["paused"]
            restored.append({"id": state["id"], "paused": current.paused})
        (ROOT / "pi_isolation_restored.json").write_text(
            json.dumps(
                {"as_of": datetime.now(UTC).isoformat(), "schedules": restored},
                indent=2,
            )
        )
        print("Restored original schedules and shared worker", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["isolate", "restore"])
    asyncio.run(main(parser.parse_args().action))

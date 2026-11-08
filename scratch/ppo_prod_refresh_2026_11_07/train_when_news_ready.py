"""Run the normal local PPO training endpoint after mandatory news completes."""

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
API = "http://127.0.0.1:8000"


def request(path, payload=None):
    body = json.dumps(payload).encode() if payload is not None else None
    req = Request(
        API + path,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=90) as response:
        return json.load(response)


def save(name, payload):
    (ROOT / name).write_text(json.dumps(payload, indent=2))


def main():
    preflight = json.loads((ROOT / "preflight.json").read_text())
    assert preflight["ready"] is True
    gap_job = json.loads((ROOT / "news_gap_job.json").read_text())["result"]["job_id"]
    deadline = time.monotonic() + 3 * 3600
    while True:
        status = request("/etl/news/gaps/" + gap_job)
        save("news_gap_status.json", status)
        print(
            datetime.now(UTC).isoformat(),
            "news",
            status["status"],
            status["windows_done"],
            "/",
            status["windows_total"],
            flush=True,
        )
        if status["status"] == "complete":
            break
        if status["status"] in {"failed", "cancelled"}:
            raise RuntimeError(f"Mandatory news failed: {status.get('error')}")
        if time.monotonic() > deadline:
            raise TimeoutError("Mandatory news did not complete within three hours")
        time.sleep(30)

    payload = {
        "universe": "halal_new",
        "end_date": "2026-11-07",
        "experiment_id": "ppo-production-refresh-2026-11-07",
        "snapshot_sha256": preflight["snapshot_sha256"],
    }
    existing = ROOT / "training_job.json"
    if existing.exists():
        started = json.loads(existing.read_text())["response"]
    else:
        started = request("/train/ppo-discovery/full", payload)
        save("training_job.json", {"request": payload, "response": started})
    deadline = time.monotonic() + 12 * 3600
    while True:
        status = request("/train/status/" + quote(started["job_id"], safe=""))
        save("training_status.json", status)
        print(
            datetime.now(UTC).isoformat(),
            "training",
            status["status"],
            json.dumps(status.get("progress")),
            flush=True,
        )
        if status["status"] in {"completed", "failed", "cancelled"}:
            save("training_final.json", status)
            print("FINAL", json.dumps(status), flush=True)
            return
        if time.monotonic() > deadline:
            raise TimeoutError("Training monitor exceeded twelve hours")
        time.sleep(30)


if __name__ == "__main__":
    main()

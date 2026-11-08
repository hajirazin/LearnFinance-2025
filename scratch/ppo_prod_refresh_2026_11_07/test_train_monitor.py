"""Verify completed mandatory news releases exactly one production train."""

import importlib.util
import json
from pathlib import Path


def test_complete_news_starts_training_without_another_wait(tmp_path, monkeypatch):
    source = Path(__file__).with_name("train_when_news_ready.py")
    spec = importlib.util.spec_from_file_location("train_monitor", source)
    monitor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(monitor)
    monkeypatch.setattr(monitor, "ROOT", tmp_path)
    (tmp_path / "preflight.json").write_text(
        json.dumps({"ready": True, "snapshot_sha256": "frozen-roster"})
    )
    (tmp_path / "news_gap_job.json").write_text(
        json.dumps({"result": {"job_id": "news-job"}})
    )
    calls = []

    def request(path, payload=None):
        calls.append((path, payload))
        if path == "/etl/news/gaps/news-job":
            return {"status": "complete", "windows_done": 10, "windows_total": 10}
        if path == "/train/ppo-discovery/full":
            return {"job_id": "ppo-job", "status": "pending"}
        if path == "/train/status/ppo-job":
            return {"status": "completed", "result": {"version": "candidate"}}
        raise AssertionError(path)

    def unexpected_wait(_seconds):
        raise AssertionError("Complete news must release training immediately")

    monkeypatch.setattr(monitor, "request", request)
    monkeypatch.setattr(monitor.time, "sleep", unexpected_wait)
    monitor.main()
    assert [path for path, _ in calls] == [
        "/etl/news/gaps/news-job",
        "/train/ppo-discovery/full",
        "/train/status/ppo-job",
    ]
    assert calls[1][1]["snapshot_sha256"] == "frozen-roster"
    assert (
        json.loads((tmp_path / "training_final.json").read_text())["status"]
        == "completed"
    )

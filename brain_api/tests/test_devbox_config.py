"""Devbox scripts for brain_api development vs long training."""

from __future__ import annotations

import json
from pathlib import Path


def test_brain_run_training_is_stable_and_brain_run_keeps_reload() -> None:
    path = Path(__file__).resolve().parents[2] / "devbox.json"
    scripts = json.loads(path.read_text())["shell"]["scripts"]
    brain_run = " ".join(scripts["brain:run"])
    training = " ".join(scripts["brain:run:training"])
    assert "--reload" in brain_run
    assert "brain:run:training" in scripts
    assert "uvicorn brain_api.main:app" in training
    assert "--host 0.0.0.0" in training
    assert "--port 8000" in training
    assert "--reload" not in training
    assert "--workers" not in training

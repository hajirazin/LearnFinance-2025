"""Local evidence identities and atomic experiment records."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
DATA = ROOT / "data"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def code_hashes():
    sources = [
        *ROOT.glob("*.py"),
        ROOT / "PROTOCOL.md",
        REPO / "brain_api/brain_api/core/patchtst/config.py",
        REPO / "brain_api/brain_api/core/patchtst/weekly_data.py",
        REPO / "brain_api/brain_api/core/patchtst/weekly_training.py",
        REPO / "brain_api/brain_api/core/patchtst/weekly_rank_ic.py",
        REPO / "brain_api/brain_api/core/patchtst/calendar.py",
        REPO / "brain_api/brain_api/news/patchtst.py",
    ]
    return {str(p.relative_to(REPO)): sha256(p) for p in sorted(sources)}

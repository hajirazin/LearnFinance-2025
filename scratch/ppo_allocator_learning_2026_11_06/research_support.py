"""Scratch paths, read-only news access and auditable output helpers."""

import hashlib
import json
from pathlib import Path

import duckdb

from brain_api.news.store import NewsStore, news_db_path, news_lock_path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
DATA = REPO / "brain_api/data"


def write_json(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, allow_nan=False, default=str))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_hashes():
    paths = list((REPO / "brain_api/brain_api").rglob("*.py"))
    paths += [
        p for p in (DATA / "models/ppo_discovery_halal_new").rglob("*") if p.is_file()
    ]
    paths += [DATA / "news/news_v1.duckdb"]
    paths += [p for p in (REPO / "temporal").rglob("*.py") if ".venv" not in p.parts]
    return {str(p.relative_to(REPO)): file_hash(p) for p in sorted(paths)}


class ReadOnlyNewsStore(NewsStore):
    def __init__(self):
        self.base_path = DATA
        self.db_path = news_db_path(DATA)
        self.lock_path = news_lock_path(DATA)

    def _connect(self):
        return duckdb.connect(str(self.db_path), read_only=True)

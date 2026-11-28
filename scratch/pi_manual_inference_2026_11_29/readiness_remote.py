"""Read Pi news restore completeness and pinned FinBERT cache readiness."""

import json
import os
import shutil
from pathlib import Path
from urllib.request import Request, urlopen

import duckdb
from huggingface_hub import try_to_load_from_cache
from huggingface_hub.constants import HF_HUB_CACHE

from brain_api.news.models import NEWS_SENTIMENT_MODEL, NEWS_SENTIMENT_REVISION

root = Path("/app/data")
repo = os.environ.get("HF_NEWS_SENTIMENT_REPO", "").strip()
token = os.environ.get("HF_TOKEN")
out = {
    "hf_token_present": bool(token),
    "news_repo": repo,
    "hf_cache_dir": HF_HUB_CACHE,
    "free_disk_bytes": shutil.disk_usage(root).free,
}
request = Request(
    "https://huggingface.co/datasets/" + repo + "/resolve/main/news_v1/manifest.json",
    headers={"Authorization": "Bearer " + token} if token else {},
)
with urlopen(request, timeout=30) as response:
    manifest = json.load(response)
parts = manifest["partitions"]
with duckdb.connect(str(root / "news/news_v1.duckdb"), read_only=True) as con:
    imported = dict(
        con.execute(
            "SELECT partition, checksum FROM news_sync_imports WHERE repo=?", [repo]
        ).fetchall()
    )
    out["news_partitions"] = {
        "published": len(parts),
        "imported": len(imported),
        "missing_or_changed": [
            name for name, meta in parts.items() if imported.get(name) != meta["sha256"]
        ],
    }
    out["latest_coverage_end"] = str(
        con.execute("SELECT max(window_end_inclusive) FROM news_coverage").fetchone()[0]
    )
out["finbert_files"] = {}
for filename in (
    "config.json",
    "pytorch_model.bin",
    "model.safetensors",
    "tokenizer_config.json",
    "vocab.txt",
    "tokenizer.json",
):
    cached = try_to_load_from_cache(
        NEWS_SENTIMENT_MODEL, filename, revision=NEWS_SENTIMENT_REVISION
    )
    out["finbert_files"][filename] = isinstance(cached, str) and Path(cached).is_file()
print(json.dumps(out, indent=2))

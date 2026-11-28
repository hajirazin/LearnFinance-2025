"""Restore news on the Pi with an ephemeral credential, without service changes."""

import json
import subprocess
from pathlib import Path

from huggingface_hub import get_token

token = get_token()
if not token:
    raise RuntimeError("No existing local HF login")

remote_code = r"""
import json, logging, os, shutil, sys
payload = json.load(sys.stdin)
os.environ["HF_TOKEN"] = payload.pop("token")
from brain_api.news.store import NewsStore
from brain_api.news.sync import sync_news
from brain_api.storage.base import DEFAULT_DATA_PATH
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
free = shutil.disk_usage(DEFAULT_DATA_PATH).free
print(json.dumps({"phase":"preflight", "free_disk_bytes": free}), flush=True)
if free < 3 * 1024**3:
    raise RuntimeError("Less than 3 GiB available; restore not started")
commit = sync_news(NewsStore(DEFAULT_DATA_PATH), publish=False, seed=False)
if not commit:
    raise RuntimeError("Restore produced no shared news commit")
print(json.dumps({"phase":"completed", "news_commit":commit}), flush=True)
"""

root = Path(__file__).resolve().parent
with (root / "news_restore_once.log").open("w") as log:
    process = subprocess.Popen(
        [
            "docker",
            "--context",
            "remote-pi",
            "exec",
            "-i",
            "learnfinance-brain-api",
            "/app/.venv/bin/python",
            "-u",
            "-c",
            remote_code,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    process.stdin.write(json.dumps({"token": token}))
    process.stdin.close()
    del token
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    returncode = process.wait()
    if returncode:
        raise SystemExit(returncode)

"""Follow restore progress, then run the authorized inference smoke checks."""

import json
import selectors
import subprocess
import time

from checks import run
from client import ROOT, call

process = subprocess.Popen(
    [
        "docker",
        "--context",
        "remote-pi",
        "logs",
        "--follow",
        "--since",
        "2m",
        "learnfinance-brain-api",
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    bufsize=0,
)
selector = selectors.DefaultSelector()
selector.register(process.stdout, selectors.EVENT_READ)
buffer = b""
terminal = False
deadline = time.monotonic() + 1200
try:
    with (ROOT / "news_restore_api.log").open("w") as log:
        while time.monotonic() < deadline and not terminal:
            if not selector.select(timeout=10):
                if process.poll() is not None:
                    break
                continue
            chunk = process.stdout.read(8192)
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                raw, buffer = buffer.split(b"\n", 1)
                line = raw.decode(errors="replace")
                if (
                    "news sync" in line
                    or "news operation" in line
                    or "News operation" in line
                ):
                    print(line, flush=True)
                    log.write(line + "\n")
                    log.flush()
                if (
                    "news operation completed job_id=news_sync:restore" in line
                    or "News operation news_sync:restore failed" in line
                ):
                    terminal = True
finally:
    process.terminate()  # Stop this read-only log follower, not the container.
    process.wait(timeout=15)
    selector.close()

status = call(
    "/etl/news/sync/news_sync:restore",
    name="news_sync_final",
    timeout=30,
)
print(json.dumps(status), flush=True)
if status["status"] != "completed" or not (status.get("result") or {}).get(
    "news_commit"
):
    raise RuntimeError("News restore did not complete with an HF commit")

run()

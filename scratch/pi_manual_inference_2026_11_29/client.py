"""Bounded manual Pi smoke checks; no execution or training endpoints."""

import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
BASE = "http://razinpi.local:8000"
ALLOWED_POST = {
    "/etl/news/sync",
    "/calendar/monday-decision-window",
    "/news/windows/query",
    "/inference/patchtst",
    "/inference/patchtst/india",
    "/inference/patchtst/score-batch",
    "/signals/prices",
    "/signals/market-history",
    "/inference/sac",
}


def call(path, *, payload=None, name, timeout=600):
    method = "GET" if payload is None else "POST"
    route = path.split("?", 1)[0]
    if method == "POST" and route not in ALLOWED_POST:
        raise ValueError(f"Forbidden operation: {path}")
    if route == "/etl/news/sync" and payload != {"publish": False, "seed": False}:
        raise ValueError("Only HF-to-Pi restoration is allowed")
    request = Request(
        BASE + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    print(f"{datetime.now(UTC).isoformat()} {method} {path}", flush=True)
    started = datetime.now(UTC)
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
            status = response.status
    except HTTPError as exc:
        status = exc.code
        raw = exc.read().decode()
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {"body": raw}
    record = {
        "method": method,
        "path": path,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "status": status,
        "response": result,
    }
    (ROOT / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"HTTP {status}; saved {name}.json", flush=True)
    if status >= 400:
        raise RuntimeError(json.dumps(result))
    return result


def saved(name):
    return json.loads((ROOT / f"{name}.json").read_text())["response"]

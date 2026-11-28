"""Read-only Pi configuration, artifact, news, and HF-access diagnostics."""

import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import duckdb

token = os.environ.get("HF_TOKEN", "")
repo = os.environ.get("HF_NEWS_SENTIMENT_REPO", "").strip()
print(
    json.dumps(
        {
            "hf_token_present": bool(token),
            "hf_news_repo": repo,
            "storage_backend": os.environ.get("STORAGE_BACKEND"),
        }
    ),
    flush=True,
)
for label, url in (
    ("hf_token_validation", "https://huggingface.co/api/whoami-v2"),
    ("hf_news_repo_access", f"https://huggingface.co/api/datasets/{repo}"),
):
    if not token or (label == "hf_news_repo_access" and not repo):
        print(json.dumps({"check": label, "skipped": "missing configuration"}))
        continue
    try:
        request = Request(url, headers={"Authorization": "Bearer " + token})
        with urlopen(request, timeout=20) as response:
            data = json.load(response)
            details = (
                {"private": data.get("private")} if label.endswith("access") else {}
            )
            print(
                json.dumps({"check": label, "http_status": response.status, **details}),
                flush=True,
            )
    except HTTPError as exc:
        print(json.dumps({"check": label, "http_status": exc.code}), flush=True)
    except Exception as exc:
        print(
            json.dumps({"check": label, "error_type": type(exc).__name__}), flush=True
        )

root = Path("/app/data")
for bucket in ("patchtst_halal_new", "sac_halal_filtered", "sac_halal"):
    current = root / "models" / bucket / "current"
    if not current.exists():
        print(json.dumps({"bucket": bucket, "local_current_exists": False}), flush=True)
        continue
    version = current.read_text().strip()
    artifact = current.parent / version
    result = {"bucket": bucket, "version": version}
    for filename in ("config.json", "metadata.json"):
        path = artifact / filename
        if path.exists():
            data = json.loads(path.read_text())
            keys = (
                "market",
                "feature_names",
                "context_length",
                "context_calendar_days",
                "forecast_recipe",
                "patchtst_recipe",
                "training_timestamp",
                "training_date",
                "data_window",
                "news_schema_version",
                "finbert_revision",
            )
            result[filename] = {key: data[key] for key in keys if key in data}
    print(json.dumps(result, default=str), flush=True)

db = root / "news" / "news_v1.duckdb"
if db.exists():
    try:
        with duckdb.connect(str(db), read_only=True) as con:
            for label, query in (
                (
                    "news_events",
                    "SELECT count(*) AS rows, count(DISTINCT symbol) AS symbols, min(created_at) AS first_event, max(created_at) AS last_event FROM news_events",
                ),
                (
                    "news_coverage",
                    "SELECT count(*) AS rows, count(DISTINCT symbol) AS symbols, min(window_start_exclusive) AS first_start, max(window_end_inclusive) AS last_end FROM news_coverage",
                ),
                (
                    "coverage_windows",
                    "SELECT window_start_exclusive, window_end_inclusive, count(DISTINCT symbol) AS symbols FROM news_coverage GROUP BY 1,2 ORDER BY 2 DESC LIMIT 8",
                ),
            ):
                cur = con.execute(query)
                columns = [d[0] for d in cur.description]
                print(
                    json.dumps(
                        {
                            "check": label,
                            "rows": [dict(zip(columns, row)) for row in cur.fetchall()],
                        },
                        default=str,
                    ),
                    flush=True,
                )
    except Exception as exc:
        print(
            json.dumps(
                {"check": "news_db_read_only", "error_type": type(exc).__name__}
            ),
            flush=True,
        )

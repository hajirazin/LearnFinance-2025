"""Read existing model identities and monthly production universe caches."""

import json
from pathlib import Path

root = Path("/app/data")
result = {"models": {}, "universes": {}}
for market, bucket in (
    ("us", "patchtst_halal_new"),
    ("india", "patchtst_nifty_shariah_500"),
):
    directory = root / "models" / bucket
    pointer = directory / "current"
    if not pointer.exists():
        result["models"][market] = {"local_current_exists": False}
        continue
    version = pointer.read_text().strip()
    metadata = json.loads((directory / version / "metadata.json").read_text())
    config = json.loads((directory / version / "config.json").read_text())
    result["models"][market] = {
        "version": version,
        "symbols": metadata.get("symbols", []),
        "feature_names": config.get("feature_names"),
        "training_timestamp": metadata.get("training_timestamp"),
    }
for market, universe in (("us", "halal_new"), ("india", "nifty_shariah_500")):
    path = root / "cache/universe" / (universe + "_2026-11.json")
    result["universes"][market] = (
        json.loads(path.read_text()) if path.exists() else None
    )
print(json.dumps(result))

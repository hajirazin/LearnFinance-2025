"""Apply the production pure sticky selector to saved scores and read-only history."""

import json
from dataclasses import asdict
from pathlib import Path

from brain_api.core.sticky_selection import rank_by_score, select_with_rank_band

ROOT = Path(__file__).resolve().parent
SMOKE = ROOT.parent / "pi_manual_inference_2026_11_29"
state = json.loads((ROOT / "pi_state.json").read_text())
batch = json.loads((SMOKE / "full_patchtst_batch_us.json").read_text())["response"]
prior_rows = [
    r
    for r in state["alpha_selected_history"]
    if r["year_week"] == "202647" and r["final_allocation_pct"] is not None
]
prior = {r["stock"] for r in prior_rows}
selection = select_with_rank_band(batch["scores"], prior, top_n=15, hold_threshold=20)
ranks = {s: rank for s, rank, score in rank_by_score(batch["scores"])}
result = {
    "decision_date": "2026-11-23",
    "model_version": batch["model_version"],
    "previous_year_week": "202647",
    "previous_selected": sorted(prior),
    "selection": asdict(selection),
    "rows": [
        {
            "symbol": s,
            "rank": ranks[s],
            "forecast_pct": batch["scores"][s],
            "reason": selection.reasons[s],
        }
        for s in selection.selected
    ],
    "raw_top15": [
        {"symbol": s, "rank": rank, "forecast_pct": score}
        for s, rank, score in rank_by_score(batch["scores"])[:15]
    ],
    "forecast_anchor_date": batch["forecast_anchor_date"],
    "forecast_end_date": batch["forecast_end_date"],
}
(ROOT / "alpha_selection.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))

"""Read production strategy history and experience; never write Pi data."""

import json
import sqlite3
from pathlib import Path

root = Path("/app/data")
out = {}
db = root / "allocation/sticky_history.db"
with sqlite3.connect("file:" + str(db) + "?mode=ro", uri=True) as con:
    con.row_factory = sqlite3.Row
    out["weekly_periods"] = [
        dict(row)
        for row in con.execute(
            "SELECT universe,year_week,count(*) AS rows,sum(selected_in_final) AS selected "
            "FROM stage1_weight_history WHERE universe='halal_new_alpha' "
            "GROUP BY 1,2 ORDER BY year_week DESC LIMIT 5"
        )
    ]
    out["alpha_selected_history"] = [
        dict(row)
        for row in con.execute(
            "SELECT * FROM stage1_weight_history WHERE universe='halal_new_alpha' "
            "AND year_week IN ('202647','202648') "
            "AND (selected_in_final=1 OR final_allocation_pct IS NOT NULL) "
            "ORDER BY year_week,stage1_rank"
        )
    ]
    out["monthly_periods"] = [
        dict(row)
        for row in con.execute(
            "SELECT partition,period_key,count(*) AS rows,sum(selected) AS selected "
            "FROM screening_history GROUP BY 1,2 ORDER BY period_key DESC LIMIT 8"
        )
    ]

out["experiences"] = []
for path in (root / "experience").glob("*2026-11-23*"):
    data = json.loads(path.read_text())
    # Preserve the source record locally for accurate model/weight provenance.
    out["experiences"].append({"file": str(path.relative_to(root)), "data": data})
cache = root / "cache/universe/halal_filtered_2026-11.json"
out["monthly_filtered_cache"] = (
    json.loads(cache.read_text()) if cache.exists() else None
)
print(json.dumps(out, default=str))

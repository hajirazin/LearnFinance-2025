from datetime import datetime

from brain_api.core.weekly_decision import require_monday_decision_cutoff

as_of = datetime.fromisoformat("2026-08-31T09:00:00-04:00")
print(require_monday_decision_cutoff(as_of))

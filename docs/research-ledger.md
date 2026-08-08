# QuantBMAD research ledger

Append-only shared memory of settled (and failed) hypotheses. Every `qb-plan` run must search this file first via:

```bash
python3 _bmad/quantbmad/scripts/search_ledger.py --query "<topic>"
```

## Human audit rule

- Log **FAIL** as well as PASS — failures must not be silently discarded.
- Set `Human audit?` to `pending` for:
  - **every Go-live** row, and
  - **every 5th Research PASS** row (count PASS Research rows in this table).
- Razin spot-checks raw probe/experiment output against the Skeptic verdict, then sets `Human audit?` to `passed`.
- Until `passed`, do not treat that Skeptic PASS as fully settled ground truth (especially for Go-live).
- Use `n/a` for Quick rows and Research PASS rows not selected for sampling.

## Columns

| Date | Hypothesis | Track | Experiment path | Result | Skeptic verdict | Shipped? | Re-review by | Human audit? | Author model | Skeptic model |
|------|------------|-------|-----------------|--------|-----------------|----------|--------------|--------------|--------------|---------------|
| 2026-08-08 | Installing QuantBMAD v3 routes Research via qb-plan/qb-implement with CI backstop without reimplementing BMM | Go-live | _bmad/quantbmad/tests/ + sync_skill_mirrors --check | PASS | PASS | Yes | 2026-09-08 | pending | cursor-agent-implementer | cursor-agent-install-verification |

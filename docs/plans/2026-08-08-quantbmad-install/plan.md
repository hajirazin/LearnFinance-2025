# Plan: Inaugural QuantBMAD v3 install
Date: 2026-08-08
Track: Go-live
Requested by: Razin
Status: Approved

## Ledger search
no prior entry

## Problem / Request
Implement QuantBMAD v3 installation plan (canonical package, three skills mirrored to Cursor/Gemini/Codex, refuse hooks, CI gate, ledger, tests). User explicitly asked to implement the attached install plan in Agent mode.

## Hypothesis (Research/Go-live only)
Installing QuantBMAD per docs/quantbmad-spec-v3.md makes Research-glob changes route through qb-plan/qb-implement with CI backstop, without reimplementing BMM.
Falsification criteria: skill mirrors diverge; research_globs miss known math paths; gate scripts fail unit tests; BMM skills lack refuse hooks.

## Scope (one variable at a time)
- Included: QuantBMAD package, skills, mirrors, custom TOML, Cursor plan redirect, AGENTS.md section, help CSV, ledger, pre-commit, unit tests, amended spec in docs/
- Explicitly excluded: uninstalling BMM/WDS/CIS; marketplace packaging; changing brain_api math

## Small Verification Probe
Script: python3 -m pytest _bmad/quantbmad/tests -q
Result: see evidence-cert / CI
Verdict: Supports hypothesis
Skeptic probe adequacy: PASS (inaugural meta-install; verification = unit tests + mirror check)

## Architecture Impact
Adds `_bmad/quantbmad/`, IDE skill mirrors, pre-commit hooks; does not change Temporal/brain_api runtime math.

## Risk & Compliance Notes
Go-live for process/compliance tooling only — no live trading flip. research_globs.py introduced as gate list.

## Skeptic Pre-Review
docs/plans/2026-08-08-quantbmad-install/skeptic-plan-review.md

## Open Questions Asked to Human
None — user directed: implement the plan as specified.

## Next Step
QuantBMAD installed; future Research-glob app changes use qb-plan / qb-implement.

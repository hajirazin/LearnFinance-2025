---
name: qb-implement
description: >-
  QuantBMAD implementation for LearnFinance-2025. Requires human-Approved
  plan.md AND green ruff + brain_api + temporal tests before start and before
  done. Full evidence, party quantbmad consensus, Skeptic cert after consensus.
  Never skips failing tests as unrelated — fix them. Never writes Status Approved.
---

# qb-implement — QuantBMAD Implement

Spec: `{project-root}/docs/quantbmad-spec-v3.md`.

## Section 0 — stop and ask Razin

Same triggers as qb-plan. Peers only — only Razin has authority.

## Hard rules

1. No skill/agent may write `Status: Approved` on the plan file.
2. Chat `Approved <plan path>` may unblock this invocation only if plan was not created in the same user turn; CI still needs human-edited Approved.
3. Research-glob **edits** forbidden until evidence-cert PASS **and** implement-party consensus Approve (or Razin override) **and** approval gate.
4. **q-skeptic** is the only author of `evidence-cert.md`, and only **after** consensus Agree / Razin override. `written_by: qb-agent-skeptic`.
5. Implementation is **not done** without `party-consensus-implement.md` validating Approve or escalate+razin_decision.
6. Exact plan scope only.
7. **Repo-green (mandatory):** Must run and pass **before starting** and **before marking done**:
   ```bash
   python3 {project-root}/_bmad/quantbmad/scripts/check_repo_green.py --phase pre
   # ... implement ...
   python3 {project-root}/_bmad/quantbmad/scripts/check_repo_green.py --phase post
   ```
   Checks: `ruff check` + `ruff format --check` on `brain_api`, `temporal`, `_bmad/quantbmad`; then `uv run pytest` in `brain_api`; then `uv run pytest` in `temporal`.
8. **No “unrelated” escape:** If ruff or any test fails — even clearly unrelated to this plan — **fix it** (or stop and ask Razin with a specific blocker). You may **not** claim “out of scope / not my change” and proceed. Implement is incomplete until repo-green is exit 0.

## Procedure

### 0. Approval gate

Read `plan.md`. Abort unless human `Status: Approved` or valid current-user Approved message.

Also require plan-time `party-consensus-plan.md` already valid (from qb-plan).

### 0b. Pre-implement repo-green (ABORT if red)

```bash
python3 {project-root}/_bmad/quantbmad/scripts/check_repo_green.py --phase pre
```

If non-zero: **do not start** evidence work or code edits. Fix all failures (related or not), re-run until green, or ask Razin if stuck.

### 1. Full evidence (Research/Go-live; skip empirical gate for Quick)

Author writes full experiment under `docs/plans/<slug>/experiments/` (import read-only).

### 2. Mandatory party consensus (`quantbmad`)

1. `bmad-party-mode --party quantbmad --mode subagent` with stripped packet: `{hypothesis, experiment path, falsification criteria, ledger excerpt, proposed ship summary}`.
2. Up to **3** rounds; unanimous Agree or escalate to Razin.
3. Write `docs/plans/<slug>/party-consensus-implement.md` (`kind: implement`).
4. Validate with `validate_party_consensus.py`.

### 3. Skeptic cert (after consensus Agree or Razin override)

**qb-agent-skeptic** independently re-runs experiment and writes `evidence-cert.md` (template under `_bmad/quantbmad/templates/`). Prefer different model than author. Parent must not overwrite verdict.

- PASS → ledger row; continue  
- FAIL → ledger; back to qb-plan  
- AMBIGUOUS → ask Razin  

### 4. Implementation

**qb-agent-dev** (peer) implements exact scope with AGENTS.md facts. While fixing repo-green failures found pre/post, those fixes are allowed even if outside the plan’s one-variable scope (repo-green overrides “exact scope only” for making the suite pass).

### 5. Code review / validation

Separate peer pass (Architect/Dev review lens + Validation Specialist numbers). Still peers — deadlock → Razin.

### 6. Go-live checklist (Go-live track)

Risk peer surfaces; Razin confirms before live-adjacent ship.

### 7. Post-implement repo-green (ABORT if red — not done)

```bash
python3 {project-root}/_bmad/quantbmad/scripts/check_repo_green.py --phase post
```

If non-zero: implement is **not complete**. Fix everything (related or unrelated), re-run. Do not summarize as done.

### 8. Ledger + summary

Append ledger (Human audit? pending for Go-live and every 5th Research PASS). Summarize for Razin only after post repo-green is exit 0.

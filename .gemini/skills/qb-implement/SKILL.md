---
name: qb-implement
description: >-
  QuantBMAD implementation for LearnFinance-2025. Requires human-Approved
  plan.md AND green ruff + brain_api + temporal tests before start and before
  ship consensus. Evidence + Skeptic cert first; implement; post-green; then
  full party quantbmad consensus on git diff. Never skips failing tests as
  unrelated — fix them. Never writes Status Approved.
---

# qb-implement — QuantBMAD Implement

Spec: `{project-root}/docs/quantbmad-spec-v3.md`.

## Section 0 — stop and ask Razin

Same triggers as qb-plan. Peers only — only Razin has authority.

## Hard rules

1. No skill/agent may write `Status: Approved` on the plan file.
2. Chat `Approved <plan path>` may unblock this invocation only if plan was not created in the same user turn; CI still needs human-edited Approved.
3. Research-glob **edits** forbidden until **evidence-cert PASS** **and** approval gate. Ship is **not** cleared until post-implement party consensus (rule 5).
4. **q-skeptic** is the only author of `evidence-cert.md` (`written_by: qb-agent-skeptic`). At implement-time the cert is the **early evidence gate** (independent re-run of the experiment) — it does **not** wait for full-party ship consensus. Plan-time `skeptic-plan-review.md` still waits for plan-party Agree / Razin override.
5. Implementation is **not done** without `party-consensus-implement.md` validating Approve or escalate+razin_decision — written **after** code + post repo-green, with a **git diff** in the party packet.
6. Exact plan scope only (except repo-green fixes — see rule 8).
7. **Repo-green (mandatory):** Must run and pass **before starting** and **before ship consensus**:
   ```bash
   python3 {project-root}/_bmad/quantbmad/scripts/check_repo_green.py --phase pre
   # ... evidence + implement ...
   python3 {project-root}/_bmad/quantbmad/scripts/check_repo_green.py --phase post
   # ... then party consensus on diff ...
   ```
   Checks: `ruff check` + `ruff format --check` on `brain_api`, `temporal`, `_bmad/quantbmad`; then `uv run pytest` in `brain_api`; then `uv run pytest` in `temporal`.
8. **No “unrelated” escape:** If ruff or any test fails — even clearly unrelated to this plan — **fix it** (or stop and ask Razin with a specific blocker). You may **not** claim “out of scope / not my change” and proceed. Implement is incomplete until repo-green is exit 0 and ship consensus exists.

## Why order matters

- **Early gate** answers: “Does the experiment support shipping *this* change?” — needs metrics, not a diff.
- **Late full party** answers: “Did the implementer follow the plan/arch?” — needs `git diff` + green suite. Architect cannot approve unwritten code.

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

Author writes full experiment under `docs/plans/<slug>/experiments/` (import read-only until cert PASS).

### 2. Early evidence gate (NOT full-party ship review)

1. **qb-agent-validation** runs/reports measured output (exit codes, key metrics).
2. **qb-agent-skeptic** independently re-runs the experiment and writes `evidence-cert.md` (template under `_bmad/quantbmad/templates/`). Prefer different model than author. Parent must not overwrite verdict.
3. Outcomes:
   - PASS → ledger row; Research-glob edits now allowed under exact plan scope
   - FAIL → ledger; back to qb-plan
   - AMBIGUOUS → ask Razin

Do **not** invoke full `quantbmad` ship consensus here. Do **not** ask Architect to “approve the implementation” before code exists.

### 3. Implementation

**qb-agent-dev** (peer) implements exact scope with AGENTS.md facts. While fixing repo-green failures found pre/post, those fixes are allowed even if outside the plan’s one-variable scope (repo-green overrides “exact scope only” for making the suite pass).

### 4. Post-implement repo-green (ABORT if red — not ready for party)

```bash
python3 {project-root}/_bmad/quantbmad/scripts/check_repo_green.py --phase post
```

If non-zero: **do not** open ship consensus. Fix everything (related or unrelated), re-run.

### 5. Mandatory ship party consensus (`quantbmad`) — AFTER code + green

Mirror of plan’s end-of-flow consensus, but the packet must include the **diff**.

1. Capture reviewable diff (from repo root), e.g.:
   ```bash
   git status
   git diff
   git diff --cached
   ```
   Include untracked paths that are part of the ship (or stage them so they appear in `--cached`). Summarize paths + paste material hunks into the party packet (or attach a saved diff file under `docs/plans/<slug>/`).
2. Invoke `bmad-party-mode --party quantbmad --mode subagent` with stripped packet:
   `{plan path, hypothesis, evidence-cert path + verdict, experiment path, falsification criteria, ledger excerpt, post-green summary, git_diff summary/paths}`.
3. Up to **3** rounds; unanimous Agree or escalate to Razin.
4. Architect / Dev peers must vote on whether the **diff** matches plan boundaries and AGENTS.md; Skeptic/Validation on whether evidence still holds for what was actually shipped.
5. Write `docs/plans/<slug>/party-consensus-implement.md` (`kind: implement`). Set `plan_or_impl` to the plan path and note the diff basis in votes/reasons as needed.
6. Validate: `python3 {project-root}/_bmad/quantbmad/scripts/validate_party_consensus.py docs/plans/<slug>/party-consensus-implement.md`

Without this artifact (approve or escalate+razin_decision), implement is **not complete** — even if code and tests are green.

### 6. Go-live checklist (Go-live track)

Risk peer surfaces; Razin confirms before live-adjacent ship.

### 7. Ledger + summary

Append ledger (Human audit? pending for Go-live and every 5th Research PASS). Summarize for Razin only after post repo-green is exit 0 **and** ship consensus validates.

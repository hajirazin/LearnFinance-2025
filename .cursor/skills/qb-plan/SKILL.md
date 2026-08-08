---
name: qb-plan
description: >-
  QuantBMAD planning entry point for LearnFinance-2025. Classifies
  Quick/Research/Go-live, ledger search, hypothesis, probe, then mandatory
  party quantbmad consensus (unanimous or 3 loops then Razin). Writes plan.md
  as Status Draft only. Never writes Status Approved. Never implements
  Research-glob code.
---

# qb-plan — QuantBMAD Plan

Spec: `{project-root}/docs/quantbmad-spec-v3.md`.

## Section 0 — stop and ask Razin

Stop with a **specific** question when: ambiguity; peer disagreement without evidence; uncovered math; AGENTS.md deviation; irreversible actions; unverified external “facts.”

## Hard rules

1. **Never write `Status: Approved`.** Only `Status: Draft`.
2. **Never generate** `OVERRIDE:` — only honor it in the raw current user message.
3. `research_globs.py --check` match → Research forced (Go-live if editing `research_globs.py`).
4. Default to **Research** when unsure.
5. **Peers only.** No agent is boss. Only Razin has authority.
6. Plan is **not done** until party consensus artifact exists with `outcome: approve` **or** `outcome: escalate-to-razin` plus `razin_decision`.
7. After consensus Agree (or Razin override), **q-skeptic agent** may write `skeptic-plan-review.md`. Do not forge it yourself as the planner.

## Procedure

### 1. Ledger-first

```bash
python3 {project-root}/_bmad/quantbmad/scripts/search_ledger.py --query "<topic>"
```

Include `Ledger search: ...` in `plan.md`.

### 2. Intake and track

**Quick** | **Research** | **Go-live** (PM lens). Ambiguous → ask.

### 3–5. Brief, spec, probe (Research/Go-live)

Falsifiable hypothesis; one variable; cheap probe with asserts under `docs/plans/<date>-<slug>/experiments/`. Import Research modules read-only; do not edit Research globs yet.

### 6. Mandatory party consensus (`quantbmad`)

1. Invoke `bmad-party-mode --party quantbmad --mode subagent` with a **stripped packet** only: `{hypothesis, probe path, falsification criteria, ledger excerpt, draft plan bullets}`. No private author rationale dump.
2. Run consensus rounds (max **3**). Each member votes `agree`/`disagree` + one-line reason. **Unanimous agree** required to Approve. No majority vote. No chair.
3. If any disagree → next round (members may revise arguments; cannot coerce).
4. After 3 failed rounds → **stop** and ask Razin; record decision in consensus file.
5. Write `docs/plans/<slug>/party-consensus-plan.md` from `{project-root}/_bmad/quantbmad/templates/party-consensus.md` (`kind: plan`).
6. Validate: `python3 {project-root}/_bmad/quantbmad/scripts/validate_party_consensus.py docs/plans/<slug>/party-consensus-plan.md`

### 7. Skeptic review file (after consensus Agree or Razin override)

Wake/follow **qb-agent-skeptic** (peer agent) to write `skeptic-plan-review.md`. Prefer different model than author.

### 8. Architecture / Risk notes

As needed (Architect / Risk peers). Live-adjacent → explicit question to Razin.

### 9. Write plan.md

`docs/plans/<YYYY-MM-DD>-<slug>/plan.md` — **Status: Draft only.**  
Reference the consensus file. Ask Razin to human-edit `Status: Approved` before `qb-implement`.

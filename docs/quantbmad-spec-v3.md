# QuantBMAD for LearnFinance-2025 — v3 (merged, loopholes closed)

**Audience:** Implementing / reviewing AI. This is the single source of truth. It supersedes v1 and v2.

**Amendments (party consensus):** Skeptic is a **peer agent** (`qb-agent-skeptic`), not a workflow skill. All Quant agents sit in party `quantbmad` (see `_bmad/custom/bmad-party-mode.toml`). `qb-plan` / `qb-implement` are incomplete without unanimous party Agree or 3-loop escalate-to-Razin. No agent is boss of any other — only Razin.

**Amendments (repo-green):** `qb-implement` must run `check_repo_green.py` **before start** and **before done** (ruff + brain_api pytest + temporal pytest). Failing checks cannot be skipped as unrelated — fix them or ask Razin.

**Scope:** LOCAL expansion of BMad Method inside `LearnFinance-2025` only. Do **not** create a separate module repo, do **not** ship `marketplace.json`, do **not** design for reuse across other projects. Keep BMM, Core, TEA, and BMB installed. QuantBMAD sits **in front** of them and reuses `bmad-create-story`, `bmad-dev-story`, `bmad-code-review`, `bmad-agent-dev`, `bmad-agent-architect`. Do **not** reimplement PRD, architecture, epics, sprint planning, or generic code review.

**Locked decisions:** A2 (real skills + `_bmad/custom` overrides) + party `quantbmad` peers + 3-loop consensus then Razin.

**Human:** Razin. Every escalation in Section 0 goes to Razin with a specific, answerable question.

---

## 0. The one rule that overrides everything else

Every agent, in every workflow step, must **stop and ask Razin** (not assume) when any of the following is true:

- The requirement, hypothesis, or success metric is ambiguous or under-specified.
- Two agents disagree and cannot resolve it with evidence already in hand.
- A change touches math/model/reward/allocator/universe-ranking logic and existing empirical evidence does not clearly cover this exact case.
- A change would deviate from an `AGENTS.md` invariant (math > DRY, no silent fallbacks, paper-default/live-opt-in, sticky partition uniqueness, weight-band ≠ rank-band, bucket isolation, `.NS` suffix, no Monday training) — even if the agent believes the deviation is justified.
- An action is irreversible or hard to revert (live trading flip, deleting/overwriting model artifacts, dropping a DB table, force-pushing).
- The agent is about to state a "fact" about external research, library behavior, or broker rules that it has not verified against a primary source or the actual repo code **in this session**.

Ask a **specific** question (e.g. “Hypothesis claims X; probe cannot test entropy side-effect Y — test in isolation, or out of scope?”). Never quietly pick the conservative or aggressive option and continue.

This rule applies inside `/qb-plan`, `/qb-implement`, and `/qb-skeptic` at every step.

---

## 1. Three skills only (multi-IDE install)

QuantBMAD exposes **exactly three** skills. Same names, same bodies, installed so **Cursor, Gemini CLI, and Codex** all discover them.

### 1.1 The three skills

| Skill | Role |
|-------|------|
| `qb-plan` | Single entry for intake → track → (research path) → probe → plan.md (`Status: Draft` only) |
| `qb-implement` | Approved plan → full evidence gate → implement → review → validation → ledger |
| `qb-skeptic` | Falsify-only. Manual for Go-live; also the skill body Task/subagents must follow for Research SoD |

### 1.2 Canonical source + three IDE trees

**Canonical (edit here only):**

```
_bmad/quantbmad/
├── research_globs.py          # SINGLE SOURCE OF TRUTH for Research path matching (skills + CI import this)
├── scripts/
│   ├── search_ledger.py
│   └── check_research_gate.py # pre-commit/CI entry
├── templates/
│   ├── plan.md
│   ├── evidence-cert.md
│   └── skeptic-verdict.md
└── skills/
    ├── qb-plan/SKILL.md
    ├── qb-implement/SKILL.md
    └── qb-skeptic/SKILL.md
```

**Install identical copies** (byte-identical **literal** `SKILL.md` files — **no symlinks**) into all three discovery roots:

| Tool | Project skill root | Required paths |
|------|--------------------|----------------|
| **Codex** | `.agents/skills/` | `.agents/skills/qb-plan/SKILL.md`, `qb-implement/SKILL.md`, `qb-skeptic/SKILL.md` |
| **Gemini CLI** | `.gemini/skills/` (also reads `.agents/skills/`; install both to be explicit) | `.gemini/skills/qb-plan/SKILL.md`, `qb-implement/SKILL.md`, `qb-skeptic/SKILL.md` |
| **Cursor** | `.cursor/skills/` **and** `.agents/skills/` | `.cursor/skills/qb-plan/SKILL.md`, `qb-implement/SKILL.md`, `qb-skeptic/SKILL.md` **plus** the `.agents/skills/` copies |

**Rule:** There are three skill *names* (`qb-plan`, `qb-implement`, `qb-skeptic`). Each has one canonical `SKILL.md` under `_bmad/quantbmad/skills/`, and that file is mirrored into **Cursor + Gemini + Codex** discovery dirs so no IDE can “not see” QuantBMAD. Do not maintain divergent bodies per IDE.

**Sync requirement:** Any edit to a QuantBMAD skill body is done in `_bmad/quantbmad/skills/...` first, then mirrored via `python3 _bmad/quantbmad/scripts/sync_skill_mirrors.py` (literal `shutil.copy2`; refuses/replaces symlinks). CI fails if `.agents/skills/qb-*/SKILL.md`, `.cursor/skills/qb-*/SKILL.md`, and `.gemini/skills/qb-*/SKILL.md` differ from canonical (checksum) or if any mirror is a symlink.

### 1.3 bmad-help registration

Register all three in `_bmad/_config/bmad-help.csv` (and/or module help merge) with:

- `required=true` on the Research/Go-live path that precedes implementation
- `preceded-by` / `followed-by` wiring to existing BMM IDs (`bmad-create-story`, `bmad-dev-story`, `bmad-code-review`) so help routes QuantBMAD **in front of** BMM implement, not in parallel as an optional side quest

### 1.4 `_bmad/custom/` — overrides only (not skill bodies)

```
_bmad/custom/bmad-agent-dev.toml
_bmad/custom/bmad-agent-architect.toml
_bmad/custom/bmad-quick-dev.toml
_bmad/custom/bmad-dev-story.toml
_bmad/custom/bmad-dev-auto.toml          # if present / used
```

Plus Cursor-local plan skill override/replacement — see §3.

---

## 2. Commands and tracks

### 2.1 `/qb-plan` (or skill `qb-plan`)

Single entry point for anything from “fix a bug” to “add a new allocator.” Order:

1. **Ledger-first (mandatory)**  
   Run: `python3 _bmad/quantbmad/scripts/search_ledger.py --query "<topic>"`  
   Write into `plan.md`: `Ledger search: <results or 'no prior entry'>`.  
   If that line is missing, the plan is incomplete — do not continue drafting.

2. **Intake & track classification** — Quant PM classifies:
   - **Quick** — routes, Temporal wiring (non-allocation-math), email templates, non-math refactors, doc/test-only
   - **Research** — forecaster, RL reward, HRP/sticky math, universe ranking, promotion thresholds, broker cost math, **any path matching `research_globs.py`**
   - **Go-live/Compliance** — live env flip, new broker, new universe screening rule, safety cap changes, edits to `research_globs.py` itself

   Ambiguous → STOP and ask. **Default to Research if unsure.**  
   Path match against `research_globs.py` → **Research forced**. Agent may **not** self-override. Only a human message in **that turn** containing `OVERRIDE: <reason>` may reclassify (agent must never generate the `OVERRIDE:` string itself; validate against the raw user message only).

3. **Research brief** (Research/Go-live only) — falsifiable hypothesis: claim, why expected true, what would prove it wrong. Ban vague “should improve performance.”

4. **Strategy/model spec** (Research/Go-live only) — exact inputs, exact math, target metric, **one variable**. Bundled unrelated changes → split into separate plans.

5. **Small verification probe** — Quant Validation Specialist writes and runs a cheap isolated script (100–500 step train probe, numerical simulation, tiny backtest). Probe must include **asserts** that fail if the falsification criteria fire. Contradicts hypothesis → do not accept plan; revise or escalate.

6. **Plan-time Skeptic (SoD)** — Invoke Skeptic via **separate Task/subagent** (or require Razin to run `/qb-skeptic` in a fresh chat) with input **ONLY**: `{hypothesis, probe script path, falsification criteria, ledger excerpt}`. No authoring transcript. Same-session “I am Skeptic now” = **INVALID**. Skeptic must confirm probe asserts are adequate; may reject the probe. Write result to `docs/plans/<date>-<slug>/skeptic-plan-review.md` (Skeptic Task writes this file; plan author must not).

7. **Architecture impact** (Research/Go-live, or Quick touching >1 service) — reused BMM Architect + `AGENTS.md` facts.

8. **Risk & Compliance pre-check** (Go-live always; Research if universe/screening/broker) — live-adjacent → explicit human confirmation question before finalizing Draft.

9. **Write `plan.md`** at `docs/plans/<date>-<short-slug>/plan.md` with **`Status: Draft` only**.  
   **`qb-plan` MUST NEVER write `Status: Approved`.**  
   Present summary and ask: “Plan ready at `.../plan.md`. Edit Status to Approved (or reply Approved naming that path) when ready for `/qb-implement`.”

### 2.2 `/qb-implement` (or skill `qb-implement`)

Takes an approved plan. First steps are hard gates:

0. **Approval gate**  
   - Read `plan.md`.  
   - Proceed only if **either**: (a) file `Status: Approved` was set by a **human file edit** (not by any agent in this or prior agent turns), **or** (b) the **current user message** explicitly approves that plan path (e.g. `Approved docs/plans/.../plan.md`) **and** the plan was not created/modified by the assistant in the same user turn.  
   - **No skill may write `Status: Approved` into the file.** Chat approval unblocks this invocation only; CI still requires human-edited `Status: Approved` before merge of Research diffs.  
   - If neither holds → **ABORT**.

1. **Full evidence gate** (Research/Go-live; skip Quick)  
   - Author writes full isolated experiment under `docs/plans/<slug>/experiments/` or `scratch/` — may **import** Research-glob modules read-only; **must not edit** those paths yet.  
   - Invoke Skeptic/Validator as **separate Task** with ONLY `{hypothesis, experiment path, ledger excerpt, plan falsification criteria}`. Task follows `qb-skeptic/SKILL.md`.  
   - **Only that Task may write** `docs/plans/<slug>/evidence-cert.md`. Parent reads it. Parent must not create/overwrite verdict `PASS`. Missing cert or `FAIL` → no prod edits.  
   - `PASS` → append ledger row; continue.  
   - `FAIL` → ledger row + return to `/qb-plan` (new cycle).  
   - `AMBIGUOUS`/`PARTIAL` → STOP; ask Razin; no confident reframe.

2. **Go-live extra:** Razin must run `/qb-skeptic` in a **fresh chat** before step 3; verdict file required.

3. **Implementation** — reused BMM Dev (`bmad-dev-story` patterns / Expert Dev with `AGENTS.md` facts). Exact plan scope only. Incomplete plan → STOP and ask.

4. **Code review** — reused BMM code review; **different** session/Task from implementer. Focus `AGENTS.md` invariants.

5. **Validation** — Validation Specialist Task: re-run evidence against **merged/prod** code; walk-forward/regression as specified; pytest (scoped with rationale, or full suite). Patch≠plan bugs must fail here.

6. **Risk/Go-live checklist** (Go-live only) — HF `make_current`, CAGR floor, per-account live flip, partition/A-B isolation. Human confirmation mandatory before live-adjacent ship.

7. **Ledger drift row** — what shipped, evidence, `Re-review by` date.

8. **Final summary to Razin** — what changed, evidence, track; ask merge/deploy or hold.

### 2.3 `/qb-skeptic` (or skill `qb-skeptic`)

- Falsify only; **no fix authorship**.  
- Input contract: hypothesis + paths + ledger excerpt only.  
- Writes `skeptic-plan-review.md` and/or `evidence-cert.md` / `skeptic-verdict.md` per templates.  
- Required manually (fresh chat) on **Go-live** before implement proceeds.  
- Used as the Task prompt body for Research SoD.

---

## 3. Block alternate entry points (Cursor / Gemini / Codex)

### 3.1 BMad skills — refuse Research globs

`_bmad/custom/bmad-quick-dev.toml`, `bmad-dev-story.toml`, and `bmad-dev-auto.toml` (if used):

```toml
[workflow]
activation_steps_prepend = [
  "Resolve changed/target paths with: python3 _bmad/quantbmad/research_globs.py --check <paths>",
  "If any path is Research-tracked: REFUSE. Output exactly: 'This touches Research-track code. Use qb-plan first.' Do not implement.",
]
```

### 3.2 Cursor Plan skill — closed (prior fatal hole)

Replace or wrap [`.cursor/skills/plan/SKILL.md`](.cursor/skills/plan/SKILL.md) so activation **first**:

1. If intent/paths match Research globs → **REFUSE** and redirect to `qb-plan`.  
2. Do not call CreatePlan / write an implementation plan for Research-tracked work outside QuantBMAD.

Also add a short **Cursor rule** (e.g. `.cursor/rules/quantbmad.mdc` or project rule): Research-glob work → only `qb-plan` / `qb-implement` / `qb-skeptic`.

### 3.3 AGENTS.md backstop

Add a QuantBMAD section to `AGENTS.md`:

- Research-glob changes require QuantBMAD artifacts (`plan.md` Approved + `evidence-cert.md` PASS).  
- Do not use Plan-mode-only, quick-dev, dev-auto, or loop to ship Research-glob diffs.  
- `bmad-loop` / unattended loops must not touch Research globs without those artifacts.

### 3.4 Gemini / Codex

Same three skills under `.gemini/skills/` and `.agents/skills/`. No separate “just plan” skill that bypasses globs; if a tool-specific plan skill exists, apply the same refuse prepend.

---

## 4. Canonical Research globs (`research_globs.py`)

**Single source of truth** — skills and CI import this module. **Not** maintained only inside SKILL.md (prevents agents shrinking the list in prose).

Initial list (tune to repo; must include known math/order/cost/universe surfaces):

```python
QUANTBMAD_RESEARCH_GLOBS = [
    "brain_api/brain_api/core/**",
    "brain_api/brain_api/universe/**",
    "brain_api/brain_api/storage/ibkr_orders.py",
    "brain_api/brain_api/routes/orders.py",
    "**/sticky*.py",
    "**/strategy_partitions.py",
    "**/promotion*.py",
    "**/hrp.py",
    "**/broker_costs.py",
    "**/rewards.py",
    "temporal/workflows/**",  # allocation/orchestration semantics
    "_bmad/quantbmad/research_globs.py",  # editing the gate list = Compliance
]
```

- Match → force Research (or Go-live if the matched file is `research_globs.py` or live/safety config).  
- Changing `research_globs.py` → **Go-live/Compliance** + human `OVERRIDE:` + mandatory `/qb-skeptic`.

---

## 5. Hard SoD, approval, probe, evidence authorship

| Rule | Enforcement |
|------|-------------|
| Author ≠ Skeptic/Validator | Separate Task/fresh chat; stripped input; same-session role-play = invalid |
| Plan-time Skeptic | Required before Draft is presented for approval (§2.1 step 6) |
| Go-live | Extra manual `/qb-skeptic` in fresh chat |
| `Status: Approved` | **No agent/skill may write it**; human edits file for CI; chat Approved only unblocks current `qb-implement` |
| Probe adequacy | Skeptic confirms asserts ↔ falsification criteria; checkbox in plan |
| `evidence-cert.md` | **Only Skeptic/Validator Task writes verdict**; parent read-only |
| Prod edits | Forbidden on Research globs until `evidence-cert` PASS **and** approval gate satisfied |
| Scratch | May `import` Research modules read-only |
| Cross-model Skeptic | Where feasible, Skeptic/Validator runs on a **different model provider** than the authoring agent (e.g. Claude authors → Grok or GPT Skeptic, and vice versa). Session isolation prevents context leakage; model diversity prevents shared blind spots. Record `author_model` and `skeptic_model` on evidence-cert. If only same-model is available, stop and ask Razin. |

### 5.1 `evidence-cert.md` minimum schema

```markdown
# Evidence cert
plan_path: docs/plans/<slug>/plan.md
hypothesis: <text>
experiment_path: <path>
command: <exact command run>
exit_code: <int>
key_metrics: <bullet list>
falsification_criteria_checked: <yes + notes>
author_model: <provider/model>
skeptic_model: <provider/model — prefer different vendor>
skeptic_task_id_or_fresh_chat: <id or "fresh-chat:<date>">
verdict: PASS | FAIL | AMBIGUOUS
written_by: skeptic-task | qb-skeptic-fresh-chat
# Parent agents must not edit the verdict field.
```

CI checks: file present, schema fields present, `verdict: PASS`, linked `plan.md` has `Status: Approved`.

---

## 6. `plan.md` schema

```markdown
# Plan: <short title>
Date: <date>
Track: Quick | Research | Go-live
Requested by: Razin
Status: Draft | Approved

## Ledger search
<results or 'no prior entry'>

## Problem / Request
<verbatim or lightly cleaned>

## Hypothesis (Research/Go-live only)
<falsifiable claim>
Falsification criteria: <what disproves this>

## Scope (one variable at a time)
- Included: ...
- Explicitly excluded: ...

## Small Verification Probe
Script: <path>
Result: <summary>
Verdict: Supports | Contradicts | Inconclusive (escalated)
Skeptic probe adequacy: PASS | REJECT (see skeptic-plan-review.md)

## Architecture Impact
<...>

## Risk & Compliance Notes
<...>

## Skeptic Pre-Review
<path to skeptic-plan-review.md>

## Open Questions Asked to Human
<question → answer; unanswered = do not proceed>

## Next Step
Proceed to qb-implement only after Status: Approved (human).
```

---

## 7. Mandatory agents

1. **Quant Researcher** — hypotheses, experiment design  
2. **Quant Portfolio Manager** — track classification, go-live business call  
3. **ML Engineer** — `brain_api/core/{lstm,patchtst,sac}` / training  
4. **Risk & Compliance Officer** — halal, safety caps, live gating, broker/currency; veto-by-escalation to human (cannot self-approve live)  
5. **Skeptic** — falsify-only; SoD; writes certs/reviews  
6. **Quant Validation Specialist** — runs probes and post-merge empirical checks; executes and reports numbers  
7. **Expert Dev / Architect** — reused BMM agents + `persistent_facts` from `AGENTS.md`  

Segregation: hypothesis author / implementer ≠ Skeptic, Validator grader, or code reviewer instance.

---

## 8. Research ledger

Path: `docs/research-ledger.md` (append-only, shared).

| Date | Hypothesis | Track | Experiment path | Result | Skeptic verdict | Shipped? | Re-review by | Human audit? | Author model | Skeptic model |
|------|------------|-------|-----------------|--------|-----------------|----------|--------------|--------------|--------------|---------------|

Log **FAIL** as well as PASS. Every `qb-plan` searches before restating settled facts.

**Periodic Skeptic audit:** every **5th Research PASS** row and **all Go-live** rows set `Human audit? = pending`. Razin personally spot-checks raw probe/experiment output against the Skeptic’s stated verdict, then sets `passed`. Until then, do not treat that Skeptic PASS as fully settled ground truth. This catches Skeptic hallucination/drift without reviewing every entry.

---

## 9. Facts seeded into agent persistent context (`AGENTS.md`)

- Math correctness > DRY — never break math to simplify code  
- No silent fallbacks  
- Paper default; live = per-account env opt-in  
- Sticky partitions unique; weight-band ≠ rank-band  
- Model buckets isolated (`current` pointers)  
- India `.NS` end-to-end  
- No Monday training  
- `paper:` in `run_id` / `client_order_id` is cosmetic — live vs paper = `ALPACA_{ACCOUNT}_URL`  

---

## 10. Honest limits + hard backstop

Prompt/skill rules make correct behavior the default; they are not cryptographically unbypassable. Hard backstop:

**Pre-commit + CI** (`_bmad/quantbmad/scripts/check_research_gate.py`):

On any diff matching `research_globs.py`:

1. Require `docs/plans/<slug>/plan.md` with `Status: Approved` in the same PR (or referenced by cert).  
2. Require `evidence-cert.md` with schema + `verdict: PASS` + `written_by` skeptic.  
3. Fail commit/PR otherwise.  
4. Optionally: fail if IDE mirrors of `qb-*/SKILL.md` diverge from `_bmad/quantbmad/skills/`.

Local `--no-verify` is a human escape; do not use it for Research work.

---

## 11. Repo layout (final)

```
_bmad/quantbmad/                    # canonical logic + skill bodies
_bmad/custom/*.toml                 # BMad overrides / refuse hooks
.agents/skills/qb-{plan,implement,skeptic}/SKILL.md    # Codex + shared
.cursor/skills/qb-{plan,implement,skeptic}/SKILL.md    # Cursor
.gemini/skills/qb-{plan,implement,skeptic}/SKILL.md    # Gemini CLI
.cursor/skills/plan/SKILL.md        # redirected/refusing Research
.cursor/rules/quantbmad.mdc         # Cursor rule backstop
docs/plans/<date>-<slug>/plan.md
docs/plans/<date>-<slug>/evidence-cert.md
docs/plans/<date>-<slug>/skeptic-plan-review.md
docs/plans/<date>-<slug>/experiments/
docs/research-ledger.md
AGENTS.md                           # QuantBMAD section
.pre-commit-config.yaml             # hook → check_research_gate.py
```

---

## 12. Implementation order for the building AI

1. Add `research_globs.py` + `search_ledger.py` + `check_research_gate.py` + templates.  
2. Write canonical `qb-plan` / `qb-implement` / `qb-skeptic` under `_bmad/quantbmad/skills/`.  
3. Mirror the three `SKILL.md` files into `.agents/skills/`, `.cursor/skills/`, and `.gemini/skills/`.  
4. Wire `_bmad/custom` TOML refuse hooks + Dev/Architect facts.  
5. Redirect `.cursor/skills/plan` + add Cursor rule + `AGENTS.md` section.  
6. Register help CSV rows.  
7. Seed empty `docs/research-ledger.md`.  
8. Add pre-commit hook; run once on a dry-run Research path.  
9. Do **not** uninstall BMM.

---

## 13. Reviewer checklist (devil’s advocate)

Confirm before accepting an implementation of this spec:

- [ ] Three skills exist and are **byte-identical** across `_bmad/quantbmad/skills`, `.agents`, `.cursor`, `.gemini`  
- [ ] Cursor Plan skill refuses Research globs  
- [ ] Globs live in `research_globs.py`, not only in markdown  
- [ ] No skill writes `Status: Approved`  
- [ ] Only Skeptic Task/`qb-skeptic` writes evidence verdict  
- [ ] Plan-time SoD is separate Task or fresh `/qb-skeptic`  
- [ ] Go-live requires fresh-chat `/qb-skeptic`  
- [ ] Pre-commit/CI gate on Research diffs  
- [ ] BMM not reimplemented  
- [ ] Section 0 escalate-to-human present in all three skill bodies
- [ ] Skill mirrors are literal copies (no symlinks); checksum gate enforced
- [ ] Cross-model Skeptic preference + author_model/skeptic_model on certs
- [ ] Ledger human-audit sampling (1/5 Research PASS + all Go-live)

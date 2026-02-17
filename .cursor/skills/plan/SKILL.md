---
name: plan
description: Create structured implementation plans for features, fixes, and refactors. Use when the user asks to plan, design an approach, break down a task, or is in plan mode. Enforces reuse-first, math correctness, file size limits, DDD naming, and test planning.
---

# Implementation Planning

Produce structured, actionable implementation plans that enforce engineering discipline before any code is written.

## Core Principles (enforce in every plan)

### 1. Reuse first

Before planning any new code, search the codebase for existing helpers, utilities, patterns, or similar implementations. Plans must explicitly list:

- What existing code will be reused
- What is genuinely new

Duplication is a planning failure. Best engineers factor out and reuse similar code.

### 2. Math correctness over code simplicity

Always write clean code, but **NEVER** simplify mathematical formulas, algorithms, or numerical logic for the sake of readability.

- If a formula is correct but complex, **keep it correct**
- Annotate with comments explaining the math instead of simplifying
- Math bugs are the most expensive bugs in a finance/ML system -- they silently corrupt every downstream decision
- Reference source papers or documentation for non-trivial formulas

### 3. No big files

Every new or modified file must stay under **600 lines**. If a plan would push a file over the limit, include a refactoring step to split by responsibility -- not arbitrarily.

### 4. Domain-accurate naming (DDD)

All classes, functions, variables, and files must use real-world domain names that match the business domain:

- `WeeklyReturnForecast`, not `ModelOutput`
- `AllocationWeight`, not `result`
- `HalalScreeningDecision`, not `item`
- `OrderIdempotencyKey`, not `key`

Plans must call out naming choices for all new entities.

## Structured Planning Process

### Step 1: Requirements analysis

Understand the "what" and "why". Clarify ambiguity before proceeding.

### Step 2: Reuse scan

Search the codebase for:

- Existing helpers and utilities that can be leveraged
- Similar implementations or patterns already in use
- Shared modules that should be extended rather than duplicated

List reusable pieces explicitly in the plan output.

### Step 3: Impact analysis

Identify all affected:

- Files and modules
- Domains and bounded contexts
- API endpoints (existing and new)
- Data schemas and storage
- Downstream consumers

### Step 4: Architecture alignment

Verify the plan respects architecture boundaries (reference `Agent.md`):

- **Prefect** = orchestrator only (schedule, call brain_api, track status)
- **brain_api** = all business logic (screening, signals, forecasting, allocation, orders)
- No business logic in Prefect tasks beyond orchestration
- API design: stateless, storage abstraction, JSON in/out, idempotent training, thin endpoints

### Step 5: Risk assessment

Evaluate:

- Breaking changes to existing APIs or data formats
- Data migrations required
- Math correctness risks (formulas, numerical stability, edge cases)
- Impact on non-negotiable invariants (see below)

### Step 6: Dependency mapping

Order tasks by dependencies. Identify parallelizable work.

### Step 7: Task breakdown

Create actionable, sequenced steps. For each task:

- Specify the file(s) to create or modify
- Check file size -- add refactor step if approaching 600 lines
- Note naming choices for new entities
- Note what is being reused vs. written new

### Step 8: Test planning

For every feature or change, plan which tests to write or update:

- **Routers/handlers:** API integration tests that call the endpoint. Test constraints like `min_items`, `max_items`, min/max length via API calls. Never write schema-only tests.
- **Pure functions** (feature transforms, idempotency keys, screening logic): deterministic unit tests
- **Goal:** 100% business logic coverage (all edge cases, error paths, boundary conditions) -- NOT 100% code coverage
- Sequence test tasks after the implementation task for the same module

### Step 9: Invariant validation

Check the plan against non-negotiable invariants:

- `run_id = paper:YYYY-MM-DD`, `attempt` starts at 1
- Rerun is read-only after any non-terminal order
- `client_order_id` format: `paper:YYYY-MM-DD:attempt-<N>:<SYMBOL>:<SIDE>`
- Monday = inference only, never training
- Training writes new versioned artifact, never overwrites
- Promotion requires evaluation gate
- Endpoints remain stateless
- Storage abstraction used (not hardcoded paths)
- LSTM = pure price (no signals), PatchTST = OHLCV 5-channel, PPO/SAC = RL with dual forecasts

## Project-Specific Context

Reference `Agent.md` for full details on:

- Architecture boundaries and code structure
- API endpoint inventory and design rules
- Model hierarchy (LSTM, PatchTST, PPO, SAC, HRP)
- Signal state vector composition
- Data storage rules and model versioning
- Operational requirements (idempotency, timeouts, retries, observability)
- Prefect flow configuration (`persist_result=True`, task retries)

## Plan Output Template

Every plan must follow this structure:

```markdown
## Goal
<What and why>

## Reuse Inventory
- <existing_module.function> -- reuse for <purpose>
- <existing_pattern> -- extend for <purpose>
- (genuinely new: <list>)

## Impact Analysis
- Files affected: ...
- APIs affected: ...
- Data/storage affected: ...

## Approach
<High-level design with naming choices for new entities>

## Tasks
1. <task> -- file: <path>, reuse: <what>, new: <what>
   - [ ] Implementation
   - [ ] Tests: <what to test>
2. ...

## Test Plan
- Unit tests: <list with coverage targets>
- Integration tests: <list with API calls to make>
- Edge cases: <specific scenarios>

## Risks
- <risk and mitigation>

## Cleanup (always include)
- [ ] Fix all ruff linting issues (related and unrelated to the change)
- [ ] Run and fix all tests (related and unrelated to the change)
```

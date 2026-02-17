---
name: review
description: Review code for correctness, math rigor, reuse, DDD naming, security, performance, and test quality following project standards. Use when reviewing pull requests, code changes, or when the user asks for a code review, review feedback, or quality check.
---

# Code Review

Structured code review that prioritizes math correctness and enforces reuse, file size limits, and domain-accurate naming.

## Priority-Ordered Review Principles

These four principles are **hard requirements**. Violations are always Critical severity.

### 1. Math correctness is non-negotiable

Review mathematical formulas, algorithms, and numerical logic with extreme rigor:

- **Never** approve a simplification that alters mathematical behavior
- Flag any "cleaned up" math as **Critical**
- Verify formulas against reference papers or documentation
- Check numerical stability (overflow, underflow, division by zero, floating-point precision)
- In a finance/ML system, a math bug silently corrupts every downstream decision

### 2. Reuse violations

Flag any new code that duplicates existing helpers, utilities, or patterns:

- Search the codebase for existing implementations before approving new ones
- "Could this reuse X?" is a **mandatory** review question for every new function
- Suggest the specific existing code to reuse when flagging

### 3. File size

Reject any file exceeding **600 lines**:

- Suggest split-by-responsibility refactoring
- This is a hard limit, not a guideline

### 4. Domain naming (DDD)

Reject generic names. All names must reflect real-world domain concepts:

- **Reject:** `data`, `result`, `output`, `item`, `process`, `handler`, `manager`, `utils`
- **Accept:** `AllocationWeight`, `WeeklyReturnForecast`, `HalalScreeningDecision`, `OrderIdempotencyKey`
- Naming is a first-class review concern, not a style nit

## Review Dimensions

Evaluate every change across these dimensions, in priority order:

### 1. Math / algorithmic correctness (HIGHEST PRIORITY)

- Are formulas correct and complete?
- Numerical stability: overflow, underflow, precision loss
- Edge cases in calculations (empty arrays, single-element, zero values)
- Does the implementation match the referenced paper or algorithm?
- Are mathematical comments accurate?

### 2. Correctness

- Logic bugs and off-by-one errors
- Edge cases and boundary conditions
- Error handling completeness
- Null/None handling
- Async/concurrency issues

### 3. Security

- Input validation and sanitization
- Injection vulnerabilities (SQL, command, path traversal)
- Secrets or credentials in code
- API key exposure

### 4. Performance

- Unnecessary loops or repeated computation
- N+1 query patterns
- Memory leaks or unbounded growth
- Missing pagination for large datasets

### 5. Reuse

- Duplicated logic across files
- Missed opportunities to use existing helpers
- New utilities that should be shared modules

### 6. Maintainability

- Readability and clarity (without sacrificing math correctness)
- Single responsibility principle
- File size under 600 lines
- Function length and complexity
- Clear separation of concerns

### 7. Naming

- Domain-accurate, DDD-aligned naming for all new entities
- Consistent terminology throughout the change
- Names match business concepts, not implementation details

### 8. Test quality

- Business logic coverage: all edge cases, error paths, boundary conditions
- No schema-only tests (schemas validated through API usage)
- Routers tested via API integration tests (call the endpoint, test `min_items`, `max_items`, constraints)
- Pure functions tested with deterministic unit tests
- Goal: 100% business logic coverage, NOT 100% code coverage

### 9. Architecture compliance

- DDD alignment: domain boundaries, aggregate roots, service layers
- Layer boundaries respected (core has no FastAPI dependency)
- Stateless endpoints (no in-memory state across requests)
- Storage abstraction used (not hardcoded paths)
- JSON in, JSON out for core functions
- Thin route handlers (validate + call core + return response)

## Project-Specific Safety Checklists

### Trading logic changes

- [ ] Rerun behavior is still read-only after any order submission
- [ ] `client_order_id` format unchanged (or migration handled)
- [ ] Safety caps exist and are enforced (max turnover, max orders, cash buffer)

### ML / model code changes

- [ ] Monday inference does NOT trigger training
- [ ] Training writes new versioned artifact (not overwrite)
- [ ] Promotion requires evaluation gate
- [ ] Endpoints remain stateless (no global model cache)
- [ ] Storage abstraction is used (not hardcoded paths)
- [ ] LSTM remains pure-price (no signals in input)
- [ ] PatchTST receives OHLCV 5-channel input
- [ ] PPO/SAC receive correct signal state vector with dual forecasts

### Operational requirements

- [ ] Idempotency: safe reruns
- [ ] Timeouts + retries with exponential backoff for external APIs
- [ ] Rate limit awareness and batching
- [ ] Observability: `run_id` + `attempt` in logs, stage duration metrics, clear error propagation

## Review Output Format

Structure feedback by severity and dimension:

**Critical** -- must fix before merge:
- Math correctness issues (always Critical)
- Reuse violations with specific existing code to use
- File size violations
- Security vulnerabilities
- Broken invariants (trading safety, model lifecycle)

**Suggestion** -- consider improving:
- Performance improvements
- Better naming alternatives
- Test coverage gaps
- Architecture alignment opportunities

**Nice to have** -- optional enhancement:
- Style improvements (that don't affect math)
- Documentation additions
- Minor readability tweaks

## Reference

See `Agent.md` for full project context including:

- Architecture boundaries and code structure
- Complete API endpoint inventory
- Model hierarchy and signal state vector
- Data storage rules and model versioning
- Non-negotiable invariants
- Operational requirements

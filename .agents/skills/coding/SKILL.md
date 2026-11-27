---
name: coding
description: Implement repository features, bug fixes, and refactors, including execution of an approved plan. Use for authorized code changes with small reviewable steps, reuse, domain-driven design, clean code, test-driven development, offline tests, and protected production data/model storage. Verify the real execution path while preserving scope and compatibility. Planning-only requests belong to plan-generator; read-only reviews belong to review-agent.
---

# Coding

Implement the authorized change through the repository's real production path.
An existing plan is an implementation input, not evidence that its assumptions
or tests match the current checkout.

## Work in small, reviewable steps

- Implement one cohesive behavior or defect at a time. Keep the application
  runnable between steps; complete required dependent changes before calling a
  slice done. Separate mechanical moves/renames from behavior changes when that
  makes the diff easier to assess.
- Review each completed slice with its tests before building the next one.
  Recheck changed assumptions at the next boundary. Do not postpone all review
  until a large multi-concern diff has accumulated.
- Keep functions focused and modules cohesive. Follow the repository's file-size
  limit: changed/new source files must stay below 600 lines. Split by domain or
  responsibility, never by arbitrary line ranges or numbered continuation files.
- Keep PRs and commits focused when they are requested. Explain necessary
  cross-cutting changes rather than forcing an unsafe partial change to meet an
  arbitrary diff size. Small steps do not authorize committing or opening PRs.
- Separate new requirements discovered along the way from the authorized task.
  Fix issues needed for the task and required repository checks; record other
  findings without bundling speculative redesigns or unrelated cleanup.

## Enter the right workflow

- Read applicable repository instructions and inspect the working tree. Preserve
  user edits and unrelated work.
- If the user asks only for a plan, use `$plan-generator` when available and stop
  at the plan. If implementation is authorized, execute it; do not restart a
  planning-only workflow or request approval already given.
- When implementing an existing plan, check its paths, signatures, dependencies,
  and acceptance criteria against the checkout. Correct routine stale details
  within scope. Surface conflicts that require a new business decision before
  doing the dependent work; continue independent authorized work.

## Establish the change contract

Identify requested behavior changes, behavior to preserve, and the evidence that
will distinguish success from a passing but irrelevant test.

- Preserve unrequested defaults and hyperparameters. For requested configuration
  changes, trace version/hash identities, artifact compatibility, and downstream
  consumers. Do not rewrite existing artifacts merely to match new defaults.
- Trace the default endpoint or entry point through dependency injection,
  application orchestration, core functions, and storage. Confirm which code
  actually runs before editing it or choosing mocks.
- For replacement code, preserve applicable logs, metrics, progress/status,
  cancellation, error propagation, idempotency, and persistence/publication
  ordering, alongside the domain outputs.
- Check active jobs before edits that could trigger auto-reload, restart, or
  shutdown. Reuse known session state; do not restart a job just to obtain logs.
- Keep operational actions within authorization. Code changes do not imply
  permission to train production models, publish artifacts, back up databases,
  submit orders, or rewrite unrelated release documentation.

## Reuse and design in the domain

- Search for existing functions, services, registries, adapters, fixtures, and
  test builders before creating another implementation. Read their behavior and
  callers, including error and boundary cases. Extend or compose a suitable
  implementation instead of adding a near-duplicate with another name.
- Reuse the same domain rule; do not merge rules merely because their code looks
  similar. Preserve algorithm-specific mathematics and context boundaries.
  Extract a shared abstraction only when its meaning and invariants are shared.
- Use the domain's existing vocabulary for functions, types, modules, and
  parameters. Make units, dates/timezones, identities, and ownership explicit
  where they affect correctness. Keep model buckets, account identities, and
  strategy partitions distinct according to repository rules.
- Protect invariants at the boundary that owns them. Keep domain calculations
  independent of HTTP request objects, environment lookup, provider clients, and
  filesystem operations. Application services coordinate use cases; adapters
  translate I/O. Do not add enterprise layers or aggregate classes to a simple
  function merely to claim DDD compliance.
- Prefer the simplest design that meets the current requirement: explicit data
  flow, cohesive functions, narrow interfaces, and composition. Apply single
  responsibility and dependency inversion where they reduce coupling. Avoid
  speculative frameworks, extension points, feature flags, and generic helpers
  whose only purpose is a hypothetical future use.

## Keep code clear and failures explicit

- Use descriptive names and explicit inputs/results. Replace meaningful magic
  values with named domain constants. Avoid boolean mode switches that combine
  unrelated responsibilities, deeply nested branches, and hidden mutable state.
- Keep side effects visible. Inject storage, clocks, and provider boundaries
  where the architecture supports them rather than hiding access in utilities
  or module imports. Use context managers and bounded resource lifetimes.
- Validate external data at its boundary; do not repeatedly parse or serialize
  it inside core calculations. Preserve precision, missing-value semantics,
  temporal ordering, and numerical behavior during refactoring.
- Catch exceptions only where they can be handled meaningfully. Preserve the
  cause and return an explicit failure for unexpected errors. Do not turn a
  failed provider call into empty data, zero signals, or a successful job unless
  that behavior is an explicit domain contract.
- Comments explain non-obvious reasons, constraints, or math. Remove stale
  comments and unreachable code; do not narrate obvious statements. Use the
  existing formatter, linter, type conventions, and public API style.

## Implement with behavioral evidence

For behavioral changes, follow red → green → refactor: reproduce the failure or
write the missing behavior test, observe the relevant failure, implement the
smallest complete fix, then refactor with tests green. Do not claim a red phase
that was not run. For moves with unchanged behavior, use existing behavioral
tests and verify imports/callers rather than inventing tests of file layout.

- Exercise the actual default endpoint/configuration with synthetic inputs and
  temporary artifacts. An overridden legacy configuration or a mock of a
  bypassed trainer does not cover the new production path.
- Stub external provider and publication boundaries deliberately. Keep focused
  core tests real; use deliberately small model configurations and synthetic
  data when model execution is needed.
- When retiring an implementation, search all callers, migrate relevant tests,
  and remove obsolete branches, dependencies, and exports. Preserve checkpoint
  decoding/inference compatibility needed by installed artifacts or rollback.

## Make ordinary tests offline and disposable

Isolation covers reads, writes, deletion, and publication, not just network calls.

- Use the repository's offline guard. Where absent and isolation is part of the
  task, add enforcement at the actual transports/artifact boundaries, including
  native/curl-backed clients when used. Record attempted I/O so application
  exception handling cannot turn an accidental external call into a passing
  test. Do not weaken the guard or substitute permissive mocks to make it green.
- Never download live market/news/provider data or pretrained weights in the
  ordinary test suite. Use small synthetic datasets with explicit timestamps,
  expected boundary cases, and deterministic seeds where randomness is needed.
  Do not make tests pass only because provider data or a model happens to be
  cached on the developer's laptop.
- Construct every test model/data store under a per-test temporary root. Inject
  that root before initialization or background work. Resolve symlinks and paths
  before destructive operations; fail if the target escapes the owned test root.
  Cover real serialization with tiny temporary artifacts when required.
- Never read, overwrite, promote, rename, delete, or clean up the application's
  real model/data folders in a test. Never swap a real `current` pointer or move
  installed weights aside and restore them afterward. A backup is not test
  isolation. Cleanup may remove only resources created and owned by that test.
- Clear or replace production credentials, publication repositories, and storage
  settings before imports/initialization that consume them. Account for dotenv
  loading and default storage constructors. Test publication against a fake
  adapter with temporary artifacts, never a configured production HF repo.
- Reset test-owned environment variables, registries, caches, and other mutable
  state through scoped fixtures. Tests must work alone, in a different order,
  and without leftovers from a previous run. Control the clock when dates affect
  behavior; await background work and release resources during teardown.
- Keep real-provider smoke tests separate and excluded from the default suite.
  Run them only when explicitly authorized for the specified test destination;
  their failure must not be hidden in a normally passing offline test.

## Test behavior at the right level

- Use focused unit tests for domain rules and numerical invariants, API tests
  for endpoint behavior, and a small number of integration tests for boundaries
  that need real wiring. Follow repository-specific rules, including exercising
  validation through API calls rather than writing schema-only tests.
- Assert observable outcomes and important invariants: meaningful values, state
  transitions, failures, and allowed side effects. Avoid tests that repeat the
  implementation, pin private call order unnecessarily, or assert only that a
  mock was called. Do not mock the behavior the test claims to verify.
- Include relevant edge cases such as empty or non-finite data, cutoffs, invalid
  identities, retries, and cancellation. Prefer a small fixture that exposes the
  defect over a large fixture that merely makes the happy path run.
- Keep test doubles faithful to the provider contract used by the code. Check
  adapter mapping with representative synthetic payloads; do not broaden fakes
  until they accept responses the real provider would reject.
- Keep feedback fast. Use tiny models and short synthetic histories consistent
  with the tested contract; do not add full model training, research ablations,
  long sleeps, or giant fixtures as routine regression checks. Reuse production
  math where appropriate, but keep expected-value calculations independent.

## Follow state across boundaries

For background jobs, retries, synchronization, or publication changes:

- Verify accepted, running, terminal, and retry states together with worker
  ownership. Cover duplicate requests and transitions near completion/lock
  release. Returning an accepted response is not evidence the job completed.
- Pass cancellation through every fitting entry point, including annual or
  other secondary fits and recovery. Verify framework shutdown ordering; a
  lifespan shutdown hook may run after background tasks are drained.
- Preserve completed work for retry and record recoverable partial-publication
  state. Check cancellation before an externally visible commit; once committed,
  finish necessary local finalization rather than leaving contradictory pointers.
- Inspect configured storage policy and authoritative local/shared evidence
  before declaring data missing or downloading it. When changing shared restore,
  verify an empty store restores published evidence and an existing store reuses
  it without redundant provider work. Preserve writes made during publication.

These checks apply to the affected behavior; do not add unrelated lifecycle,
storage, or deployment machinery to a simple edit.

## Keep operational changes bounded

- Give external operations finite timeouts. Retry only transient failures and
  operations that are safe to retry, with bounded attempts/backoff. Propagate
  terminal failures; do not create endless retry loops to obtain a green status.
- Preserve useful structured logs with job/run identity, stage, progress,
  duration, and actionable errors. Keep credentials, tokens, and unnecessary raw
  payloads out of logs, fixtures, and committed artifacts.
- Watch for unbounded memory use, repeated I/O, and per-item remote calls when
  changing data/model paths. Batch or reuse evidence where supported. Measure
  the affected path when performance is part of the task; avoid speculative
  optimization that obscures correct code.
- Express reproducible configuration changes in versioned code/configuration.
  Preserve locked dependencies and documented artifact compatibility. Keep
  production workflows free of scratch experiments and diagnostic retraining.

## Verify and hand off

- Run focused tests during development and the required project checks at the
  end. Inspect failures, including attempted I/O reported at fixture teardown.
  Report commands/results actually observed and any blocked checks.
- Keep automated checks and commit hooks enabled. Fix the underlying failure;
  do not bypass hooks, delete meaningful assertions, loosen tolerances without
  mathematical justification, or hide failures behind skips or broad catches.
- Review the final tracked and relevant untracked diff for scope drift,
  unintended configuration changes, lost operational behavior, obsolete callers,
  and compatibility breaks. Follow repository size and architecture rules.
- Use `$review-agent` for a requested read-only review when available. Do not
  label self-review independent verification or invoke delegation without
  authorization.
- Update the requested fixes/checklist document with verified completed and
  pending items. Keep release procedures separate unless they were requested.
- Report what changed, the evidence, and material limits. State whether any
  production operation was actually run. Do not commit, push, publish, or start
  production work beyond the user's authorization.

Keep skill text, generated documentation, and examples suitable for a public
repository: no organization branding, private links, credentials, production
datasets, or machine-specific paths unless the user explicitly requests them.

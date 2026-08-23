# Conventional Commit Skill

You are the **Release-Ready Commit** agent. You produce commits that are
Conventional Commits 1.0.0 compliant and safe for automated semantic
releases (`semantic-release`, `release-please`, `commitizen`).

Your output is a commit that a machine can parse to decide the next
SemVer bump. Every commit must be unambiguous, atomic, and reviewable.

## Reference example (must emulate)

Commit `f3c4dbefe822b50efc80846dad00aa072b51c68c`:

```
feat(snapshot): Implement canonical vs rejected snapshot persistence

- Added `persist_forecaster_snapshot` function to handle writing of canonical and rejected snapshots based on model health.
- Introduced `SnapshotPersistResult` dataclass to encapsulate the result of snapshot persistence operations.
- Enhanced `SnapshotLocalStorage` with methods for writing rejected snapshots and checking their existence.
- Created utility functions for writing snapshot artifacts and evicting stale directories.
- Integrated HuggingFace upload/download functionality into snapshot management.
- Updated metadata creation to include failure reasons for rejected snapshots.
- Added comprehensive tests for snapshot persistence, including scenarios for healthy and unhealthy models.
- Ensured that rejected snapshots do not interfere with existing canonical snapshots.
```

Pattern to copy:
- Header: `type(scope): Subject` imperative, lowercase after `:`, no period, <=72 chars
- Blank line
- Body: `- ` bullets, each a complete sentence starting with verb in past/present (`Added`, `Fixed`, `Enhanced`), wrapping what + why
- No footer here (no BREAKING CHANGE / Closes)

## Specification (Conventional Commits 1.0.0)

Format:
```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### 1. Header

- `type` MUST be one of: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`
  - `feat`: new user-facing feature. Release: **minor** (`0.x.0` / `x.1.0`)
  - `fix`/`perf`: bug / performance fix. Release: **patch** (`0.0.x`)
  - `BREAKING CHANGE` or `!` after type/scope: **major** (`x.0.0`)
  - All others (`chore`, `docs`, `style`, `refactor`, `test`, `build`, `ci`): **no release** unless `BREAKING CHANGE`
- `scope` SHOULD be present, lowercase, kebab or single word. Repo scopes:
  `snapshot`, `patchtst`, `lstm`, `sac`, `research`, `temporal`, `brain_api`, `universe`, `storage`, `deps`, `config`
  Use `research` for `scratch/**` experiments, `snapshot` for forecaster snapshots, `patchtst` for forecaster code.
- `description` MUST be imperative mood, no capital first letter after `:`, no trailing period, <=72 chars.

### 2. Body

- Blank line after header, then `- ` bullets.
- Each bullet: what changed + where (`file:line` if useful) + why. Keep factual, no superlatives.
- Wrap at 100 chars if possible. Use backticks for symbols (`persist_forecaster_snapshot`, `SnapshotLocalStorage`).
- For multi-concern commits, group bullets by area but keep single type/scope; prefer atomic commits if `feat` vs `chore` would conflict.

### 3. Footer

- `BREAKING CHANGE: <description>` for breaking API/behavior.
- `Closes #<issue>` / `Refs #<issue>` on new line.
- Use `!` in header alternatively: `feat(snapshot)!: ...`

### 4. Release intent mapping (semantic-release)

| Commit | Release |
|---|---|
| `feat` | minor |
| `fix`, `perf` | patch |
| `feat!`, `fix!`, `BREAKING CHANGE:` | major |
| `chore`, `docs`, `style`, `refactor`, `test`, `build`, `ci` | none |

Choose `chore(research)` for `scratch/**` audits that must not trigger a release. Choose `feat` only when production code gains capability.

## Workflow (MUST follow in order)

1. **Inspect**: `git status --porcelain`, `git diff --stat HEAD`, `git diff HEAD`, `git log --oneline -10` to understand history style.
2. **Group**: Make commits atomic. Do not mix `feat` + `chore` in one commit. If staged changes span scopes, split.
3. **Stage**: `git add <paths>` only intended files. Never `git add -A` blindly if unrelated untracked files exist.
4. **Craft message**: Follow header/body/footer rules above. Base scope on primary area. Keep body bullets derived from actual diff, not hallucinated.
5. **Commit**: `git commit -m "<header>" -m "<body>"` (use heredoc or `-m` per paragraph; preserve blank line).
6. **Verify**: `git show --stat HEAD`, `git log --oneline -1`
7. **Push**: `git push` (or `git push --set-upstream origin <branch>` if needed) only after verification.

## Rules

- NEVER use `WIP`, `update`, `fix stuff`. Always conventional.
- NEVER add silent fallbacks in code to make a commit pass; surface errors.
- NEVER create markdown files unless required (the commit body is the documentation).
- ALWAYS verify with `git show` that bullet claims match the diff.
- ALWAYS keep commits under ~500 lines diff avg; split large `scratch` binary blobs if needed but this repo allows research artifacts.
- Research artifacts (`scratch/**`) are `chore(research)` by default, not `feat`, to avoid release churn.

## Anti-patterns

- `feat: stuff` (missing scope) -> `feat(snapshot): ...`
- `Fix bug` (capital, no scope) -> `fix(patchtst): ...`
- No body -> Add 4-10 bullets like the reference commit.
- Mixing prod + research in one `feat` -> Split.

## Self-check before commit

- [ ] Header matches `<type>(<scope>): <description>`?
- [ ] Description imperative and <=72 chars?
- [ ] Body has `- ` bullets derived from `git diff`?
- [ ] Release impact matches intent (chore = no bump, feat = minor)?
- [ ] Scope is from allowed list?

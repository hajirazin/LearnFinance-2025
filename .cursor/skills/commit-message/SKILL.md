---
name: commit-message
description: Generate conventional commit messages by analyzing staged git diffs. Use when the user asks to commit, write a commit message, or review staged changes for committing.
---

# Commit Message Generator

## Critical Rules

### Never ask for confirmation

When this skill is invoked, the user wants to commit **now**. Do NOT ask "shall I commit?" or "ready to proceed?" -- analyze, generate the message, and commit immediately.

### Never bypass pre-commit hooks

- **FORBIDDEN:** `--no-verify`, `-n`, `noqa`, skip-test, skip-ruff
- **CORRECT:** `git commit -S -m "message"` (hooks run normally)
- If hooks fail, **fix the issue** instead of bypassing

### Never add CoAuthor

No `Co-authored-by` lines for any AI agent.

### Always prevent pager blocking

Prefix every git command with `GIT_PAGER=cat`:

```bash
GIT_PAGER=cat git diff --cached --name-status
GIT_PAGER=cat git diff --cached
GIT_PAGER=cat git log --oneline -5
```

## Workflow

### Step 1: Check staged files

```bash
GIT_PAGER=cat git diff --cached --name-status
GIT_PAGER=cat git diff --cached --stat
```

If **nothing is staged**, stop immediately:

```
Nothing is staged for commit.

Stage changes first:
  git add <file>     # specific files
  git add -A         # all changes
  git add -p         # interactive staging
```

### Step 2: Analyze changes

Read the actual diff:

```bash
GIT_PAGER=cat git diff --cached
```

For each staged file:

1. **Categorize**: new feature, bug fix, refactor, test, docs, config, dependency
2. **Identify patterns**: related changes across files, domain/component being modified, breaking changes

### Step 3: Generate commit message

Format:

```
<type>(<scope>): <short summary>

<detailed description>

Changes:
- <bullet point 1>
- <bullet point 2>

Impact:
- <impact description>

Files Modified:
- <file 1>
- <file 2>
```

**Types:** `feat`, `fix`, `refactor`, `test`, `docs`, `style`, `chore`, `perf`

**Scope:** domain/component name (e.g., `inference`, `allocation`, `etl`). Use `*` for cross-cutting changes.

**Rules:**
- Summary line under 72 characters
- Imperative mood ("add feature" not "added feature")
- Be specific about what changed, not just where
- Highlight breaking changes prominently

### Step 4: Commit immediately (no confirmation)

**Do NOT ask the user for confirmation. Execute the commit directly.**

Only signed commits:

```bash
git commit -S -m "<message>"
```

Display success confirmation with the commit hash.

## Example Output

```
feat(allocation): add HRP risk-parity allocation endpoint

Implemented hierarchical risk parity allocation using covariance matrix
input, producing portfolio weights that respect risk-parity constraints.

Changes:
- Created HRP core function with quasi-diagonal clustering
- Added POST /allocation/hrp endpoint with weight response schema
- Added integration tests calling the endpoint with sample covariance data

Impact:
- Enables HRP as baseline allocator for comparison against PPO/SAC

Files Modified:
brain_api/
├── core/hrp.py (new)
├── routes/allocation.py (new)
└── tests/test_allocation_hrp.py (new)

Statistics: 3 files changed, 280 insertions
```

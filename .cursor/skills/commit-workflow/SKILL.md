---
name: commit-workflow
description: Full automated commit, branch, push, and PR workflow. Use when the user runs /commit, asks to commit and push, create a PR, or publish changes. Handles branch creation from main, signed commits, and GitHub PR management via gh CLI.
---

# Commit Workflow

Fully automated workflow: analyze changes, generate commit message, branch if needed, commit, push, and handle PR -- all without asking questions.

## Critical Rules

### Never bypass pre-commit hooks

- **FORBIDDEN:** `--no-verify`, `-n`, `noqa`, skip-test, skip-ruff
- **CORRECT:** `git commit -S -m "message"` (hooks run normally)
- If hooks fail, **fix the issue** instead of bypassing

### Auto-commit prevention

Once this workflow completes and exits:

- **NEVER** commit automatically in subsequent messages
- **NEVER** push unless the user explicitly runs `/commit` again
- If user says "commit this", respond: "Please run the `/commit` command to enter COMMIT MODE"

### Always prevent pager blocking

Prefix every git command with `GIT_PAGER=cat`:

```bash
GIT_PAGER=cat git diff --staged
GIT_PAGER=cat git diff
GIT_PAGER=cat git log --oneline -5
```

### Only signed commits

Every commit must use `git commit -S -m "..."`.

### No CoAuthor

No `Co-authored-by` lines for any AI agent.

## Workflow

### Step 1: Analyze changes and generate commit message

```bash
GIT_PAGER=cat git diff --staged   # or git diff if not staged
git status --short
```

Analyze what changed, why, and scope. Generate conventional commit message:

- Format: `<type>(<scope>): <description>`
- Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `style`, `perf`
- Show the generated message (informational, not asking approval)

### Step 2: Check current branch

```bash
git branch --show-current
```

### Step 3: Execute branch-appropriate workflow

**Workflow A -- on main/master:**

```bash
git stash push -m "Auto-stash before branch creation"
git pull origin main
git checkout -b <branch-name>       # generated from commit message
git stash pop
git add .
git commit -S -m "<message>"
git push -u origin <branch-name>
# Then create PR (step 5)
```

**Workflow B -- on feature branch:**

```bash
git add .
git commit -S -m "<message>"
git push                             # or git push -u origin <branch> if needed
# Then check/update PR (step 4 -> step 5)
```

### Step 4: Check for existing PR (Workflow B only)

```bash
gh pr view --json number,title,body,url
```

### Step 5: Handle PR

**If PR exists (update):**

- Keep existing PR content
- Add `## Latest Changes` section with: commit message, date, key files changed
- Update title if new changes warrant it

```bash
gh pr edit <number> --title "<title>" --body "<description>"
```

**If no PR exists (create):**

- Title: concise, follows semantic-release convention
- Body: summary, file list grouped by type, purpose, breaking changes, testing notes

```bash
gh pr create --title "<title>" --body "<description>" --base main
```

**If gh CLI unavailable:**

- Provide direct GitHub URL
- Provide pre-formatted title and description for manual creation

### Step 6: Exit and show summary

```
Branch: <branch-name>
Commit: "<message>"
Pushed successfully
PR: <URL> (created/updated)

COMMIT MODE now exited.
Future commits require explicitly running /commit command.
```

## Branch naming

Generate from commit message, kebab-case:

- "Add GPU filters" -> `feat/add-gpu-filters`
- "Fix memory leak" -> `fix/memory-leak`
- "Update documentation" -> `docs/update-documentation`

## Defaults (never ask to confirm)

- PR base branch: `main`
- Staging: `git add .`
- Use `gh` CLI if available, fallback to manual instructions
- No confirmation prompts -- execute fully automatically

## Error handling

| Error | Recovery |
|-------|----------|
| git stash fails | Check if changes exist; proceed without stash if none |
| git commit fails | Show error; check if no changes to commit; exit |
| git push fails | Try `git push -u origin <branch>`; if still fails, report and provide manual command |
| Branch creation fails | Check if branch exists; suggest switching to it |
| gh CLI missing | Provide GitHub URL + pre-formatted title/body for manual PR |
| PR operation fails | Show error; provide direct GitHub URLs and manual steps |

## PR description guidelines

**New PR:**
- Summary explaining the "why"
- Changes section listing what was modified
- Testing section if relevant
- Notes section for breaking changes or migration steps

**Updating existing PR:**
- Preserve all existing sections
- Append `## Latest Changes` with commit message, date, key files

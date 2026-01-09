# Commit Message Generator Command

## 🚨 CRITICAL: NEVER BYPASS PRE-COMMIT HOOKS

**NEVER USE `--no-verify` OR `-n` FLAGS WITH GIT COMMIT**

Pre-commit hooks run ruff linting and tests for a reason - to catch issues before they reach the repo. Bypassing them defeats the entire purpose of code quality gates.

- ❌ `git commit --no-verify` - FORBIDDEN
- ❌ `git commit -n` - FORBIDDEN  
- ❌ `git commit -S --no-verify` - FORBIDDEN
- ✅ `git commit -S -m "message"` - CORRECT (hooks will run)

If hooks fail, FIX THE ISSUE instead of bypassing the check.

## Purpose
Analyze all staged Git changes and generate a beautiful, detailed commit message that clearly explains what was changed, why, and the impact.

## ⚠️ CRITICAL: Prevent Pager/User Input Issues

**ALWAYS prefix git commands with `GIT_PAGER=cat` to prevent interactive pagers from blocking:**

```bash
# ✅ CORRECT - prevents pager
GIT_PAGER=cat git diff --cached --name-status
GIT_PAGER=cat git diff --cached
GIT_PAGER=cat git log --oneline -5

# ❌ WRONG - may open pager and hang
git diff --cached
git log
```

This prevents `less`, `vim`, or other pagers from opening and waiting for user input.

## Instructions

### Step 1: Check Staged Files
1. Run `GIT_PAGER=cat git diff --cached --name-status` to see what files are staged
2. Run `GIT_PAGER=cat git diff --cached --stat` to see the summary of changes
3. If **nothing is staged**, immediately stop and respond:
   ```
   ❌ Nothing is staged for commit.
   
   Please stage your changes first using:
   - `git add <file>` for specific files
   - `git add -A` for all changes
   - `git add -p` for interactive staging
   ```
   Then exit without proceeding further.

### Step 2: Analyze Changes (Only if files are staged)
For each staged file, examine:
1. **Read the actual diff** using `GIT_PAGER=cat git diff --cached` to understand exact changes
2. **Categorize changes**:
   - New features
   - Bug fixes
   - Refactoring
   - Tests
   - Documentation
   - Configuration
   - Dependencies
3. **Identify patterns**:
   - Related changes across multiple files
   - Domain/component being modified
   - Breaking changes or API modifications

### Step 3: Generate Commit Message
Create a commit message following this structure:

```
<type>(<scope>): <short summary>

<detailed description>

Changes:
- <bullet point 1>
- <bullet point 2>
- <bullet point 3>

Impact:
- <impact description>

Files Modified:
- <file 1>
- <file 2>
- <file 3>
```

#### Commit Type Guidelines:
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code refactoring without functional changes
- `test`: Adding or updating tests
- `docs`: Documentation changes
- `style`: Code style/formatting changes
- `chore`: Build process, dependencies, or tooling
- `perf`: Performance improvements

#### Scope Guidelines:
- Use domain/component names (e.g., `inventory`, `provider`, `frontend`, `backend`)
- Use `*` for changes spanning multiple areas
- Examples: `feat(reservations)`, `fix(api)`, `refactor(ui)`

CRITICAL: ONLY SIGNED COMMITS ARE ALLOWED.  

### Step 4: Commit with the Generated Message
1. Display the generated commit message in a beautiful, readable format with:
   - Clear sections and proper spacing
   - Emoji for visual appeal (✨ for features, 🐛 for fixes, ♻️ for refactoring, etc.)
   - File tree showing what was modified
   - Summary statistics (files changed, insertions, deletions)

2. **Immediately commit** using the generated message with `git commit -S -m "<message>"` (the `-S` flag signs the commit)

3. Display success confirmation with the commit hash

## Important Notes
- Always read actual file diffs, don't just guess from filenames
- Group related changes logically
- Be specific about what changed, not just where
- Highlight any breaking changes prominently
- Consider the project context (DDD, monorepo structure, etc.)
- Follow conventional commits format
- Keep the summary line under 72 characters
- Use imperative mood ("add feature" not "added feature")

## Example Output
```
✨ feat(reservation): Add matrix view for provider requests

Implemented a new pivot table component to display reservation requests
in a matrix format, allowing providers to see request counts organized
by GPU model and geographical region.

Changes:
- Created PivotTable shared component with generic type support
- Added RequestMatrixView component for reservation requests
- Implemented MatrixCountCell with click-to-expand functionality
- Added useMatrixData hook for data transformation
- Created comprehensive test suite for PivotTable component

Impact:
- Providers can now visualize request patterns more effectively
- Improved UX for handling multiple reservation requests
- Reusable PivotTable component for future features

Files Modified:
Frontend:
├── src/components/shared/PivotTable/
│   ├── PivotTable.tsx (new)
│   ├── types.ts (new)
│   ├── index.ts (new)
│   └── __tests__/PivotTable.test.tsx (new)
└── src/app/provider/requests/
    └── loading.tsx (modified)

Statistics: 10 files changed, 1,200+ insertions
```


You are now in COMMIT MODE. This is a SPECIALIZED mode for finalizing and publishing code changes. STRICTLY follow the workflow below.

## 🚨 CRITICAL: Auto-Commit Prevention Rules

**ONCE YOU EXIT COMMIT MODE, YOU MUST NEVER AUTO-COMMIT AGAIN IN THE SAME CHAT SESSION.**

After completing the commit workflow and exiting this mode:
- ❌ DO NOT commit changes automatically in subsequent messages
- ❌ DO NOT push changes unless explicitly told to run `/commit` command again
- ❌ DO NOT assume the user wants to commit just because they made changes
- ❌ DO NOT offer to commit or suggest committing changes
- ✅ Only commit again when user EXPLICITLY runs the `/commit` command
- ✅ If user asks "can you commit this?" respond: "Please run the `/commit` command to enter COMMIT MODE"

**Remember:** The user has explicitly configured this workflow to prevent auto-commits. Respect this preference absolutely.

## Commit Mode Workflow

**IMPORTANT: This workflow is FULLY AUTOMATED. No confirmation prompts. User runs /commit = execute everything.**

**Steps to follow:**

1. **Analyze Changes and Generate Commit Message**
   - Run `git diff --staged` (if files already staged) OR `git diff` (if not staged)
   - Run `git status --short` to see modified/added/deleted files
   - Analyze the changes to understand:
     - What was changed (files, functions, features)
     - Why it was changed (bug fix, new feature, refactor, docs, etc.)
     - Scope of changes (backend, frontend, tests, docs, infrastructure)
   - Generate a clear, descriptive commit message following convention:
     - Format: `<type>: <description>`
     - Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `style`, `perf`
     - Example: "feat: add GPU filtering with memory range support"
     - Example: "fix: resolve memory leak in reservation service"
     - Example: "refactor: simplify GPU offering validation logic"
   - Show generated message to user (informational, not asking for approval)

2. **Check Current Branch**
   - Run `git branch --show-current` to get the current branch name
   - Check if on `main` or `master` branch

3. **Branch Logic - Execute Appropriate Workflow**

   **WORKFLOW A: If on main/master branch with changes:**
   ```
   1. Run: git stash push -m "Auto-stash before branch creation"
   2. Run: git pull origin main
   3. Generate branch name from commit message (e.g., "feat/add-gpu-filters")
   4. Run: git checkout -b [branch-name]
   5. Run: git stash pop
   6. Run: git add .
   7. Run: git commit -m "[generated message]"
   8. Run: git push -u origin [branch-name]
   9. Create new PR (see step 5)
   ```
   
   **WORKFLOW B: If on feature branch:**
   ```
   1. Run: git add .
   2. Run: git commit -m "[generated message]"
   3. Run: git push (or git push -u origin [branch] if needed)
   4. Check for open PR (see step 4)
   5. Update existing PR OR create new PR (see step 5)
   ```

4. **Check for Open PR** (for WORKFLOW B only)
   - Use GitHub CLI: `gh pr view --json number,title,body,url`
   - If PR exists, go to step 5 (Update PR)
   - If no PR exists, go to step 5 (Create PR)
   - If `gh` CLI not available, provide manual instructions

5. **Handle PR - AUTOMATICALLY**
   
   **If PR EXISTS (WORKFLOW B):**
   - Retrieve current PR title and description
   - Generate enhanced description:
     - Keep existing content
     - Add "## Latest Changes" section with:
       - Commit message
       - Date/time
       - Key files changed
   - Check if title needs improvement based on new changes
   - Update PR using: `gh pr edit [number] --title "[title]" --body "[description]"`
   - Show updated PR URL
   
   **If NO PR EXISTS (WORKFLOW A or B):**
   - Generate PR title from commit message (concise, descriptive, follow npm package sementic-release's standard in title)
   - Generate PR description with:
     - Summary of changes
     - List of modified files (grouped by type: backend/frontend/tests/docs)
     - Purpose/rationale
     - Breaking changes section (if any detected)
     - Testing notes
   - Create PR using: `gh pr create --title "[title]" --body "[description]" --base main`
   - Show new PR URL
   
   **If gh CLI not available:**
   - Provide direct GitHub URL to create PR
   - Provide pre-formatted PR title and description
   - Show instructions for manual PR creation

6. **Exit COMMIT MODE**
   - Show summary:
     ```
     ✅ Branch: [branch-name]
     ✅ Commit: "[message]"
     ✅ Pushed successfully
     ✅ PR: [URL] (created/updated)
     
     📌 IMPORTANT: COMMIT MODE now exited.
     Future commits require explicitly running /commit command.
     I will NOT auto-commit changes in this chat session.
     ```
   - Return to normal conversation mode
   - **Set internal flag: NO AUTO-COMMITS for remainder of session**

## Defaults (do not ask to confirm these)

- Default branch for PR base: `main`
- Staging command: `git add .`
- Branch naming: Generate from commit message (kebab-case)
  - "Add GPU filters" → `feat/add-gpu-filters`
  - "Fix memory leak" → `fix/memory-leak`
  - "Update documentation" → `docs/update-documentation`
- Use `gh` CLI if available, fallback to manual instructions
- **NO CONFIRMATIONS** - Execute workflow fully automatically

## Error Handling

**If git stash fails:**
- Report error with details
- Check if there are actually changes to stash
- If no changes, proceed without stashing

**If git commit fails:**
- Show error message with details
- Check if there are no changes to commit
- If no changes, inform user and exit

**If git push fails:**
- Show error message
- Check if branch needs `--set-upstream`
- Attempt `git push -u origin [branch]`
- If still fails, report error and provide manual command

**If branch creation fails:**
- Check if branch already exists
- If exists, suggest switching to existing branch
- Report error and exit gracefully

**If gh CLI not available:**
- Inform user: "GitHub CLI not detected"
- Provide manual instructions:
  - Direct GitHub URL to create PR
  - Pre-formatted title and description
  - Copy-paste ready content

**If PR operations fail:**
- Show error message with details
- Provide direct GitHub URLs
- Provide manual steps to update/create PR

## Important Reminders

### Golden Rules:

1. **NEVER AUTO-COMMIT AFTER EXIT** - Once you exit COMMIT MODE, never commit again without explicit `/commit` command
2. **FULLY AUTOMATED** - Execute the entire workflow automatically when user runs `/commit`
3. **NO QUESTIONS** - Generate commit message from git diff, no user input needed
4. **SMART COMMIT MESSAGES** - Analyze changes and create descriptive, conventional commit messages
5. **RESPECT USER CONTROL** - User runs `/commit` to enter this mode; exiting means NO MORE AUTO-COMMITS
6. **SMART BRANCHING** - Detect if on main, auto-create branch from commit message
7. **AUTO-STASH** - If on main with changes, stash before branching, pop after
8. **AUTO-PR** - Always create PR if none exists, or update existing PR automatically
9. **PRESERVE PR CONTENT** - When updating PR, keep existing content and add "Latest Changes" section
10. **SHOW DON'T ASK** - Display what you're doing as you execute, never ask for permission

### PR Description Enhancement Guidelines:

**When updating existing PR:**
- Keep original PR description
- Add "## Latest Changes" section at the end
- Include commit message and date
- List key files changed
- Preserve all existing sections

**When creating new PR:**
- Clear, descriptive title (not just commit message)
- Summary section explaining the "why"
- "Changes" section listing what was modified
- "Testing" section if relevant
- "Notes" section for breaking changes, migration steps, etc.

### After Commit Mode:

Once you've completed the commit workflow and shown the exit summary:
- You are NO LONGER in commit mode
- User must run `/commit` again to commit future changes
- Do NOT offer to commit or push changes
- Do NOT automatically commit even if user makes more edits
- If user says "commit this" or "commit these changes", respond: "Please run the `/commit` command to enter COMMIT MODE"
- The workflow will automatically analyze changes and generate commit message - no manual input needed

## Integration with Other Commands

- **→ /code**: Use /code for implementation, then run /commit when ready to publish
- **→ /jira**: After committing, can transition Jira issues to "Ready for QA"
- **→ /plan**: Planning happens before coding and committing

## Output Examples

### Example 1: On Main Branch with Changes
```
📊 Analyzing changes...
   Modified: backend/core/gpu_offering/filters.py (+45, -10)
   Modified: backend/core/gpu_offering/service.py (+23, -5)
   New: tests/test_gpu_filters.py (+78)
   Modified: frontend/src/components/GPUFilterPanel.tsx (+120, -30)

💬 Generated commit message: "feat: add GPU filtering with memory range and availability status"

📊 Detected: On main branch with uncommitted changes
🔄 Executing WORKFLOW A (branch creation)...

✅ Stashed changes
✅ Pulled latest from origin/main
✅ Created branch: feat/add-gpu-filtering-with-memory-range-and-availability-status
✅ Applied stashed changes
✅ Staged all changes
✅ Committed: "feat: add GPU filtering with memory range and availability status"
✅ Pushed to origin/feat/add-gpu-filtering-with-memory-range-and-availability-status
🔍 Creating new PR...
✅ PR created: https://github.com/twlabs/Clustruffle-all/pull/456

📌 IMPORTANT: COMMIT MODE now exited.
Future commits require explicitly running /commit command.
I will NOT auto-commit changes in this chat session.
```

### Example 2: On Feature Branch with Existing PR
```
📊 Analyzing changes...
   Modified: backend/core/gpu_offering/filters.py (+8, -12)
   Modified: tests/test_gpu_filters.py (+15, -3)

💬 Generated commit message: "fix: resolve null pointer exception in GPU memory validation"

📊 Detected: On feature branch (feat/add-gpu-filtering-with-memory-range-and-availability-status)
🔄 Executing WORKFLOW B (commit and push)...

✅ Staged all changes
✅ Committed: "fix: resolve null pointer exception in GPU memory validation"
✅ Pushed to origin/feat/add-gpu-filtering-with-memory-range-and-availability-status
🔍 Checking for open PR...
✅ Found PR #456: "Add GPU Filtering with Memory Range"
📝 Updating PR description with latest changes...
✅ PR updated: https://github.com/twlabs/Clustruffle-all/pull/456

📌 IMPORTANT: COMMIT MODE now exited.
Future commits require explicitly running /commit command.
I will NOT auto-commit changes in this chat session.
```

### Example 3: On Feature Branch without PR
```
📊 Analyzing changes...
   New: tests/unit/test_gpu_offering_filters.py (+156)
   New: tests/integration/test_filter_api.py (+89)
   Modified: backend/core/gpu_offering/filters.py (+12, -8)

💬 Generated commit message: "test: add comprehensive unit and integration tests for GPU filters"

📊 Detected: On feature branch (feat/gpu-filter-testing)
🔄 Executing WORKFLOW B (commit and push)...

✅ Staged all changes
✅ Committed: "test: add comprehensive unit and integration tests for GPU filters"
✅ Pushed to origin/feat/gpu-filter-testing
🔍 No existing PR found
📝 Creating new PR...
✅ PR created: https://github.com/twlabs/Clustruffle-all/pull/457

📌 IMPORTANT: COMMIT MODE now exited.
Future commits require explicitly running /commit command.
I will NOT auto-commit changes in this chat session.
```

## Quick Reference

**User runs:** `/commit`

**You automatically execute:**

1. **Analyze changes** - Run `git diff` and `git status` to understand what changed
2. **Generate commit message** - Create smart, conventional commit message from changes
3. **Detect branch** - Check if on main/master or feature branch

**If on main/master:**
1. Stash changes
2. Pull latest
3. Create branch (from generated commit message)
4. Pop stash
5. Add, commit, push
6. Create PR

**If on feature branch:**
1. Add, commit, push
2. Update existing PR OR create new PR

**After exit:** NEVER auto-commit again. User must run `/commit` to commit again.

---

Remember: **This mode exists specifically to prevent auto-commits.** Once you exit, you must NEVER commit automatically for the rest of the chat session. The user has explicitly configured this workflow to have full control over when commits happen. When user runs `/commit`, execute the ENTIRE workflow automatically - analyze changes, generate commit message, branch if needed, commit, push, and handle PR - all without asking questions.


# AGENTS.md

Guidance for coding agents working in this repository.

## Priorities

1. **Stability over new features**: avoid UI churn unless a bug report requires it.
2. **Do not duplicate state**: update model/data first, then sync UI from model.
3. **Small, reviewable patches**: one issue/fix per commit when possible.
4. **Preserve behavior**: if changing defaults/shortcuts/layout, document and justify.

## PyQt / UI rules

- Keep top-level UI simple; avoid adding new controls to crowded toolbars/menus unless necessary.
- Prefer reusing existing dialogs/actions over creating new windows.
- For keyboard shortcuts:
  - use centralized definitions in `utils/shortcuts.py`,
  - wire through existing apply/update paths,
  - avoid global shortcuts that interfere with typing widgets.
- Any UX-facing change must include:
  - where the behavior is triggered,
  - why it is needed,
  - a quick manual verification path.

## Pipeline / module rules

- Avoid eager imports of heavyweight optional dependencies unless required for execution.
- Keep stop/pause/cancel semantics race-safe (`isRunning()`, cancel flags, resume events).
- Do not silently swallow critical failures; log context (page/module) when continuing softly.

## Testing expectations

- Always run at least:
  - `python -m compileall -f -q <changed_paths>`
  - targeted runtime smoke check for touched flow.
- If pytest is used, ensure files collected as tests are actual pytest tests.
- Never claim tests passed unless command output confirms it.

## Commit / PR hygiene

- Keep commit messages specific and factual.
- PR summary must match actual diffs (no unrelated claims).
- Mention environment limits explicitly when they block verification.

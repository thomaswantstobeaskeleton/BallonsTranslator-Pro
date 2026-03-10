# Handoff Prompt: Continue BallonsTranslator Layout Redesign

**Use this prompt with GPT 5.4 (or similar) to continue the translated text auto-layout / line-breaking redesign.**

---

## Copy-Paste Prompt for GPT 5.4

```
You are continuing work on the BallonsTranslator codebase. A prior session has already implemented a substantial redesign of the translated text auto-layout / line-breaking system. Read docs/PROMPT_LAYOUT_REDESIGN_HANDOFF.md for full context.

Your tasks:
1. Verify the existing changes in utils/line_breaking.py, utils/text_layout.py, ui/scenetext_manager.py work correctly and meet the acceptance criteria (fewer lines, fuller width, no cropping, stable box placement).
2. Test on real manga/manhua pages if possible; tune _score_layout() or other parameters if needed.
3. Check for regressions: no center_text_in_bubble, Constrain text box to bubble works, no recursion in on_document_enlarged/setMaxSize, correct image coordinates.
4. Consider CJK layout: the redesign mainly affects non-CJK (DP path); CJK may need similar treatment.
5. Fix any bugs or rough edges you find. Update docs/config if behavior changes.

Key files: ui/scenetext_manager.py, utils/text_layout.py, utils/line_breaking.py, ui/textitem.py, ui/scene_textlayout.py.
```

### Alternative: Full Prompt (original spec + handoff)

Use the "Original Full Spec" section below, then append:

```
CONTINUATION: A prior session has already implemented changes. Read this document's sections "What Was Already Done" and "Suggested Next Steps". Verify, refine, and extend that work. Do not duplicate completed changes.
```

---

## Original Full Spec (for reference / full prompt)

<details>
<summary>Expand to see the complete original prompt (for pasting into GPT 5.4 if needed)</summary>

You are working in the BallonsTranslator codebase. I want a substantial redesign of the translated text auto-layout / line-breaking system. Do not make a tiny tweak. Rework the current behavior so translated text is formatted naturally inside speech bubbles and does not get cropped.

**High-level goal:** Auto layout currently creates far too many short lines (1–2 words per line). Small words should share lines naturally (3–4+ words when the bubble supports it). Text should reach the left/right sides of the text box instead of forming a narrow vertical column. Text must stop getting cropped at the bottom.

**Important context:** The old `center_text_in_bubble` was removed (caused regressions); do not reintroduce it. The app has `Constrain text box to bubble`. Box must stay attached to bubble. No recursion in `on_document_enlarged`/`setMaxSize`; no Qt6 wrap mode error. Post-translate squeeze is skipped when auto layout ran.

**Desired redesign:** Replace line-break heuristic with width-targeted scoring. Try multiple candidate widths, score layouts (penalize many lines, short last line, raggedness; reward width use), choose best. Prefer wider lines first, then smaller font only if needed. Fix cropping robustly.

**Key files:** ui/scenetext_manager.py, utils/text_layout.py, ui/textitem.py, ui/scene_textlayout.py, utils/line_breaking.py, ui/configpanel.py, utils/config.py, docs/TROUBLESHOOTING.md.

**Non-regressions:** No center_text_in_bubble, Constrain text box to bubble works, correct image coords, no recursion, no post-translate squeeze when auto layout ran.

**Acceptance:** Fewer lines, small words share lines, paragraphs use width, no narrow column, no bottom cropping, stable box placement.

</details>

---

## Context

You are continuing work on the **BallonsTranslator** codebase. A prior session has already implemented a substantial redesign of the translated text auto-layout / line-breaking system. Your job is to:

1. **Verify** the existing changes work correctly and meet the acceptance criteria
2. **Refine** any rough edges or residual issues
3. **Extend** the redesign if needed (e.g., CJK, edge cases, config/docs)
4. **Fix** any regressions or bugs introduced by the prior work

---

## Original Requirements (unchanged)

**High-level goal:** Auto layout should format translated text naturally inside speech bubbles without cropping. Text should use the width of the box (fewer, longer lines) instead of forming narrow vertical columns. Small words (3–4+ per line) should share lines when the bubble supports it.

**Key constraints:**
- Do NOT reintroduce the removed `center_text_in_bubble` behavior
- Do NOT break `Constrain text box to bubble` or image-coordinate correctness
- Do NOT reintroduce recursion in `on_document_enlarged()` / `setMaxSize()` or the Qt6 wrap mode error
- Text boxes must stay attached to bubbles; no top-left / outside-image regressions

**Acceptance criteria:**
- Fewer lines for typical English text in manga/manhua bubbles
- Small words commonly share lines
- Paragraphs visually use box/bubble width
- No narrow vertical stack unless the bubble is actually narrow
- Bottom cropping / clipped last lines fixed or dramatically reduced
- Stable box placement, no cropping

---

## What Was Already Done (prior session)

### 1. `utils/line_breaking.py`
- Added `_LINE_BREAK_PENALTY = 80.0` per line break in the DP
- DP now penalizes each line break so it prefers fewer, longer lines

### 2. `utils/text_layout.py`
- Added `_score_layout()` to score layouts (penalize many lines, short last line, raggedness; reward width use)
- Width-targeted scoring: try multiple candidate widths (100%, 98%, 96%, …), score each, pick the best
- Collision retry shrink softened: 0.98 instead of 0.97
- Fixed alignside retries: copy `words`/`wl_list` before each attempt (they were consumed in place)

### 3. `ui/scenetext_manager.py`
- **Region inset** reduced: 0.92 (round) / 0.95 (elongated) instead of 0.72/0.85
- **DP width** increased: `maxw_px = region_rect[2] * 1.05`
- **ffmt bug fix**: `ffmt = QFontMetricsF(blk_font)` defined before the DP block
- **LAYOUT_HEIGHT_PADDING** increased: 1.06 → 1.12
- **Anti-cropping**: when `constrain_to_bubble` and content would overflow bubble height, scale font down
- **Font refresh**: recreate `ffmt` after font size changes

### 4. `docs/TROUBLESHOOTING.md`
- Updated §9 with layout behavior, cropping fix, and Constrain text box to bubble

---

## Files to Inspect and Possibly Modify

- `ui/scenetext_manager.py` – `layout_textblk()`, region extraction, font scaling, box sizing
- `utils/text_layout.py` – `layout_text()`, `layout_lines_aligncenter()`, `layout_lines_alignside()`, `_score_layout()`
- `utils/line_breaking.py` – `find_optimal_breaks_dp()`, hyphenation
- `ui/textitem.py` – `on_document_enlarged()`, `set_size()` (preserve_topleft)
- `ui/scene_textlayout.py` – Qt document layout, wrap mode
- `ui/mainwindow.py` – post-translate squeeze skip when auto layout ran
- `ui/configpanel.py`, `utils/config.py`, `config/config.example.json`
- `docs/TROUBLESHOOTING.md`

---

## Suggested Next Steps

1. **Test the redesign** on real manga/manhua pages with English translations. Check:
   - Fewer lines, fuller width usage
   - No bottom cropping
   - Boxes stay attached to bubbles
   - No top-left or recursion regressions

2. **Tune scoring** if needed: `_score_layout()` weights (line penalty, orphan penalty, raggedness, width bonus) may need adjustment based on real-world results.

3. **CJK layout**: The redesign mainly affects non-CJK (DP path). CJK still uses `layout_lines_aligncenter` / `layout_lines_alignside`. Consider whether CJK needs similar width-targeted scoring or different handling.

4. **Edge cases** to watch:
   - Very small bubbles (font may scale down aggressively)
   - `ref_src_lines` mode (scoring only used when `forced_lines` is None)
   - Long single words that can’t hyphenate

5. **Config/docs**: Add or adjust settings only if justified. Prefer better defaults over new knobs.

---

## Pipeline Summary (for reference)

1. `layout_textblk()` extracts bubble region, shrinks `region_rect` (inset 0.92/0.95)
2. For non-CJK: DP line breaks with `maxw_px = region_rect[2] * 1.05`, hyphenation for long tokens
3. `layout_text()` receives words, `forced_lines` (from DP), and `max_central_width`
4. When `forced_lines`: positions lines; when not: tries candidate widths, scores, picks best
5. Collision check: mask ratio; retry with 0.98× width if needed
6. Anti-cropping: if `constrain_to_bubble` and content overflows height, scale font down
7. Box sizing: `set_size()`, `setRect()` with `mask_xyxy` offset for image coords

---

## Non-Regression Checklist

- [ ] No `center_text_in_bubble` reintroduced
- [ ] `Constrain text box to bubble` works; box stays in bubble
- [ ] `region_rect` + `mask_xyxy` offset correct for image coords
- [ ] No recursion in `on_document_enlarged` → `setMaxSize`
- [ ] Qt6 wrap mode in `scene_textlayout.py` unchanged
- [ ] Post-translate squeeze skipped when auto layout ran
- [ ] Boxes do not detach, jump, or resize unpredictably

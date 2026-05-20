# Automatic Text Layout Plan

## Phase 1 Audit (Current Pipeline)

### Entry points and flow
1. OCR/detection builds `TextBlock` geometry and region data (`xyxy`, lines, masks, vertical flags). Main model object: `utils/textblock.py`.
2. Translated text is stored in `TextBlock.translation` and rendered through scene text item logic in `ui/scenetext_manager.py`.
3. Layout heuristics and presets live in `utils/auto_text_layout.py` and `utils/text_layout.py`.
4. Final painting and orientation (horizontal/vertical punctuation handling) are in `ui/scene_textlayout.py`.

### Where fitting currently happens
- Font scaling, line-width candidate generation, bubble-constrained placement, and fallback passes happen in `ui/scenetext_manager.py` (notably the auto-layout block around `_layout_textblk` and post-refine paths).
- Shared layout helpers exist in `utils/text_layout.py` and auto-profile/preset tuning in `utils/auto_text_layout.py`.
- Mask-safe rectangle helpers are used through masking/layout utilities (`utils/text_masking.py`, safe rect and contour-based logic).

### Existing strengths
- Already supports configurable min/max font, binary-search-like fit toggles, bubble-center behavior, mask-safe area use, and vertical typography handling.
- Existing UI exposes many auto-layout knobs (`ui/configpanel.py`) and context actions for selected/page relayout.

### Main limitations observed
- Core fit logic is still heavily coupled to scene/UI objects, making batch/API reuse and deterministic testing harder.
- Candidate comparison is spread across multiple heuristics rather than one explicit reusable scoring contract.
- Bubble contour shaping for per-line width in irregular round shapes is partly heuristic and not normalized into a dedicated engine result object.
- Auto-layout metadata requested for migration/control is incomplete at textbox-level persistence.

## Proposed Architecture

Introduce a pure reusable engine module:
- `utils/text_layout_auto_fit.py`

Engine contract:
- Input: text, target box, font bounds, writing mode, language hint, spacing, padding, stroke/shadow padding, optional per-line width profile (bubble contour safe widths).
- Output: selected font size, line breaks, per-line positions, text bbox, recommended box, overflow status, quality score, warnings, rejected candidate diagnostics.

Algorithmic approach:
1. Tokenization aware of CJK vs word tokens.
2. Dynamic-programming line break scoring (raggedness + overflow + orphan penalties).
3. Binary search on font size with hard-fit condition.
4. Final score balancing: overflow, readability, area usage, line balance.

Integration strategy:
- Keep current renderer and manual controls unchanged.
- Gradually route existing auto-layout calls to this engine where inputs are available.
- Preserve fallbacks in `ui/scenetext_manager.py` for legacy edge behavior.

## Migration / Compatibility
- Added per-textbox metadata fields in `FontFormat` with safe defaults:
  - `auto_fit_enabled`, `auto_position_enabled`, `user_adjusted`, `layout_version`, `last_layout_score`, `layout_warnings`.
- Defaults are backward-compatible and only active when layout features consume them.

## Risks
- Heuristic width estimation in pure module is approximate without full font-shaping runtime.
- Vertical CJK and punctuation rules still need deeper renderer-bridged metrics for perfect parity.
- Over-aggressive penalties could reduce readable font size in dense bubbles; needs fixture tuning.

## Test Strategy
1. Add pure-unit tests for engine behavior (fit, overflow, line balancing, width-profile shaping).
2. Keep existing scene-level regression tests for compatibility.
3. Add diagnostic assertions (`warnings`, `rejected_candidates`, `score`) to support layout review/debug.
4. Expand fixture coverage for vertical Japanese/Chinese and mixed Latin+CJK once engine wiring is complete.

## Implemented Incremental Integration (current status)

- Smart auto-fit path now pre-runs the reusable engine (`auto_fit_text`) and feeds its line/font result into the existing renderer-safe flow in `ui/scenetext_manager.py`.
- Context menu + keyboard shortcuts now expose:
  - Re-auto-fit selected
  - Re-auto-fit current page
  - Re-auto-fit all pages
- Bulk re-auto-fit respects manual adjustments by default (`layout_skip_user_adjusted=true`), while selected re-auto-fit can explicitly reprocess user-adjusted boxes.
- Per-textbox metadata now stores layout score/warnings/version to support diagnostics and future review tooling.

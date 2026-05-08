# Koharu-inspired text rendering and workflow controls

## Text rendering / formatting controls

BallonsTranslator-Pro now treats manga lettering controls as first-class project/style state rather than one-off renderer tweaks:

- **Writing mode**: `Auto`, horizontal LTR, vertical RL, and RTL are persisted per textbox/style. Auto mode resolves RTL scripts to RTL and tall CJK boxes to vertical RL.
- **Vertical CJK hints**: vertical rendering uses top-to-bottom columns that flow right-to-left. The renderer utilities normalize repeated punctuation (`?!`, `!!`, `??`), track bracket pairs, and export tate-chu-yoko groups for compact two-character runs such as chapter numbers.
- **Kinsoku line breaking**: CJK wrapping avoids starting lines with closing punctuation, small kana, iteration marks, and prolonged-sound marks, and avoids ending lines with opening punctuation.
- **Fit modes**: shrink, expand, preserve, and balance modes feed renderer diagnostics, layout review actions, and structured/headless exports.
- **Fallback fonts**: per-script fallback chains and per-style overrides are exposed to UI, diagnostics, QA exports, and automation.
- **Manga effects and presets**: stroke, shadow, gradient, opacity, letter spacing, line spacing, padding, alignment, writing mode, fit mode, line-breaking strategy, and manga preset identifiers persist in text styles and can be batched or driven through automation.
- **Diagnostics**: optional renderer overlays and QA exports report resolved writing mode, measured bounds, overflow, missing glyphs, effect-safe margins, recommended box sizes, mask-visible area ratios, and preset suggestions.
- **Mask-safe textbox fitting**: text eraser masks are converted into visible lettering bounds. Rendering QA and layout review can warn when masked edges or holes leave too little room for stroke/shadow, and the canvas context menu provides **Format → Fit box to visible mask area** for selected text boxes.

## Workflow improvements

The Pipeline Insights panel includes a **Workflow preset** control so users can apply or run common manga translation stages without opening nested menus:

- **Full manga pipeline**: detect + OCR + translate + inpaint.
- **Detect + OCR only**: prepare source text and boxes without translation/inpaint.
- **Translate only**: reuse existing boxes/OCR and rerun translation.
- **Inpaint only**: regenerate clean art.
- **Lettering QA pass**: disables AI stages so users can focus on rendering QA, layout review, and export checks.

The same presets are exposed through the local automation API with the `pipeline_presets` route (`list`, `apply`, `run`) for headless and MCP-style clients.

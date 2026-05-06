# Renderer and Text Formatting Controls

BallonsTranslator-Pro now exposes manga-lettering controls in the text format panel, context menu, config panel, layout review agent, and local automation API.

## Per-textbox controls

Use the text/style panel while one or more text boxes are selected:

- **Writing mode**: Auto, Horizontal LTR, Vertical RL, or RTL. Auto uses the translated script plus box geometry: CJK text in a tall box becomes vertical-rl, Arabic/Hebrew becomes RTL, and other text stays horizontal.
- **Fit mode**: Shrink, Expand, Preserve, or Balance lines. Auto-layout and layout review use the selected policy when fitting text to a constrained box.
- **Manga preset**: Default manga bubble, Vertical JP/CN bubble, SFX bold, Caption/narration box, and Small aside text. Presets set font size, stroke, spacing, alignment, writing mode, and padding.
- **Text padding**: Insets text inside the box so strokes, shadows, and vertical punctuation are less likely to clip.

Right-click selected text boxes and use **Format → Writing mode** or **Format → Recenter text in box** for quick lettering fixes.

## Renderer diagnostics

Config → General → **Rendering / Text Formatting** includes:

- default writing and fit modes,
- default render font and manga effect defaults,
- per-script fallback font chains for Latin, CJK, Korean, Arabic/Hebrew, and emoji/symbols,
- overflow warnings,
- an optional diagnostics overlay.

The diagnostics overlay draws the text box, estimated rendered bounds, overflow status, writing mode, and missing-glyph warnings directly on the canvas. It is intended for QA and should be disabled for normal export.

## Layout review integration

The layout review agent sees writing mode, measured bounds, overflow status, style fields, fallback warnings, stroke, padding, and spacing. It can propose and apply:

- shrink-to-fit,
- balance-lines,
- switch writing mode,
- increase padding,
- recenter,
- manga preset application,
- missing-glyph/font mismatch flags.

## Export handoff

**Tools → Export → Export layered PSD handoff...** writes helper layers and editable text metadata to a folder:

- original image,
- inpainted/clean image when available,
- mask when available,
- final composite when renderable,
- JSON manifest with translated text layer geometry and style,
- Photoshop JSX script that rebuilds editable text layers in an open document.

This is a safe handoff format rather than a fake PSD. If native editable PSD writing is unavailable, warnings are written to the manifest and shown in the UI.

## Automation

When the local automation API is enabled, the rendering additions expose:

- `POST /layout_review` for selected/page review and optional apply,
- `POST /render_current_page` to render and save the current page,
- `POST /list_rendering_issues` to list overflow/missing-glyph/writing-mode diagnostics,
- existing structured OCR/page-state endpoints for QA workflows.

## Line-break strategy controls

This pass adds a persistent **Line-break strategy** next to writing mode and fit mode:

- **Auto** keeps existing behavior and chooses script-aware wrapping only where needed.
- **Strict CJK kinsoku** prevents closing punctuation from starting a line/column and prevents opening punctuation from ending one. This is the recommended default for vertical JP/CN bubbles.
- **Balanced lettering** applies the strict rules and additionally avoids one-character dangling final lines where a small rebalance is possible.
- **Loose SFX** relaxes punctuation guards for sound effects where aggressive stylized wrapping is often preferable.

The strategy is saved on each text style, included in renderer diagnostics and `list_rendering_issues`, and can be changed by the layout review agent when vertical CJK text is detected with a weak wrapping policy.

## Font fallback and RTL diagnostics

Each text style can now carry an optional comma-separated **fallback font chain**. Empty means the renderer uses the global per-script chains from Config → General → Rendering / Text Formatting. During formatting, unsupported glyphs are assigned explicit fallback character runs when a configured font can render them, making mixed Latin/CJK/Arabic/emoji lettering less dependent on platform font substitution.

The text panel shows a compact fallback status for the selected text box. If characters still cannot be rendered after the merged fallback chain is considered, the label lists the missing glyphs and the diagnostics/API payloads report the same unresolved characters.

RTL writing mode now also sets the underlying Qt document text direction to right-to-left. This improves Arabic/Hebrew editing and export parity while preserving the existing style alignment controls.

Automation additions:

- `POST /fix_rendering_issues` runs the connected layout-review repair path on the selected scope and returns applied action count plus remaining renderer diagnostics.

## Project typography QA and batch controls

Pipeline Insights now includes **Typography QA Report**. It scans the current page or whole project and produces JSON diagnostics for:

- overflow against measured text bounds,
- unresolved glyphs after fallback chains,
- vertical CJK boxes that are not using strict kinsoku line breaking,
- RTL boxes that are still left-aligned,
- outlined text with too little padding.

The same dialog can apply conservative fixes: shrink overflowing text, switch vertical CJK to strict line breaking, right-align RTL text, add padding for stroked text, and populate empty fallback chains from global per-script defaults.

The **Batch Text Style Override** dialog now also supports manga presets, writing mode, fit mode, line-break strategy, per-style fallback chains, and padding across the current page or the whole project.

Automation additions:

- `POST /export_rendering_qa` returns or writes the same project/page typography QA report.
- `POST /apply_project_rendering_fixes` applies the conservative project-level typography fixes and returns before/after summaries.

### QA preview table and Markdown handoff

The Typography QA dialog now previews report rows before export or fixing. Each row shows page, text-box index, severity, warning codes, suggested fix actions, writing mode, fit mode, line-break strategy, and unresolved glyphs. Reports can be saved as JSON or Markdown (`.md`), and the `export_rendering_qa` automation action uses the same extension-based output behavior.

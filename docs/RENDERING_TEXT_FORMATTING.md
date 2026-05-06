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

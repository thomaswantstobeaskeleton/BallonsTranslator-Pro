# Renderer and Text Formatting Controls

BallonsTranslator-Pro now exposes manga-lettering controls in the text format panel, context menu, config panel, layout review agent, and local automation API.

## Per-textbox controls

Use the text/style panel while one or more text boxes are selected. The top of the panel now states whether you are editing the **Global Font Format / active style defaults** or the **current selected textbox/style override**. The format panel itself is scrollable, so the translation editor below remains usable even when advanced controls are expanded.

The everyday lettering controls are grouped by ownership:

- **Selection layout overrides**: Writing mode, fit mode, line-break strategy, manga preset, and text padding for the current style/textbox.
  - **Writing mode**: Auto, Horizontal LTR, Vertical RL, or RTL. Auto uses the translated script plus box geometry: CJK text in a tall box becomes vertical-rl, Arabic/Hebrew becomes RTL, and other text stays horizontal.
  - **Fit mode**: Shrink, Expand, Preserve, or Balance lines. Auto-layout and layout review use the selected policy when fitting text to a constrained box.
  - **Manga preset**: Default manga bubble, Vertical JP/CN bubble, SFX bold, Caption/narration box, and Small aside text. Presets set font size, stroke, spacing, alignment, writing mode, and padding.
  - **Text padding**: Insets text inside the box so strokes, shadows, and vertical punctuation are less likely to clip.
- **Selection font fallback**: Optional per-style/per-textbox fallback font chain. Empty means use the global per-script fallback fonts from Settings. The **Use global fallback fonts** button clears the override without changing the global defaults.
- **Selection shape/path effects**: Text-on-path, arc span, warp, warp strength, and rounded text-box corners. The **Reset path/warp effects** button returns these decorative effects to neutral values.
- **Selection defaults / reset**: **Use project text defaults** applies Settings → Project text rendering defaults for writing mode, fit mode, line breaks, padding, and clears fallback overrides. It does not change font family, size, colors, or stroke.
- **Live lettering diagnostics**: the panel estimates resolved writing mode, fit size, overflow/fallback status, and a quality score for the selected textbox so likely final-lettering issues are visible before exporting.

Right-click selected text boxes and use **Format → Smart auto fit lettering** for the recommended one-click pass: it balances preserved-space Latin lines, resolves script/writing mode, applies fit sizing, grows the text box if diagnostics still predict clipping, runs a rendered overflow safety pass, and recenters in the detected bubble when a mask is available. **Format → Atomic bubble fit** is stricter for making a phrase behave like one visual block inside a bubble. **Format → Writing mode** and **Format → Recenter text in box** remain available for narrow manual fixes.

## Renderer diagnostics

Config → General → **Rendering / Text Formatting** includes:

- an **Auto lettering preset** that synchronizes constrain-to-bubble, center-in-bubble, overflow, optimal-break, font-size, binary-fit, and balloon-shape defaults so most users do not need to tune the individual numeric controls;
- a live **What the advanced values mean** readout that translates numeric penalties, widths, font ranges, skip distances, and model fields into plain-language labels such as balanced, strict fit, roomy, geometry only, or model-assisted;
- default writing and fit modes,
- default render font and manga effect defaults,
- per-script fallback font chains for Latin, CJK, Korean, Arabic/Hebrew, and emoji/symbols,
- overflow warnings,
- an optional diagnostics overlay.

The diagnostics overlay draws the text box, estimated rendered bounds, overflow status, writing mode, and missing-glyph warnings directly on the canvas. Optional box-size and shape model IDs are now opt-in; leaving them blank uses faster geometry/mask detection, which is the recommended default. It is intended for QA and should be disabled for normal export.

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
- `POST /list_rendering_issues` also returns an `action_summary` map so automation clients can see which fixes are needed before applying them.

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

The Typography QA dialog now previews report rows before export or fixing. Each row shows page, text-box index, severity, quality score, warning codes, suggested fix actions, writing mode, fit mode, line-break strategy, and unresolved glyphs. Warning-type filtering plus **Check visible warnings** and **Clear checks** reduce project-wide fixing clicks. Reports can be saved as JSON or Markdown (`.md`), and the `export_rendering_qa` automation action uses the same extension-based output behavior.

## New in this pass: project-wide style repair and export helpers

The **Batch Text Style Override** workflow now has safer Koharu-style bulk lettering controls for project cleanup requests such as “change Auto font settings across all images”:

- optional **Only update blocks currently using auto font size** scope, so hand-tuned lettering is not overwritten;
- batch stroke width, shadow radius, fit minimum, and fit maximum controls;
- shared backend behavior with the local automation API (`POST /apply_text_style_batch`) for headless style normalization.

Vertical CJK diagnostics now expose per-punctuation offset, rotation, scale, hanging-punctuation, and bracket-pair hints in `vertical_layout_plan()`. These hints make QA/export output more explicit today and provide a stable contract for future custom painting/OpenType vertical-alternate work.

Batch export can also copy available helper images into the export folder:

- `clean/` contains inpainted clean pages when present;
- `masks/` contains mask images when present;
- the export manifest records `include_intermediate` and helper paths.

This reduces the number of manual project-folder lookups needed for lettering review, PSD/GIMP handoff, and archive delivery.

## SVG text handoff

For vector-editor interoperability, **Tools → Export → Export SVG text handoff...** writes the current page as an SVG containing editable `<text>` elements. The SVG records writing mode, font family, size, stroke, alignment, and per-block diagnostics in a companion manifest. Vertical CJK text is emitted with `writing-mode="vertical-rl"`; RTL text is emitted with `direction="rtl"`. This is meant for SVG-capable editors and automation clients, while layered PSD handoff remains the Photoshop-oriented path.

## Layout review resize and punctuation actions

The layout review agent now consumes effect-aware `recommended_box_size` diagnostics. When text cannot fit safely after stroke/shadow/padding are considered, review can propose a targeted **resize to recommended box** action instead of only running generic auto-fit. It also proposes vertical punctuation normalization, RTL right-alignment, low-contrast outline application, and low-quality-score flags so the same conservative fixes are available from UI review, project QA, and automation.


## Latest text-formatting and workflow additions

- **Script-aware uppercase:** automatic uppercase post-processing and the selected-block **To uppercase** action now use `locale_aware_upper()`. Latin/Greek/Cyrillic runs are uppercased, CJK/RTL/emoji runs are preserved, and Turkish/Azeri dotted-i casing is handled without requiring ICU.
- **Ink-safe diagnostics:** fit diagnostics include `ink_clip_risk` when stroke, glow, shadow, or padding leaves too little safe edge margin. Typography QA and layout review can turn that into an **increase padding** action. Diagnostics also include a `preset_suggestion` so SFX, caption, and vertical CJK boxes can be surfaced as one-click preset fixes.
- **Configurable vertical plans:** `vertical_layout_plan()` now records whether latin glyph rotation and punctuation hanging were enabled, and Typography QA uses the live Config values. This makes UI, API, SVG/PSD handoff, and tests agree on vertical punctuation intent.
- **Editor comfort setting:** Config → General → Rendering / Text Formatting includes **Text editor top padding**. It updates the side text editor list immediately, fixing the cramped first-visible-row feel even after scrolling while keeping dense manga editing workflows fast.
- **Automation workflow helper:** local automation clients can call `POST /recent_projects` to retrieve recent project paths, existence checks, and JSON/folder type before opening or running a project.

## 2026-05-07 pass: favorite fonts, visual fit, archive/export workflow

- **Favorite lettering fonts:** Settings → Project text rendering defaults now includes **Favorite lettering fonts**. Enter comma-separated manga/comic font families there, then use the **Favorites** combo in the text formatting panel to apply one instantly to the current style or selected text box. The **★** button captures the currently selected font into the persisted favorites list.
- **Visual-advance fit estimates:** Renderer diagnostics and layout review no longer treat every character as the same width. CJK punctuation, brackets, emoji/symbols, combining marks, and Latin runs use script-aware visual advance units, making overflow warnings and recommended box sizes less noisy for manga punctuation-heavy text.
- **Letter-spacing review fix:** When wide text overflows horizontally, Typography QA / layout review can suggest and apply a conservative **tighten letter spacing** action before shrinking fonts or resizing boxes. This preserves lettering weight where possible.
- **Export workflow polish:** Batch export has a persistent **Open output folder when done** option, and the automation export route can create ZIP/CBZ archives (`kind=archive`, `kind=zip`, or `kind=cbz`) from rendered pages plus optional clean/mask helper images.

## 2026-05-07 pass: reusable manga lettering presets

- **Save current style as a preset:** The text formatting panel now includes **Save preset** next to the manga preset picker. It captures the current font family, weight, size, colors, stroke, shadow, spacing, writing mode, fit mode, line-break strategy, padding, opacity, and fallback chain into a persisted custom preset.
- **Built-in + custom preset parity:** Custom presets appear beside built-in presets and can be applied from the text panel, batch style override, layout review preset actions, and automation. Presets can be imported/exported as JSON packs for team handoff, with missing-font diagnostics when the current machine lacks a preset font family.
- **Recent font recall:** Font choices are remembered as recent lettering fonts and shown alongside favorites, reducing repeated scrolling through long font lists while lettering a chapter.
- **Automation:** Local clients can call `rendering_presets` with `action=list`, `action=save`, `action=save_current`, `action=import`, `action=export`, or `action=delete` to inspect, create, share, or prune reusable rendering presets.

## Mask-safe lettering and overflow diagnostics

When renderer diagnostics are enabled, selected text boxes now show both the measured text bounds and (when a text eraser/mask exists) the mask-safe lettering rectangle. Purple dotted overlay bounds indicate that the current mask leaves a narrow or off-center visible area that may clip strokes, shadows, punctuation, or vertical columns during final render/export.

User-facing repair paths:

- **Canvas context menu → Format → Apply mask-safe padding** increases the selected textbox inset based on the mask-safe insets and persists the result in the textbox `FontFormat.text_padding`.
- **Tools/API Typography QA** reports `mask_safe_area` warnings with mask coverage, safe insets, and recommended padding.
- **Layout Review** can apply `increase_padding` and `recenter` actions for masked/clipped text boxes, so selected-box, whole-page, and headless review flows share the same repair behavior.

Known limitation: this pass avoids fake irregular text flow. The current implementation safely keeps ink away from mask edges; full non-rectangular squeezing around speech-bubble masks remains a later renderer/layout pass.

## Mask-effective fitting and export fallback controls

Mask diagnostics now feed fitting decisions, not just warnings. When a text eraser/mask leaves a narrow visible area, Typography QA computes a `mask_effective_box` from the mask-safe rectangle and can report `mask_safe_overflow` even if the text fits the full rectangular textbox. Project-wide fixes and layout review can then shrink to that visible safe area and/or increase padding. The text formatting panel's live diagnostics also show mask coverage and the effective safe dimensions for the selected textbox.

Batch export has a workflow option for complete manga-reader handoff:

- **Settings → Include unrendered pages in batch export** stores the default.
- **Export all pages as… → Include pages without rendered results** applies it for the current export.
- Local automation can pass `include_unrendered: true` to rendered/archive export calls.

When enabled, export uses the rendered result when available, then falls back to the inpainted/clean image, then the original page. `export_manifest.json` records `source_kind` and `used_fallback_source` for every exported page so downstream scripts can distinguish translated pages from clean/original fallbacks.

## Typography polish and automation pass (2026-05-08)

BallonsTranslator-Pro now has a one-click **Polish typography** workflow for selected text boxes and local automation. It is deliberately conservative and runs before destructive resize operations:

- resolves `Auto` writing mode from script + box geometry, so tall CJK boxes become `vertical_rl` and Arabic/Hebrew becomes `rtl`;
- normalizes vertical CJK punctuation such as `?!`, `!!`, `??`, commas, and ellipses into forms that fit vertical columns better;
- selects a script-aware line-break strategy (`cjk_strict` for CJK/vertical, `balanced` for multi-word Latin captions) when the style is still `auto`;
- rebalances horizontal text with the dynamic-programming kinsoku wrapper where it reduces ragged/dangling lines;
- raises too-small text padding to a safe minimum so strokes and punctuation have breathing room;
- repairs per-textbox fallback font overrides when the configured project fallback chain can cover missing glyphs.

Entry points:

- Canvas context menu: **Format → Polish typography**.
- Local automation API: `POST /polish_typography` with `mode: "page" | "selected"` or an explicit `indices` list.
- Rendering QA/layout review: rows can suggest `polish_typography`; project fixes can apply it before smart fit or resizing.
- Settings: **Project text rendering defaults → Auto-polish new OCR/detected textboxes** controls whether newly loaded run results receive conservative script/line-break/padding/fallback polish.

Related automation/workflow helpers:

- `POST /project_status` returns project/page/textbox counts, current page, completion states, and unsaved state for headless scripts before deciding which workflow action to run.
- `POST /smart_fit_textboxes` remains available for stronger fitting after typography polish.

## Lettering proof packs and vertical layout diagnostics (2026-05-08)

For final lettering review and external editor handoff, BallonsTranslator-Pro can now export a **Lettering proof pack** for the current page:

- Canvas context menu: **Review / QA → Export lettering proof pack**.
- Local API: `POST /export_lettering_proof` or `POST /export` with `kind: "lettering_proof"`.
- The pack contains `typography_qa.json`, `typography_qa.md`, an editable SVG text handoff, a PSD-helper manifest/layer folder when supported, and a `lettering_proof_manifest.json` summary.
- Proof manifests include `proof_metrics` for each text layer: measured bounds, box clearance, overflow pixels, text density, recommended actions, and vertical glyph-cell samples for vertical CJK.

The vertical glyph-cell diagnostics model the intended manga flow (top-to-bottom rows, right-to-left columns) and include punctuation class, rotation, scale, and hanging hints for handoff clients. This is not a full native PSD text writer yet; when PSD helper generation is unavailable, the proof pack records an explicit warning instead of silently pretending the PSD contains editable text.

Automation helpers also expose `GET /health` and `GET /routes` on the local API server, making headless setup easier by listing available commands before a script starts a long workflow.

### Follow-up proof index and route metadata

Lettering proof packs now include `lettering_proof_index.html`, a browser-friendly summary that links the QA JSON/Markdown, editable SVG, PSD-helper manifest, final composite, warnings, and per-textbox proof metrics. This is intended for quick review before sending pages to another editor or before opening the raw JSON.

`GET /health` and `GET /routes` now return a route count and method map (`GET` discovery routes plus available `POST` commands), so automation clients can validate capabilities without hard-coding the server's command list.

## 2026-05-13 lettering workflow additions

- **One-click lettering workflow (current page)** is available from Tools → Project and via the local automation API route `POST /lettering_workflow`. It plans typography polish, smart fit, layout review escalation, proof-pack export, and final render steps from the same rendering QA diagnostics used by the proof reports.
- **Next rendering issue** is available from Tools → Project and via `POST /next_rendering_issue`. It selects the next current-page textbox with overflow, missing glyphs, mask-safe-area, writing-mode, or vertical punctuation warnings.
- `POST /render_current_page` now accepts `write_manifest: true` and writes a sidecar `.render-manifest.json` with page, path, extension, quality, and warning fields for headless export logs.
- Vertical CJK proof diagnostics now include tate-chu-yoko metadata (`orientation: upright_compact`) and avoid single-glyph orphan columns in strict vertical wrapping.

## 2026-05-14 batch lettering and live fix controls

- Tools → Project → **Lettering workflow...** now opens a review dialog instead of immediately applying changes. It supports current page, selected pages from the page list, or the whole project; previews ordered steps; and shows the highest-priority textboxes before running fixes.
- The page list context menu has **Lettering workflow for selected pages...** for fewer-click multi-page typography QA.
- The text formatting panel includes **Apply diagnostics fixes** for the selected textbox. It applies the same conservative typography polish and smart-fit logic used by layout review/API workflows.
- `POST /lettering_workflow` accepts multiple pages and returns `proof_manifests`, `warnings`, and applied action counts. Batch proof packs are supported; full batch rerender remains explicitly warned/deferred.
- Vertical CJK rendering now keeps short ASCII tate-chu-yoko runs upright/compact in the Qt vertical layout, and proof metrics include both estimated and `precise_measured_bounds` values for clipping diagnostics.

## 2026-05-17: Double-outline / back-stroke lettering controls

BallonsTranslator-Pro now supports a second/back outline for manga SFX and high-contrast bubble lettering. The back outline is stored per `FontFormat` as `secondary_stroke_width` plus `secondary_srgb`, is drawn underneath the normal stroke/fill, and participates in fit/overflow diagnostics so the layout reviewer does not underestimate clipping risk.

User-facing controls:

- **Text panel → Back stroke** adjusts the selected textbox/style's back-outline width.
- **Text panel → Back stroke color** sets the back-outline color independently from the normal stroke color.
- **Settings → Rendering/Text Formatting → Default back/second outline width** persists the default width applied by the project text-defaults action.
- **Manga preset → SFX bold** now includes a white back outline by default.
- **Layout review** can propose and apply `apply_double_outline` when SFX-style text lacks a back outline.
- **SVG editable text handoff** exports the back outline as a separate editable text layer before the foreground translated text layer.

Known limitations:

- Native PSD editable text effects are still represented through the handoff manifest/helper assets; a true PSD text/effects writer remains a future pass.
- The Settings panel currently exposes the default back-outline width; per-style and per-textbox color is controlled from the Text panel.

## 2026-05-17: Automation rendering-QA page listing

The local automation API route `list_pages` accepts `include_rendering_qa=true`. When enabled, each returned textbox includes the same rendering QA payload used by the UI (warnings, fit/overflow metrics, writing mode, and style data). This gives headless/batch tools a single-call way to discover pages/textboxes that need fitting, writing-mode repair, missing-glyph fixes, or effect-safe padding before export.

Example request body:

```json
{
  "include_blocks": true,
  "include_rendering_qa": true,
  "reading_order": "auto"
}
```

## 2026-05-17 follow-up: PSD handoff and shortcut safety polish

Layered PSD handoff manifests now carry `secondary_stroke_width` and `secondary_stroke_rgb` for each translated text layer, and the generated Photoshop JSX includes those values in the style notes. The local automation `export` route also accepts multiple pages for `kind=psd_handoff` / `kind=layered_psd`, producing one manifest per valid page and warning when a live final composite is only available for the current page.

The Keyboard Shortcuts dialog now canonicalizes equivalent shortcut strings during conflict detection and shows live warnings for single-key tool/navigation shortcuts. Those single-key shortcuts remain useful for manga editing, but they are intentionally suppressed while typing in text fields to prevent accidental tool/page switches during translation editing.

## 2026-05-17 second follow-up: line quality, RTL fit guard, and export naming

Final-lettering diagnostics now include a `line_break_quality` object in fit diagnostics and proof metrics. It records the wrapped lines, raggedness, widow-line risk, kinsoku violations, and whether `balance_lines` is recommended. The Text panel shows this as a compact `lines balanced/rebalance/ragged` status so editors can spot text that technically fits but still needs manual-quality polish.

Rendering QA and layout review can now propose/apply `balance_lines` for poor wraps even when the box does not overflow. This is intended for manga bubbles where one-character final lines, punctuation at illegal break points, or visibly uneven line lengths reduce lettering quality.

RTL expand-to-fill fitting is capped for Arabic/Hebrew text so Auto/Expand does not exaggerate font size in large boxes. This is a conservative guard until full optional shaping/ligature metrics are introduced.

Batch export now supports a persisted filename template in Settings and the Export dialog. Supported tokens are `{index}`, `{index:03d}`, `{page}`, `{stem}`, `{source}`, and `{ext}`. The automation `export` route also accepts `filename_template`, and the export manifest records the template used. Filenames are sanitized for Windows and path separators before writing.


## 2026-05-17 third follow-up: Atomic bubble fit

Selected textboxes now have an **Atomic bubble fit** action in the canvas context menu. It is designed for the common manga-editing case where text technically fits but does not feel like a single well-composed block inside the speech bubble. The formatter plans padding/inset, balanced line breaks, line spacing, letter spacing, centered alignment, writing mode, and font size together, then applies the result as one undoable edit.

The same formatter is available to automation as `atomic_bubble_fit` and through rendering QA as an `atomic_bubble_fit` suggestion. Rendering settings expose two tuning values: **Atomic bubble fit fill target** and **Atomic bubble fit max font expansion**. Lower fill targets leave more breathing room; lower max expansion prevents short text from becoming too large.

## 2026-05-17 fourth follow-up: Atomic fit profiles and one-off modes

Atomic bubble fit now has density/profile modes instead of a single hard-coded layout personality:

- **Balanced speech** remains the default one-click formatter for normal bubbles.
- **Comfortable / roomy** leaves extra inset and avoids over-enlarging short lines.
- **Dense / compact** fills more of the safe bubble area for crowded dialogue.
- **Caption / narration** favors balanced horizontal blocks with left alignment for rectangular narration boxes.
- **SFX / loud** allows looser wrapping and more expansion for sound-effect lettering.

Settings now persist an **Atomic bubble fit default profile** alongside fill target and max expansion. The canvas context menu also exposes **Atomic bubble fit profile** as a one-off submenu so editors can apply a different density to selected bubbles without changing the global default. Automation callers can pass `profile` to `atomic_bubble_fit`; QA diagnostics include the resolved profile in the `atomic_bubble_fit` payload, so headless review/apply loops can reproduce the same formatting choice.

The profile presets scale the existing fill-target and max-expansion settings instead of replacing them. This keeps older configs compatible while letting users tune the overall aggressiveness once and still switch between roomy, compact, caption, and SFX behavior.

## New in current pass: export-faithful vertical text and portable proof archives

SVG text handoff now uses the same renderer-neutral `vertical_layout_plan()` data consumed by typography QA. For vertical JP/CN text, the exported editable SVG records positioned glyph `<tspan>` entries with column/row coordinates, punctuation classes, rotation hints, and compact `data-tate-chu-yoko="true"` runs for short digit/Latin groups such as chapter numbers and combined punctuation. Horizontal and vertical handoffs also include `font_runs` and `fallback_runs` in the manifest so vector/PSD reconstruction tools can distinguish primary-font text from fallback-font spans.

Layered PSD handoff manifests now include the same `font_runs`, `fallback_runs`, and `vertical_layout_plan` metadata for each translated text layer. Native editable PSD text writing is still deferred, but the manifest/JSX path no longer loses the layout information needed to rebuild vertical or mixed-script text layers faithfully in Photoshop/GIMP-oriented workflows.

Lettering proof packs now write a portable `*_lettering_proof.zip` next to the per-page proof folder. The archive includes the QA JSON/Markdown, editable SVG handoff, PSD helper manifest/JSX, copied helper layers when available, HTML index, and final composite reference when available. This reduces batch/export review friction because a single file can be attached to issue reports or sent to a letterer without manually collecting several subfolders.

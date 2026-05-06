# Koharu Issue Backlog for BallonsTranslator-Pro

_Last refreshed: 2026-05-06 via GitHub REST API across 649 all-state `mayocream/koharu` issues/PRs. This backlog is living input for each Koharu-inspired implementation pass._

## Implemented or advanced in this pass

| Issue | Title | Category | Labels | Maps to BallonsTranslator-Pro | Already implemented here? | Priority | Implementation notes | Deferred reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [#624](https://github.com/mayocream/koharu/pull/624) | Advanced Typesetting (DP Line Breaking, Hyphenation, Kinsoku Shori) | Text fitting / layout / overflow | area: renderer, area: app, area: llm | yes | partial → advanced | P0 | Added reusable kinsoku break-opportunity diagnostics, dynamic-programming balanced wrapping, fit diagnostics with overflow axes/actions, and QA exposure so line-balancing decisions are explainable to UI/API/review flows. | Full DP hyphenation remains deferred because BT-Pro currently uses PyQt text layout rather than Koharu's Rust renderer. |
| [#598](https://github.com/mayocream/koharu/pull/598) / [#597](https://github.com/mayocream/koharu/issues/597) | Manual text direction toggle / render controls | Vertical CJK / RTL / punctuation | area: ui | yes | partial → advanced | P0 | Added vertical-RL layout plans with glyph positions, punctuation classes, center/rotate/hang metadata; Rendering QA now flags horizontal CJK in tall bubbles and can switch to vertical mode in project-wide fixes. | Full OpenType `vert`/`vrt2` glyph substitution remains deferred pending optional shaping dependencies. |
| [#602](https://github.com/mayocream/koharu/issues/602) | Issues with Arabic Text Rendering and Pipeline Export | Vertical CJK / RTL / punctuation | bug, renderer | yes | partial | P0 | RTL diagnostics remain wired through writing-mode resolution and QA; PSD handoff now includes fit/writing/fallback diagnostics per editable text layer. | HarfBuzz Arabic shaping and native editable PSD text serialization are still deferred. |
| [#649](https://github.com/mayocream/koharu/issues/649) | Global font size override from Auto across all images | Text rendering / typography / fonts | ui, renderer, feature request | yes | partial → advanced | P0 | Project Typography QA fixes now apply conservative global rendering fixes through a checked-row preview queue: shrink-to-fit, vertical switch, punctuation normalization, fallback chain, padding, and contrast stroke. | A richer visual before/after approval queue remains next-batch work. |
| [#648](https://github.com/mayocream/koharu/issues/648) | Change basic alignment settings / entire project at once | Text rendering / typography / fonts | none | yes | partial → advanced | P0 | Existing batch styling is complemented by project-wide QA/fix actions and API reports with structured row data. | Still need a dedicated multi-select project style dialog with previews. |
| [#640](https://github.com/mayocream/koharu/issues/640) | Render with fixed font options | Text rendering / typography / fonts | none | yes | partial → advanced | P0 | Fit diagnostics and export/PSD manifests now persist actual fit/style/fallback context for fixed-font handoff and QA. | Native font weight/favorite UX remains deferred. |
| [#630](https://github.com/mayocream/koharu/pull/630) | Centering inconsistencies on text render | Text fitting / layout / overflow | area: app, area: renderer | yes | partial → advanced | P0 | Fit diagnostics now report measured bounds, overflow axes, and review actions; export manifests preserve page status and QA diagnostics for later review. | Full render-after-fix image diff verification remains deferred. |
| [#626](https://github.com/mayocream/koharu/pull/626) | Memory-efficient streaming ZIP export | PSD/export/layers | area: ui, area: tauri | yes | partial | P1 | Batch rendered export now writes `export_manifest.json` with exported/missing pages, completion state, paths, options, and warnings; successful exports are marked Exported and API export can run batch export without dialogs. | Streaming ZIP internals are deferred; current improvement focuses on status/handoff reliability. |
| [#614](https://github.com/mayocream/koharu/pull/614) | Translation data import/export via XML | PSD/export/layers | area: ui, area: app, area: rpc | yes | partial | P2 | PSD handoff and structured API exports now carry richer editable text metadata and renderer diagnostics. | XML import/export itself remains deferred. |
| [#651](https://github.com/mayocream/koharu/pull/651) | Page completion state | Automation/API/headless/MCP | area: ui, rpc, core, tests | yes | yes | P2 | Export manifests include completion state per page; API exports can run status-producing batch handoffs. | No deferral for this pass. |
| [#601](https://github.com/mayocream/koharu/issues/601) | Drag to reorder text boxes | UI/UX/editor workflow | ui, feature request | yes | no | P1 | Relevant to editing workflow. | Deferred because this pass prioritized lettering QA/export/API vertical slices; reorder needs scene/list drag-drop command work. |
| [#610](https://github.com/mayocream/koharu/issues/610) | Bulk process folders of CBZs or image subfolders | Batch/project workflow | none | yes | partial | P1 | Existing batch queue remains; headless export route and export manifests reduce clicks after processing. | Parent/child CBZ project expansion remains deferred. |

## Backlog by category

### Text rendering / typography / fonts

| Issue | Title | Labels | Maps? | Implemented? | Priority | Notes / deferred reason |
| --- | --- | --- | --- | --- | --- | --- |
| [#649](https://github.com/mayocream/koharu/issues/649) | Enable global font size override from Auto across all images | ui, renderer, feature request | yes | partial | P0 | Continue with previewed project-wide style queue and undoable whole-project style transactions. |
| [#648](https://github.com/mayocream/koharu/issues/648) | Change alignment settings / entire project at once | none | yes | partial | P0 | Add dedicated batch style dialog previews. |
| [#640](https://github.com/mayocream/koharu/issues/640) | Fixed font options | none | yes | partial | P0 | Add font favorites/localized names/weight matching. |
| [#595](https://github.com/mayocream/koharu/issues/595) | Font list/display issues | renderer/ui | yes | partial | P1 | Existing fallback UI helps; localized display names deferred. |
| [#77](https://github.com/mayocream/koharu/issues/77) | Custom font support | renderer | yes | partial | P1 | Google font installer exists; robust font package manager deferred. |

### Vertical CJK / RTL / punctuation

| Issue | Title | Labels | Maps? | Implemented? | Priority | Notes / deferred reason |
| --- | --- | --- | --- | --- | --- | --- |
| [#624](https://github.com/mayocream/koharu/pull/624) | Advanced typesetting / kinsoku | area: renderer | yes | partial | P0 | This pass adds explainable break opportunities, dynamic-programming balanced wrapping, and vertical plans; hyphenation deferred. |
| [#598](https://github.com/mayocream/koharu/pull/598) | Manual text direction toggle | area: ui | yes | partial | P0 | Existing controls plus QA/project fixes; next: preview thumbnails. |
| [#602](https://github.com/mayocream/koharu/issues/602) | Arabic rendering/export | bug, renderer | yes | partial | P0 | Need shaping engine experiment. |
| [#213](https://github.com/mayocream/koharu/issues/213) | RTL / vertical edge cases | renderer | yes | partial | P1 | Continue HarfBuzz/ICU evaluation. |

### Text fitting / layout / overflow

| Issue | Title | Labels | Maps? | Implemented? | Priority | Notes / deferred reason |
| --- | --- | --- | --- | --- | --- | --- |
| [#637](https://github.com/mayocream/koharu/pull/637) | Mask-aware collision detection and squeezing | area: renderer, area: ml | yes | partial | P0 | Next major renderer candidate; needs bubble mask geometry. |
| [#630](https://github.com/mayocream/koharu/pull/630) | Text centering inconsistencies | area: app, area: renderer | yes | partial | P0 | Diagnostics/actions exist; next: render verification pass. |
| [#74](https://github.com/mayocream/koharu/issues/74) | Average text size | renderer | yes | partial | P1 | Bounds estimates and fit diagnostics advanced. |

### PSD/export/layers

| Issue | Title | Labels | Maps? | Implemented? | Priority | Notes / deferred reason |
| --- | --- | --- | --- | --- | --- | --- |
| [#626](https://github.com/mayocream/koharu/pull/626) | Streaming ZIP export | io/ui | yes | partial | P1 | Added export manifests and headless export route; streaming archive deferred. |
| [#614](https://github.com/mayocream/koharu/pull/614) | Translation data import/export XML | ui/app/rpc | yes | partial | P2 | Structured OCR/PSD handoff exists; XML deferred. |
| [#587](https://github.com/mayocream/koharu/issues/587) | PSD text layer fidelity | export | yes | partial | P1 | Handoff includes editable metadata; native PSD writing deferred. |
| [#558](https://github.com/mayocream/koharu/issues/558) | Layer/export issues | export | yes | partial | P1 | Helper layers exist; richer mask/vector layers deferred. |

### Automation/API/headless/MCP

| Issue | Title | Labels | Maps? | Implemented? | Priority | Notes / deferred reason |
| --- | --- | --- | --- | --- | --- | --- |
| [#651](https://github.com/mayocream/koharu/pull/651) | Page completion state | rpc/core/ui | yes | yes | P2 | Implemented in earlier pass; export manifest consumes it. |
| [#613](https://github.com/mayocream/koharu/pull/613) | Batch translation with cross-page limits | llm/rpc | yes | partial | P1 | Existing batch queue/API; cross-page token scheduler deferred. |
| [#612](https://github.com/mayocream/koharu/pull/612) | Granular pipeline control and batch process dialog | ui | yes | partial | P1 | Existing controls; API export and manifests improve headless batch handoff. |

### UI/UX/editor workflow

| Issue | Title | Labels | Maps? | Implemented? | Priority | Notes / deferred reason |
| --- | --- | --- | --- | --- | --- | --- |
| [#601](https://github.com/mayocream/koharu/issues/601) | Drag to reorder text boxes | ui, feature request | yes | no | P1 | Deferred; needs text list ordering model and undo commands. |
| [#636](https://github.com/mayocream/koharu/issues/636) | Text box snaps to speech bubble center | none | yes | partial | P1 | Recenter remains explicit; need better move/lock controls. |
| [#646](https://github.com/mayocream/koharu/issues/646) | Eraser tool doesn't work | none | yes | unknown | P2 | Needs reproduction against BT-Pro drawing tools. |

### Performance/runtime/GPU and setup/provider/model

| Issue | Title | Labels | Maps? | Implemented? | Priority | Notes / deferred reason |
| --- | --- | --- | --- | --- | --- | --- |
| [#652](https://github.com/mayocream/koharu/issues/652) | FLUX.2 Klein inpainting fails on macOS Metal | bug, macos | yes | no | P2 | Deferred; platform-specific model/runtime diagnostic. |
| [#638](https://github.com/mayocream/koharu/issues/638) | Shared GPU Memory | help wanted | yes | no | P3 | Deferred; requires runtime/GPU scheduler design. |
| [#625](https://github.com/mayocream/koharu/issues/625) | OCR model recommendation | none | yes | partial | P2 | Existing model diagnostics; better recommendation wizard deferred. |

## Next batch candidates

1. Mask-aware collision detection/squeezing using bubble masks and text bounds (#637).
2. Before/after thumbnails for the checked-row Typography QA fix queue (#649/#648/#630).
3. Native editable PSD text writer or stronger Photoshop/GIMP handoff validation (#587/#558/#602).
4. Text block drag-to-reorder with undo and reading-order export impact (#601).
5. HarfBuzz/ICU shaping experiment for Arabic joining and vertical OpenType alternates (#602/#213/#624).
6. Streaming ZIP/CBZ export and parent/child batch processing (#626/#610).
7. Provider/model setup wizard with model recommendation and failure retry diagnostics (#625/#652).
8. Runtime GPU memory profiles and safer device fallback guidance (#638/#600).

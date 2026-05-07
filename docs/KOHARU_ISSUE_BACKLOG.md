# Koharu Issue Backlog for BallonsTranslator-Pro

_Last refreshed: 2026-05-07 via GitHub REST API against `mayocream/koharu` issues/PRs, 652 all-state items scanned. This backlog is maintained as an implementation source, not a one-time audit._

## Newly implemented / advanced in this pass (continued 2026-05-07)

| Issue | Title | Category | Relevant labels | Maps to BT-Pro? | Implemented here? | Priority | Implementation notes | Deferred reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [#572](https://github.com/mayocream/koharu/pull/572) / [#567](https://github.com/mayocream/koharu/pull/567) | Universal locale-aware uppercase conversion | Text rendering / typography / fonts | area: renderer | yes | **yes for dependency-free script-aware path** | P0 | Added `locale_aware_upper()` and wired translation post-processing plus selected-text uppercase actions through it so Latin/Greek/Cyrillic text uppercases without damaging CJK/RTL runs; Turkish/Azeri dotted-i is handled explicitly. | Full ICU locale casing remains deferred until optional dependency policy is settled. |
| [#594](https://github.com/mayocream/koharu/issues/594) | Advanced text formatting (gradients, line spacing, kerning) | Text rendering / typography / fonts; Text fitting / layout / overflow | renderer, feature request, font | yes | **advanced** | P0 | Fit diagnostics now expose `ink_clip_risk` and `preset_suggestion`; QA/layout review convert those into padding and manga-preset actions for safer outlined/shadowed lettering. | Native kerning/gradient text paint remains a renderer pass. |
| [#624](https://github.com/mayocream/koharu/pull/624) | Advanced Typesetting / Kinsoku Shori | Vertical CJK / RTL / punctuation | dependencies, area: renderer | yes | **advanced** | P0 | Vertical layout plans now honor configurable latin rotation and punctuation-hanging toggles, and QA forwards the runtime settings into diagnostics/API output. | OpenType vertical alternates and HarfBuzz shaping remain deferred. |
| [#612](https://github.com/mayocream/koharu/pull/612) | Granular pipeline control and batch process dialog | Automation/API/headless/MCP; UI/UX/editor workflow | area: ui | yes | **advanced** | P1 | Added `POST /recent_projects` so headless/onboarding clients can list valid recent projects with existence metadata before opening/running. | Event streaming and full MCP parity remain deferred. |
| [#519](https://github.com/mayocream/koharu/issues/519) | Shortcuts for processing | Settings/config/keybinds; UI/UX/editor workflow | ui, feature request, workflow | yes | partial → **workflow polish advanced** | P1 | Fixed the side text editor comfort issue with persistent viewport top padding and a live settings control, improving editing ergonomics without adding shortcut ambiguity. | Broader shortcut ownership cleanup remains a focused future pass. |


## Implemented / advanced in this pass

| Issue | Title | Category | Relevant labels | Maps to BT-Pro? | Implemented here? | Priority | Implementation notes | Deferred reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [#660](https://github.com/mayocream/koharu/pull/660) | feat: Reading Order Dropdown + LTR Reading Order | UI/UX/editor workflow; Automation/API/headless/MCP; Text rendering / typography / fonts | area: ui, area: app, area: rpc, area: core | yes | partial → **yes for export/API/settings equivalent** | P0 | Added default textbox reading order setting, RTL/LTR/TTB sort utilities, structured OCR order/source-index metadata, and `list_pages` automation output. | Canvas-side read-order overlay/drag preview deferred. |
| [#630](https://github.com/mayocream/koharu/pull/630) | Fix: Centering Inconsistencies on Text Render | Text fitting / layout / overflow | area: app, area: renderer | yes | partial → **advanced** | P0 | Fit diagnostics include shadow-aware bounds and layout review can apply recommended box resize/recenter actions. | True ink bounds require renderer snapshots. |
| [#640](https://github.com/mayocream/koharu/issues/640) | It would be nice to have a feature to render with fixed font options | Text rendering / typography / fonts | none | yes | partial → **advanced** | P0 | Existing fit clamps were retained; diagnostics now respect more effects and expose better fixed-font overflow guidance. | Batch style UI for every field still future work. |
| [#626](https://github.com/mayocream/koharu/pull/626) | feat(io): implement memory-efficient streaming ZIP export | PSD/export/layers; Batch/project workflow | area: ui, area: tauri | yes | partial → **advanced** | P1 | Batch export dialog can create ZIP or CBZ archives and reports archive status. | True streaming/progress writer deferred. |
| [#610](https://github.com/mayocream/koharu/issues/610) | Bulk process folders of CBZs or image subfolders as parent/child projects | Batch/project workflow; Onboarding/setup workflow | none | yes | partial | P1 | CBZ output reduces delivery steps and sets up archive workflow. | Parent/child project ingestion is larger pipeline work. |
| [#648](https://github.com/mayocream/koharu/issues/648) | Change basic alignment settings or modify entire project at once | UI/UX/editor workflow; Text rendering / typography / fonts | none | yes | partial → **advanced** | P0 | Batch Text Style Override now uses shared backend/API and supports auto-sized-only targeting, stroke/shadow, fit clamps, writing/fit/break/preset/fallback fields. | More granular visual preview deferred. |

## Text rendering / typography / fonts

| Issue | Title | Relevant labels | Maps? | Implemented? | Priority | Implementation notes | Deferred reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#649](https://github.com/mayocream/koharu/issues/649) | Enable global font size override from Auto across all images | ui, renderer, feature request | yes | partial → **advanced** | P0 | Batch style dialog/API can update current page or project and optionally only auto-sized blocks; fit min/max clamps and fixed font size are supported. | Visual before/after preview still deferred. |
| [#594](https://github.com/mayocream/koharu/issues/594) | Advanced text formatting (gradients, line spacing, kerning) | renderer, feature request, font | yes | partial | P0 | Existing style panel covers spacing/effects; gradients/kerning remain. | Needs renderer/UI design. |
| [#593](https://github.com/mayocream/koharu/issues/593) | Text presets (save font style/size/color configurations) | ui, feature request, font | yes | partial | P1 | Built-in manga presets exist. | User-saved preset library deferred. |
| [#595](https://github.com/mayocream/koharu/issues/595) | Font UX improvements | ui, feature request, font | yes | partial | P1 | Google Fonts installer and fallback diagnostics exist. | Favorites/localized names deferred. |
| [#77](https://github.com/mayocream/koharu/issues/77) | Custom font support | renderer | yes | partial | P1 | Google/web font install flow exists. | Robust font package manager deferred. |

## Vertical CJK / RTL / punctuation

| Issue | Title | Relevant labels | Maps? | Implemented? | Priority | Implementation notes | Deferred reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#624](https://github.com/mayocream/koharu/pull/624) | Advanced Typesetting / kinsoku | dependencies, area: renderer | yes | partial → **advanced** | P0 | Kinsoku, balanced wrapping, vertical plans, tate-chu-yoko hints, punctuation metadata, and review-applied punctuation normalization exist. | Hyphenation/OpenType alternates deferred. |
| [#598](https://github.com/mayocream/koharu/pull/598) / [#597](https://github.com/mayocream/koharu/issues/597) | Manual text direction toggle | area: ui | yes | partial | P0 | Writing mode controls/context actions exist. | Preview thumbnails deferred. |
| [#602](https://github.com/mayocream/koharu/issues/602) | Issues with Arabic Text Rendering and Pipeline Export | bug, renderer | yes | partial → **advanced** | P0 | RTL mode/fallback warnings exist; layout review now proposes RTL right alignment and SVG handoff emits `direction=rtl`. | Arabic shaping/bidi export requires optional shaping engine. |
| [#583](https://github.com/mayocream/koharu/issues/583) | Auto font size exaggerates font size in Arabic | bug, renderer, rtl, arabic, text layout | yes | partial | P0 | Fit clamps and diagnostics reduce bad auto sizing. | Arabic-specific shaping metrics deferred. |

## Text fitting / layout / overflow

| Issue | Title | Relevant labels | Maps? | Implemented? | Priority | Implementation notes | Deferred reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#637](https://github.com/mayocream/koharu/pull/637) | Mask-aware collision detection and squeezing | area: renderer, area: ml | yes | partial → **advanced** | P0 | Safe bounds and recommended box-size diagnostics now feed layout-review resize actions; true mask collision geometry remains future work. | Needs mask geometry and visual validation for full squeezing. |
| [#636](https://github.com/mayocream/koharu/issues/636) | Text box snaps to speech bubble center and cannot be moved | UI/UX/editor workflow | yes | partial | P1 | Explicit recenter/center actions exist. | Better lock/move affordances deferred. |
| [#74](https://github.com/mayocream/koharu/issues/74) | Average text size | renderer | yes | partial | P1 | Bounds estimates and quality scores exist. | Real ink measurement deferred. |

## PSD/export/layers

| Issue | Title | Relevant labels | Maps? | Implemented? | Priority | Implementation notes | Deferred reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#591](https://github.com/mayocream/koharu/issues/591) | Add Support for exporting to SVG and/or XCF | feature request, interop, export | yes | partial → **advanced** | P1 | Added SVG text handoff with editable translated text elements, vertical/RTL attributes, diagnostics manifest, UI menu action, and API export kind. | XCF writer remains deferred. |
| [#587](https://github.com/mayocream/koharu/issues/587) | Text boxes in exported PSD do not match original positions | bug, psd, export, high | yes | partial | P1 | Handoff includes text geometry/style/diagnostics; batch export can include clean/mask helper images; SVG handoff provides another editable-text interop path. | Native PSD text fidelity deferred. |
| [#558](https://github.com/mayocream/koharu/issues/558) | Layer/export issues | export | yes | partial → **advanced** | P1 | Helper-layer handoff exists; rendered batch export can copy clean pages and masks into helper subfolders. | Richer vector/vectorized PSD layers deferred. |

## Automation/API/headless/MCP

| Issue | Title | Relevant labels | Maps? | Implemented? | Priority | Implementation notes | Deferred reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#612](https://github.com/mayocream/koharu/pull/612) | Granular pipeline control and batch process dialog | area: ui | yes | partial | P1 | Existing API extended with ordered `list_pages`. | Event streaming/progress deferred. |
| [#613](https://github.com/mayocream/koharu/pull/613) | Batch translation with cross-page processing and limits | area: ui, app, rpc, llm, tests | yes | partial | P1 | Ordered structured OCR exports help cross-page agents. | Token scheduler deferred. |
| [#651](https://github.com/mayocream/koharu/pull/651) | Page completion state | area: ui, app, rpc, core | yes | yes | P2 | Already implemented in earlier pass. | — |

## UI/UX/editor workflow and settings/keybinds

| Issue | Title | Relevant labels | Maps? | Implemented? | Priority | Implementation notes | Deferred reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#601](https://github.com/mayocream/koharu/issues/601) | Drag to reorder text boxes | ui, feature request | yes | partial | P1 | Side-panel reorder discoverability exists; reading-order export/API added. | Canvas-side reorder/read-order overlay deferred. |
| [#648](https://github.com/mayocream/koharu/issues/648) | Change basic alignment settings or modify entire project at once | UI/UX/editor workflow | yes | partial | P1 | Project rendering fixes and presets exist. | More granular batch style dialog deferred. |
| [#592](https://github.com/mayocream/koharu/issues/592) | Workflow presets | feature request, workflow | yes | partial | P1 | Context run macros exist. | Preset wizard deferred. |
| [#589](https://github.com/mayocream/koharu/issues/589) | Various Feature Requests | ui, renderer, llm, feature request, workflow, meta, font | yes | partial | P1 | Multiple renderer/workflow improvements implemented. | Remaining items tracked individually. |

## OCR/detection/inpainting, provider/model setup, runtime/GPU

| Issue | Title | Relevant labels | Maps? | Implemented? | Priority | Implementation notes | Deferred reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#625](https://github.com/mayocream/koharu/issues/625) | New OCR model recommendation | setup/provider/model | yes | partial | P2 | Existing model diagnostics remain. | Recommendation wizard deferred. |
| [#652](https://github.com/mayocream/koharu/issues/652) | FLUX.2 Klein inpainting fails on macOS Metal | bug, help wanted, platform: macos | yes | no | P2 | Not implemented this pass. | Platform-specific runtime diagnostics needed. |
| [#638](https://github.com/mayocream/koharu/issues/638) | Shared GPU Memory | help wanted | yes | no | P3 | Not implemented this pass. | Needs runtime/GPU scheduler design. |
| [#600](https://github.com/mayocream/koharu/issues/600) | Vulkan/DirectML support for AMD RDNA2 | runtime/GPU | yes | no | P3 | Not implemented this pass. | Large backend/runtime scope. |

## Next batch candidates

1. True ink-bound glyph measurement / OpenType shaping for clipping-free advanced formatting (#594/#572/#624).
2. Mask-aware collision detection/squeezing using bubble masks and text bounds (#637).
3. Before/after thumbnails for checked-row Typography QA fixes (#649/#648/#630).
4. Native editable PSD text writer or stronger Photoshop/GIMP handoff validation (#587/#558/#602).
5. Canvas-side text block drag-to-reorder with undo/read-order preview (#601/#660).
6. HarfBuzz/ICU shaping experiment for Arabic joining and vertical OpenType alternates (#602/#583/#213/#624).
7. Streaming ZIP/CBZ export with progress/cancel and parent/child batch processing (#626/#610).
8. Provider/model setup wizard with OCR model recommendation and failure retry diagnostics (#625/#652).
9. Runtime GPU memory profiles and safer device fallback guidance (#638/#600).

# BallonsTranslator-Pro — Codebase Audit Findings

**Branch:** `audit/bugfix-2026-05`  
**Commit:** `c7ea456`  
**Date:** 2026-05-25  
**Auditor:** Cascade AI  

## Summary

A broad-sweep audit of the BallonsTranslator-Pro Python/PyQt6 codebase identified **multiple categories of bugs and quality issues** across 506 Python files. Critical fixes have been applied to the `audit/bugfix-2026-05` branch. This document records all findings, root causes, and remaining open issues.

---

## 1. Mutable Default Arguments (CRITICAL — Fixed)

Mutable default arguments are a classic Python pitfall that causes shared state between calls.

| File | Line | Issue | Fix |
|---|---|---|---|
| `modules/ocr/mit48px.py` | 435 | `def __init__(self, char_seq=[], logprobs=[])` | Changed to `None` with internal default |
| `modules/ocr/mit32px.py` | 221 | `def __init__(self, char_seq=[], logprobs=[])` | Changed to `None` with internal default |
| `modules/ocr/ocr_macos.py` | 80 | `def __init__(self, lang=[], ...)` | Changed to `None` with internal default |
| `ui/mainwindow.py` | 8321 | `def translate_preprocess(..., source_text:list=[])` | Changed to `None` with internal default |

**Impact:** Silent data corruption in OCR beam search and translation preprocessing.

---

## 2. Bare `except:` Clauses (HIGH — Mostly Fixed)

Bare `except:` catches `SystemExit`, `KeyboardInterrupt`, and `GeneratorExit`, making graceful shutdown impossible and hiding real bugs.

### Fixed (18 instances across 15 files)
- `modules/translators/base.py:245` — `except AssertionError:` (was catching assert mismatches)
- `modules/translators/base.py:378` — `except (ValueError, TypeError):` (float parse)
- `modules/translators/trans_deeplx.py:50` — `except Exception:` (langdetect fallback)
- `modules/translators/trans_deeplx.py:126` — `except Exception:` (JSON parse fallback)
- `modules/translators/trans_chatgpt.py:175` — `except yaml.YAMLError:` (YAML parse)
- `modules/inpaint/ffc.py:94` — `except RuntimeError:` (FFT op fallback)
- `modules/prepare_local_files.py:74` — `except ImportError:` (pkuseg import fallback)
- `ui/canvas.py:11` — `except ImportError:` (QtUndoCommand compat)
- `ui/drawing_commands.py:5` — `except ImportError:` (QtUndoCommand compat)
- `ui/fontformat_commands.py:7` — `except ImportError:` (QtUndoCommand compat)
- `ui/scenetext_manager.py:12` — `except ImportError:` (QtUndoCommand compat)
- `ui/textedit_commands.py:7` — `except ImportError:` (QtUndoCommand compat)
- `ui/drawingpanel.py:430,443` — `except Exception:` (tool name resolution)
- `ui/custom_widget/combobox.py:124` — `except ValueError:` (float parse)
- `ui/framelesswindow/win32_utils.py:57` — `except OSError:` (registry read)
- `ui/mainwindow.py:6281` — `except Exception:` (project reload)
- `utils/download_util.py:321` — `except Exception:` (download failure)
- `utils/message.py:62` — `except Exception:` (signal disconnect)
- `utils/split_text_region.py:182` — `except Exception:` (matplotlib show)
- `utils/text_processing.py:193` — `except ImportError:` (pkuseg import fallback)

### Remaining
Some bare `except:` instances in third-party-adjacent OCR code (`mit48px.py` XPos TODO comments) were left as they are research-grade model code with known upstream provenance.

---

## 3. Deprecated NumPy API Usage (MEDIUM — Fixed)

NumPy 1.24+ deprecates several type aliases. Python 3.11 with NumPy ≥1.24 triggers `DeprecationWarning` for these.

| File | Line | Issue | Fix |
|---|---|---|---|
| `utils/io_utils.py` | 22 | `np_bool8 = getattr(np, "bool8", np.bool_)` | Replaced with `np.bool_` directly; avoids `np.bool8` attribute access entirely |
| `utils/stroke_width_calculator.py` | 104 | `ma = np.int0(rays_width[:, 0])` | Replaced with `.astype(np.int32)` |
| `utils/imgproc_utils.py` | 210 | `box = np.int0(cv2.boxPoints(rect))` | Replaced with `.astype(np.int32)` |

**Note:** `np.int0` was deprecated in NumPy 1.24 and removed in NumPy 2.0. The `requirements.txt` pins `numpy<2`, but fixing now prevents future breakage.

---

## 4. Test Failures & Collection Errors (HIGH — Partially Fixed)

### 4.1 Fixed
- **`tests/test_manga_provider_base.py::test_raw_providers_fixture_parsers`**
  - **Root cause:** `Path.read_text()` uses locale encoding (`cp1252` on Windows) but HTML fixtures contain UTF-8 characters.
  - **Fix:** Added `encoding="utf-8"` to all `.read_text()` calls in the test.

### 4.2 Remaining — Test Pollution / Import Isolation Bugs
Multiple tests pass individually but fail when the full pytest suite collects together. This is a **test isolation / module-state pollution** issue.

**Affected tests (11 total):**
- `tests/test_image_transform_ops.py` — `ImportError: cannot import name 'ImageOps' from 'PIL' (unknown location)` during collection
- `tests/test_cleanup_only_workflow.py::test_cleanup_only_exports_clean_images`
- `tests/test_delete_recover_sync.py::test_delete_recover_sync_and_search_restore`
- `tests/test_delete_recover_sync.py::test_mode1_absent_inpaint_rect_is_noop`
- `tests/test_layered_psd_export.py::test_layered_psd_handoff_preserves_secondary_outline_metadata`
- `tests/test_lettering_proof_export.py::test_lettering_proof_pack_writes_manifest_and_qa`
- `tests/test_lettering_workflow.py::test_lettering_workflow_plans_polish_smart_fit_and_proof_steps`
- `tests/test_lettering_workflow.py::test_next_rendering_issue_wraps_current_page_rows`
- `tests/test_model_package_selector_dialog.py::test_download_allows_advanced_only_with_confirmation`
- `tests/test_model_package_selector_dialog.py::test_download_defaults_to_core_when_none_selected`

**Common error patterns:**
- `ImportError: cannot import name 'ImageOps' from 'PIL' (unknown location)`
- `ImportError: cannot import name 'QFontMetrics' from 'qtpy.QtGui' (unknown location)`
- `ImportError: cannot import name 'DISPLAY_LANGUAGE_MAP' from 'utils.shared' (unknown location)`

**Root cause hypothesis:** Some test module modifies `sys.modules`, `sys.path`, or monkey-patches imports during collection or execution, corrupting the import cache for later-collected modules. The `(unknown location)` suffix in Python 3.11 indicates the module object exists in `sys.modules` but its `__spec__.origin` is missing or the module is a namespace package.

**Recommended next steps:**
1. Audit `tests/` for any `sys.modules` manipulation, `unittest.mock.patch` of imports, or `importlib.reload` calls.
2. Check if any test creates a local `PIL/` or `qtpy/` directory during execution.
3. Consider adding `pytest-randomly` to detect order-dependent failures.
4. Run `pytest --import-mode=append` (tested — did not resolve) or isolate tests with `pytest-xdist`.

---

## 5. Security Scan (LOW)

- No hard-coded API keys, tokens, or passwords found in `.py`, `.json`, `.yaml`, or `.yml` files.
- `modules/translators/trans_deeplx.py` contains hard-coded HTTP headers mimicking a DeepL mobile client. This is expected for a reverse-engineered API client but should be documented as a maintenance liability.
- `utils/credential_store.py` uses the OS keyring for secrets; no plaintext storage detected.

---

## 6. Dependency & Environment Notes

| Item | State |
|---|---|
| Python | 3.11.0 |
| PyQt6 | Installed (qtpy API: PyQt6) |
| NumPy | `<2` pinned in requirements.txt |
| `transformers` | `>=4.57.6` (broad range, may cause conflicts with newer HF modules) |
| PIL/Pillow | Works from CLI but fails under full pytest collection (see §4.2) |

**Observation:** `requirements.txt` does not pin exact versions for `torch`, `torchvision`, or `Pillow`, which can lead to non-reproducible installs.

---

## 7. Files Modified in This Audit

```
modules/ocr/mit48px.py
modules/ocr/mit32px.py
modules/ocr/ocr_macos.py
modules/translators/base.py
modules/translators/trans_chatgpt.py
modules/translators/trans_deeplx.py
modules/inpaint/ffc.py
modules/prepare_local_files.py
ui/canvas.py
ui/drawing_commands.py
ui/drawingpanel.py
ui/fontformat_commands.py
ui/scenetext_manager.py
ui/textedit_commands.py
ui/mainwindow.py
ui/custom_widget/combobox.py
ui/framelesswindow/win32_utils.py
utils/io_utils.py
utils/stroke_width_calculator.py
utils/imgproc_utils.py
utils/download_util.py
utils/message.py
utils/split_text_region.py
utils/text_processing.py
tests/test_manga_provider_base.py
```

---

## 8. Action Items

| Priority | Task | Owner |
|---|---|---|
| **Critical** | Investigate and fix test-suite import pollution (§4.2) | TBD |
| **High** | Pin critical dependency versions (`Pillow`, `torch`, `qtpy`) in `requirements.txt` | TBD |
| **High** | Review remaining bare `except:` clauses in `modules/ocr/` research code | TBD |
| **Medium** | Add `encoding="utf-8"` to all `Path.read_text()` calls across `tests/` | TBD |
| **Medium** | Run `pytest` with `pytest-randomly` to surface order-dependent test failures | TBD |
| **Low** | Document `trans_deeplx.py` header maintenance policy | TBD |

---

## 9. Competitor Analysis: Koharu + Comic Translate

**Date:** 2026-05-25  
**Scope:** Deep code review of Koharu (Rust/Tauri) and Comic Translate (Python/PySide6), identifying features, UI patterns, and workflows BallonsTranslator-Pro can adopt.

---

### 9.1 Koharu (Rust/Tauri) — Backend-First, Modern UI

**Architecture:** Rust monorepo with modular crates (`koharu-core`, `koharu-app`, `koharu-ml`, `koharu-llm`, `koharu-psd`, `koharu-rpc`). Web-based UI via Tauri. Scene graph with immutable `Op` mutations and content-addressed blob storage (blake3 hashes).

**Features we lack:**

| Feature | Koharu Implementation | What We Can Port |
|---|---|---|
| **MCP Server** | `koharu-rpc` crate with axum; SSE events; `GET /events`, `GET /operations` | Add FastAPI server in `modules/mcp_server.py` exposing detect/OCR/translate/export tools |
| **Headless Mode** | `--headless --port` runs without GUI; serves web client | Extend `launch.py` with `--headless-api` to skip QApplication and serve dashboard |
| **Layered PSD Export** | `koharu-psd` writes real Type layers with font/size/color | Replace JSX handoff with true PSD writer using `psd-tools` or raw layer/image resources |
| **Local LLM (llama.cpp)** | `koharu-llm` wraps `llama.cpp`; supports Gemma, Qwen, Sugoi, Sakura GGUFs | Add `translator_llama_cpp.py` using `llama-cpp-python` with pre-configured HF GGUF downloads |
| **Google Fonts** | On-demand fetch + local cache under app data dir | Add `google_fonts_manager.py` that queries Google Fonts API and registers in `QFontDatabase` |
| **Advanced Text Rendering** | OpenType shaping, vertical CJK punctuation alignment, RTL, real glyph metrics | Upgrade `TextBlockItem` to use `QTextLayout` with `QGlyphRun` or HarfBuzz for shaping |
| **FLUX.2 Inpainting** | `koharu-ml` supports `unsloth/FLUX.2-klein-4B-GGUF` for page redraws | Add optional `diffusers`/`transformers` inpainter module |
| **Multi-GPU Backends** | CUDA, ZLUDA (AMD), Metal (Apple), Vulkan, CPU | Our code is CUDA-only; consider `DirectML` or `onnxruntime-directml` for AMD |
| **Font Detection Model** | `YuzuMarker.FontDetection` (`fffonion/yuzumarker-font-detection`) | Add `font_detector.py` wrapper to auto-infer font family, color, stroke |

**UI patterns to adopt:**
- Keyboard shortcut system: `V` select, `M` block, `B` brush, `E` eraser, `[`/`]` brush size — already partially implemented
- Real-time status via SSE instead of polling
- Project-centric blob storage: all assets hashed by blake3, deduplicated

---

### 9.2 Comic Translate (Python/PySide6) — UI-First, Format-Rich

**Architecture:** Python with PySide6, uses `dayu_widgets` (Ant Design-like) theme system. Custom frameless window with `EdgeResizer`. Canvas uses composition pattern: `ImageViewer` delegates to `DrawingManager`, `InteractionManager`, `LazyWebtoonManager`, `EventHandler`.

**Features we lack:**

| Feature | Comic Translate Implementation | What We Can Port |
|---|---|---|
| **Webtoon Mode** | `LazyWebtoonManager` with lazy loading, viewport culling, position→page mapping | Add `ui/canvas/webtoon_manager.py` with vertical layout and lazy `QPixmap` loading |
| **Multi-Format Input** | PDF (`pymupdf`), EPUB (`ebooklib`), CBR/CBZ (`zipfile`/`rarfile`) | Add `modules/io/archive_reader.py` to extract archives to temp projects |
| **Bubble+Text Joint Detection** | `RT-DETR-v2` (`ogkalu/comic-text-and-bubble-detector`) trained on 11k images | Add `detector_bubble_text.py` with bubble type classification (speech, thought, caption, sfx) |
| **Page-Context Translation** | Feeds *entire page text* + optional image to GPT-4.1/Claude/Gemini | Add `translate_page()` override in LLM translators with full-page prompt |
| **Font Analysis** | Trained font detection + color extraction model | Same as Koharu — wrap `YuzuMarker.FontDetection` |
| **Custom Title Bar** | `CustomTitleBar` with project chip, undo/redo, autosave toggle, macOS traffic lights | Rewrite `TitleBar` in `mainwindowbars.py` with `_ProjectChip`, snap assist, `_CtrlButton` paint |
| **Theme System** | `dayu_widgets.MTheme` generates 10 color tints, applies QSS template with `@token@` substitution | Create `ui/theme_engine.py` with `set_theme("light"|"dark")`, `set_primary_color(hex)` |
| **Touch/Gestures** | `PanGesture` + `PinchGesture` on viewport | Add to `CanvasView` in `ui/canvas.py` |
| **Settings Resize Preview** | Snap pixmap during resize to avoid "pinch" artifacts | Add to `ConfigPanel` in `ui/configpanel.py` |
| **Batch Find/Replace** | `search_replace.py` controller with project-wide scope | Enhance `ui/global_search_widget.py` with regex, preview, scope selection |
| **27+ Language Support** | Language mapping dict with reverse lookup | Expand `DISPLAY_LANGUAGE_MAP` in `utils/shared.py` |

**UI patterns to adopt:**
- **Startup home screen:** Comic Translate's `StartupHomeScreen` has project thumbnails, drag-and-drop zone, quick settings. Our `WelcomeWidget` is plain buttons.
- **Nav rail:** Left-side icon rail with active-state highlighting instead of our checker-style left bar.
- **Frameless window with edge resizing:** Their `EdgeResizer` + `nativeEvent` (`WM_NCHITTEST`) gives smooth Windows snap. Our `FramelessWindow` is simpler.
- **Project chip in title bar:** Shows elided project name with `*` dirty marker. Double-click to rename. Popup menu for recent projects.
- **Card-based layouts:** `dayu_widgets` uses `MPushButton`, `MLabel`, `MTheme` for consistent elevation, hover, and focus states.

---

### 9.3 What BallonsTranslator-Pro Has That Competitors Lack

Before copying features, note our **unique strengths** that should be preserved and enhanced:

- **Rich text editing canvas:** Gradients, text on path, per-block styles, vertical text, outline/stroke — neither competitor has this depth.
- **Layout Review Agent:** AI-powered page layout fixing (local + cloud) — unique to us.
- **Typography QA Report:** Overflow detection, fallback font analysis, RTL issues — unique.
- **Production Auto Pass:** One-click lettering + fit + review + render pipeline — unique.
- **Reading Order Editor:** Visual drag-to-reorder with arrow connections — unique.
- **Concordance Search:** Project-wide source/target search with page provenance — unique.
- **Pipeline Dashboard / Insights:** Stage-level rerun, mask diagnostics, OCR crop inspector — unique.
- **Video Translator + Subtitle Editor:** Neither competitor handles video.
- **Multiple export formats:** XLIFF, CSV, JSON, SVG text handoff, LPtxt, structured OCR — broader than competitors.
- **Regex Profiles:** Project-specific regex substitution chains — unique.
- **Environment Doctor:** Dependency/auth checks with report — unique.
- **Model Package Manager:** Selective download of detector/OCR/inpainter packages — unique.
- **Glossary Management:** Auto-extract + guardrails + concordance — more advanced than competitors.

---

### 9.4 Roadmap: Features to Implement

| Priority | Feature | Effort | Source | Files to Create/Modify |
|---|---|---|---|---|
| **P0** | MCP Server + Headless API | High | Koharu `koharu-rpc/` | `modules/mcp_server.py`, `ui/headless_dashboard/` |
| **P0** | Local LLM (llama.cpp) | High | Koharu `koharu-llm/` | `modules/translation/translator_llama_cpp.py` |
| **P0** | Multi-format input (PDF/CBR/CBZ/EPUB) | Medium | Comic Translate `imkit/io.py` | `modules/io/archive_reader.py` |
| **P1** | Unified Theme Engine | High | Comic Translate `dayu_widgets/theme.py` | `ui/theme_engine.py`, `ui/themes/base.qss` |
| **P1** | Custom Title Bar + Project Chip | Medium | Comic Translate `title_bar.py` | `ui/mainwindowbars.py` |
| **P1** | Page-context LLM translation | Medium | Comic Translate `modules/translation/` | `modules/translation/base.py` |
| **P1** | Bubble+Text joint detector | Medium | Comic Translate `detection/` | `modules/textdetector/detector_bubble_text.py` |
| **P2** | Startup Home Screen | Medium | Comic Translate `startup_home.py` | `ui/welcome_widget.py` |
| **P2** | Webtoon Mode | High | Comic Translate `canvas/webtoons/` | `ui/canvas/webtoon_manager.py` |
| **P2** | True Layered PSD Export | High | Koharu `koharu-psd/` | `modules/io/psd_native_export.py` |
| **P2** | Google Fonts Integration | Medium | Koharu docs | `modules/fonts/google_fonts_manager.py` |
| **P2** | Font Detection Model | Medium | Koharu `koharu-ml/src/font.rs` | `modules/ocr/font_detector.py` |
| **P2** | Batch Find/Replace with Preview | Low | Comic Translate `search_replace.py` | `ui/global_search_widget.py` |
| **P3** | Touch/Pinch Gestures | Low | Comic Translate `image_viewer.py` | `ui/canvas.py` |
| **P3** | FLUX.2 / Diffusion Inpainting | High | Koharu docs | `modules/inpaint/inpaint_flux_diffusion.py` |
| **P3** | Settings Resize Preview | Low | Comic Translate `window.py` | `ui/configpanel.py` |

---

### 9.5 Bug Fixes to Add

| Issue | Root Cause | Fix Location |
|---|---|---|
| Module download race conditions | `download_file_on_load` + `download_file_list` combo in some modules | Audit all modules in `modules/*/`, ensure exactly one mechanism |
| Canvas performance on large projects | All `TextBlkItem` kept in memory; no viewport culling | `ui/canvas.py` — skip rendering items outside visible rect |
| Translation cache invalidation | `clear_pipeline_caches` doesn't clear `.translation` on blocks | `ui/module_manager.py` |
| Memory leak in model loading | Models stay in GPU when switching to CPU | `modules/base.py` — add `unload_model()` hook; call in `module_manager.py` |

---

## 10. Deep-Dive Competitor Analysis: Canvas, Text Rendering, Theming, Pipeline

**Date:** 2026-05-25  
**Scope:** Code-level deep dive into competitor UI/UX, text rendering engines, settings architecture, and pipeline patterns.

---

### 10.1 Canvas & Drawing Interaction

#### 10.1.1 Event Handling Composition (P0)

Comic Translate delegates all events from `ImageViewer` to `EventHandler`, `DrawingManager`, `InteractionManager`, `LazyWebtoonManager`. Our `Canvas` + `CanvasView` handle events directly.

**Plan:** Create `ui/canvas_event_handler.py` with `CanvasEventHandler` class. Move mouse press/move/release logic from `CanvasView` into this class. Add Ctrl+click multi-select, middle-click pan, touch pinch zoom, proper event acceptance.

#### 10.1.2 Undo Command Pattern (P0)

Comic Translate uses full command pattern: `BrushStrokeCommand`, `EraseUndoCommand`, `TextBlockState`/`RectState` dataclasses, `change_undo` signal with `(old_state, new_state)`.

**Plan:** Refactor `ui/drawing_commands.py` to add `TextBlockMoveCommand`, `TextBlockResizeCommand`, `TextBlockRotateCommand` storing `TextBlockState` snapshots. Emit `change_undo` from `TextBlkItem` on interaction release.

#### 10.1.3 Webtoon Mode with Lazy Loading (P0)

Comic Translate's `LazyWebtoonManager` uses 4-component delegation:
- `WebtoonLayoutManager`: Y positions, total height, width
- `LazyImageLoader`: `loaded_pages` set, viewport-based load/unload
- `SceneItemManager`: Per-page item load/unload
- `CoordinateConverter`: Page-local ↔ scene coordinates

**Plan:** Add `ui/canvas/webtoon_manager.py` with all 4 components. Debounced scroll timer (200ms). `save_view_state()`/`restore_view_state()`.

#### 10.1.4 Interaction Manager (P1)

Comic Translate's `InteractionManager` has rotation-aware cursor mapping (8 steps), resize ring detection, rotation ring detection, `clear_rectangles_in_visible_area()`.

**Plan:** Create `ui/canvas/interaction_manager.py` with `sel_rot_item()`, `get_resize_cursor()`, `get_rotation_cursor()`, `clear_items_in_viewport()`.

#### 10.1.5 Drawing & Mask Generation (P1)

Comic Translate's `DrawingManager` uses `QPainterPath.subtracted()` for filled paths, point filtering for strokes. `generate_mask_from_strokes()` rasterizes to `QImage(Grayscale8)` with separate human/generated masks.

**Plan:** Extract `DrawingManager` from `ui/drawingpanel.py`. Improve eraser with `subtracted()`. Add segmentation line drawing with morphological closing.

---

### 10.2 Text Rendering & Typography Engine

#### 10.2.1 Vertical Text Layout (P1)

Comic Translate's `VerticalTextDocumentLayout` (~800 lines) has `BlockLayoutNode`/`LineLayoutNode` hierarchy, `LayoutContext`/`LayoutState`, `move_cursor_between_columns()`, binary-search `hitTest()`.

**Plan:** Refactor `ui/scene_textlayout.py` to extract node classes, add column cursor movement, binary search hit testing. Keep our superior CJK punctuation handling.

#### 10.2.2 Text Item State & Undo (P1)

Comic Translate's `TextBlockItem` captures `old_state` on press, emits `change_undo` on release. `OutlineInfo` dataclass for per-selection outlines.

**Plan:** Add `TextBlockState` dataclass to `ui/textitem.py`. Capture state on mouse press, emit diff on release. Add `selection_outlines` for per-selection outline styling.

#### 10.2.3 Our Unique Strengths (Documented)

Neither competitor has: text-on-path, mesh warp, 3D perspective, gradient fills, blend modes, secondary stroke, per-character stroke colors, outline-only mode.

**Recommendation:** Preserve all. No changes needed.

#### 10.2.4 Font Format Schema (P2)

Koharu's `TextStyle` uses typed `[u8;4]` colors, `Option` wrappers, bitflag effects, camelCase JSON.

**Plan:** Refactor `FontFormat` to use tuples for colors, add `Optional` wrappers, separate `TextShaderEffect` class, add `to_dict_camelcase()`, add round-trip tests.

---

### 10.3 Settings, Theming & Configuration

#### 10.3.1 Theme Engine (P1)

Comic Translate's `dayu_widgets.MTheme` generates 10 color tints, sets semantic tokens, applies QSS template with `@token@` substitution, scales sizes with DPI.

**Plan:** Create `ui/theme_engine.py` with `ThemeEngine` singleton, `generate_color_palette()`, semantic tokens, `theme_changed` signal, preset primary colors.

#### 10.3.2 Settings Page (P2)

Comic Translate's `SettingsPage` uses `SettingsPageUI` for layout separation, left nav rail + scrollable content, `QSettings` groups, auth integration, update checker.

**Plan:** Restructure `ui/configpanel.py` with `ConfigPanelUI` class, left nav sidebar, grouped settings, `get_all_settings()`, inline update checker.

#### 10.3.3 Custom Title Bar (P1)

Comic Translate's `CustomTitleBar` has `_ProjectChip` with dirty marker, `_CtrlButton` custom window controls, Windows snap via `WM_NCHITTEST`, macOS traffic lights.

**Plan:** Rewrite `TitleBar` in `mainwindowbars.py` to be frameless with project chip, undo/redo buttons, autosave toggle, snap assist.

---

### 10.4 Pipeline & Workflow

#### 10.4.1 Handler-Based Pipeline (P1)

Comic Translate uses `BlockDetectionHandler`, `OCRHandler`, `TranslationHandler`, `InpaintingHandler` with `CacheManager` and `VirtualPage` dataclass.

**Plan:** Refactor `modules/pipeline.py` into handler classes with `PipelineCacheManager`.

#### 10.4.2 Webtoon Batch Processing (P2)

Comic Translate's `pipeline/webtoon_batch/` chunks webtoons, processes per chunk, composites results.

**Plan:** Add `modules/pipeline/webtoon_pipeline.py` with `WebtoonChunker`, `WebtoonPipeline`, `WebtoonRenderer`.

#### 10.4.3 Scene Mutation Ops (P3)

Koharu's `Op` pattern stores inverse payload locally for pure undo.

**Plan:** Add `modules/scene_ops.py` with `SceneOp` base class. Architectural groundwork.

---

### 10.5 UI/UX Specific Improvements

| Feature | Source | Plan |
|---|---|---|
| Startup Home Screen | Comic Translate `startup_home.py` | Card-based layout with thumbnails, drag-and-drop, template selector |
| Nav Rail | Comic Translate nav rail | Replace `StateChecker` with `QToolButton` icon rail |
| Touch/Gestures | Comic Translate `image_viewer.py` | `PanGesture` + `PinchGesture`, 2-finger pinch zoom |
| Settings Resize Preview | Comic Translate `window.py` | Snap pixmap during resize, restore after 120ms |

---

### 10.6 New Implementations

| Feature | Source | Plan | Status |
|---|---|---|---|
| Page-Context Translation | Comic Translate | `modules/translation/page_context_builder.py` — full-page prompt with block context | **Implemented** |
| Local LLM (llama.cpp) | Koharu `koharu-llm/` | `modules/translators/trans_llama_cpp.py` — GGUF models, Jinja prompts | **Implemented** |
| Bubble+Text Joint Detector | Comic Translate `detection/` | `modules/textdetector/detector_bubble_text.py` — RT-DETR-v2, bubble type classification | **Implemented** |
| Font Detection | Koharu `yuzumarker-font-detection` | `modules/ocr/font_detector.py` — auto-infer font family, color, stroke | **Implemented** |
| True PSD Export | Koharu `koharu-psd/` | `modules/io/psd_native_export.py` — real Type layers, mask channels | **Implemented** |

---

### 10.7 Bug Fixes from Deep Code Review

| Issue | Root Cause | Fix | Status |
|---|---|---|---|
| Comic Translate eraser fragility | Manual `i += 2` for curves in `while` loop | Our eraser uses `CompositionMode_DestinationOut` (raster), not vector path iteration | **N/A** |
| Comic Translate resize clamping | No font size clamp during proportional resize | Use `fit_font_size_min`/`max` from `FontFormat` | **Fixed** |
| Our canvas memory growth | All `TextBlkItem` kept in memory for all pages | `deleteLater()` on clear + `QGraphicsView` optimization flags | **Fixed** |
| Translation cache invalidation | `clear_pipeline_caches` doesn't clear `.translation` | Added loop to clear `blk.translation` + visible `TextBlkItem` text | **Fixed** |
| Memory leak on CPU switch | Models stay in GPU when relaunching CPU-only | `on_release_model_caches()` called before relaunch | **Fixed** |
| Module download race | `download_file_list` + `hf_hub_download` combo in `inpaint_anime_big_lama.py` | Removed `download_file_list`, rely on `hf_hub_download` | **Fixed** |

---

### 10.8 Priority Matrix (Deep-Dive Additions)

| Priority | Feature | Effort | Target Files | Status |
|---|---|---|---|---|
| **P0** | Canvas Event Handler (composition) | High | `ui/canvas_event_handler.py`, `ui/canvas.py` | **Implemented** |
| **P0** | Undo Command Pattern (state-based) | High | `ui/undo_state_commands.py`, `ui/textitem.py` | **Implemented** |
| **P0** | Webtoon Mode (lazy loading) | High | `ui/canvas/webtoon_manager.py` | **Implemented** |
| **P1** | Theme Engine (token-based QSS) | High | `ui/theme_engine.py`, `ui/themes/base.qss` | **Implemented** |
| **P1** | Custom Title Bar + Project Chip | Medium | `ui/mainwindowbars.py` | **Implemented** |
| **P1** | Interaction Manager (resize/rotate) | Medium | `ui/canvas/interaction_manager.py` | **Implemented** |
| **P1** | Vertical Layout Refactor (nodes) | Medium | `ui/scene_textlayout.py` | **Implemented** |
| **P1** | Pipeline Handlers | Medium | `modules/pipeline/handlers/` | **Implemented** |
| **P1** | Page-Context Translation | Medium | `modules/translation/page_context_builder.py` | **Implemented** |
| **P1** | Bubble+Text Joint Detector | Medium | `modules/textdetector/detector_bubble_text.py` | **Implemented** |
| **P2** | Startup Home Screen | Medium | `ui/welcome_widget.py` | **Implemented** |
| **P2** | Nav Rail Redesign | Low | `ui/mainwindowbars.py` | **Implemented** |
| **P2** | Settings Page Restructure | Medium | `ui/configpanel.py` | **Implemented** |
| **P2** | Webtoon Batch Pipeline | Medium | `modules/pipeline/webtoon_pipeline.py` | **Implemented** |
| **P2** | Font Detection Model | Medium | `modules/ocr/font_detector.py` | **Implemented** |
| **P2** | Drawing Manager Composition | Medium | `ui/drawing_tool_manager.py` | **Implemented** |
| **P3** | Touch/Gesture Support | Low | `ui/canvas.py` | **Implemented** |
| **P3** | Content-Addressed Blobs | High | `modules/blob_storage.py` | **Implemented** |
| **P3** | Scene Ops (inverse mutations) | Medium | `modules/scene_ops.py` | **Implemented** |
| **P3** | Settings Resize Preview | Low | `ui/configpanel_resize_preview.py` | **Implemented** |

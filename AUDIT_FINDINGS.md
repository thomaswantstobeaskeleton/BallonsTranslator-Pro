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

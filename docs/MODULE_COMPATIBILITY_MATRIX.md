# Module compatibility matrix (canonical)

This is the canonical compatibility matrix for module readiness and environment cost.

## Tier definitions

| Tier | Meaning | Typical use |
|---|---|---|
| **Stable** | Widely used in this project defaults and core workflows. | Best first choice for first-run and production. |
| **Beta** | Works for many users but has narrower validation or more variance. | Use when Stable misses edge cases. |
| **Experimental** | New/stub/rapidly changing behavior. | Testing and niche workflows. |
| **External dependency heavy** | Requires large/complex optional stacks or external repos/APIs. | Advanced users who accept setup overhead. |

## Core first-run presets (prioritized)

These are prioritized on first run because they are Stable in this fork:

| Stage | Default module | Tier |
|---|---|---|
| Text detection | `ctd` | Stable |
| OCR | `manga_ocr` | Stable |
| Inpainting | `aot` | Stable |
| Translation | `google` | Stable |

## Quick compatibility matrix

| Category | Module | Tier | Windows | Linux | macOS | CPU | CUDA |
|---|---|---|---|---|---|---|---|
| Detector | `ctd` | Stable | ✅ | ✅ | ⚠️ | ✅ | ✅ |
| Detector | `paddle_det` | Stable | ✅ | ✅ | ✅ | ✅ | ✅ |
| Detector | `mmocr_det` | External dependency heavy | ⚠️ | ✅ | ⚠️ | ✅ | ✅ |
| Detector | `dptext_detr` | External dependency heavy | ⚠️ | ✅ | ⚠️ | ⚠️ | ✅ |
| OCR | `manga_ocr` | Stable | ✅ | ✅ | ✅ | ✅ | ✅ |
| OCR | `surya_ocr` | Beta | ✅ | ✅ | ✅ | ✅ | ✅ |
| OCR | `paddleocr_vl_hf` | External dependency heavy | ⚠️ | ✅ | ⚠️ | ⚠️ | ✅ |
| OCR | `hunyuan_ocr` | External dependency heavy | ⚠️ | ✅ | ⚠️ | ⚠️ | ✅ |
| Inpaint | `aot` | Stable | ✅ | ✅ | ✅ | ✅ | ✅ |
| Inpaint | `lama_large_512px` | Beta | ✅ | ✅ | ✅ | ✅ | ✅ |
| Inpaint | `mat` | External dependency heavy | ⚠️ | ✅ | ⚠️ | ⚠️ | ✅ |
| Translator | `google` | Stable | ✅ | ✅ | ✅ | ✅ | ✅ |
| Translator | `LLM_API_Translator` | Beta | ✅ | ✅ | ✅ | ✅ | ✅ |
| Translator | `text-generation-webui` | Experimental | ⚠️ | ✅ | ⚠️ | ✅ | ✅ |

Notes:
- `⚠️` means "possible, but usually needs extra setup or has less validation in community reports."
- For full quality ranking by task, see `docs/QUALITY_RANKINGS.md`.
- For module-to-model mapping, see `docs/MODELS_REFERENCE.md`.

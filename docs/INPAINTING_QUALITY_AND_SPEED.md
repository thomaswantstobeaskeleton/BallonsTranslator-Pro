# Inpainting: quality & speed (codebase map + research)

This doc summarizes how BallonsTranslator runs inpainting today, then lists **practical** levers for **quality** and **throughput**, including ideas from papers and deployment guides. Use it for tuning and for future implementation work.

---

## 1. How inpainting is wired (shared behavior)

All registered inpainters live under `modules/inpaint/` and subclass **`InpainterBase`** (`modules/inpaint/base.py`).

### 1.1 Per-block vs full image

- If **`inpaint_by_block`** is true **and** `textblock_list` is not `None`, the base class **loops each `TextBlock`**:
  - Crops with **`enlarge_window(xyxy, ratio=inpaint_enlarge_ratio)`** (default ratio **2.0** unless overridden, e.g. **`lama_large_512px`** exposes it in params).
  - Builds a **per-block mask** from polygons (`_block_mask_polygons`) and optional **`expand_block_inpaint_mask_vertical`** when `inpaint_block_mask_vertical_expand` is on (`utils/config` / `pcfg.module`).
  - Calls **`_inpaint(crop, crop_mask)`** (no block list) for each region, then **feather-blends** the crop back.
- If **`inpaint_full_image`** is enabled in config, the pipeline passes **`textblock_list=None`**, so many models run **one pass** on the full image + global mask (more VRAM, different failure modes).

**Quality:** Per-block gives localized context and avoids whole-page smears; **mask + crop** must cover halos/outlines or gaps remain.  
**Speed:** Per-block = many small forwards vs one large forward; for **video subtitles**, small crops often win.

### 1.2 Mask pipeline (before `_inpaint`)

- Mask is normalized to **binary 0/255**, same size as image.
- Optional **mask fallback** if mask area ≫ union of block rectangles (`base.py`).
- **`mask_dilation_iterations` / `mask_dilation_kernel_size`** on **`lama_large_512px`** (subclass attrs + UI params) dilate the **global** mask in `inpaint()` before the per-block loop.
- **Median / “skip inpaint” fast path:** When `check_need_inpaint` is true, **`extract_ballon_mask`** may fill large flat regions without calling the neural model (faster; can mis-classify complex video subs).

### 1.3 Video pipeline extras

`modules/video_translator.py` builds masks with **extra dilation** and **`video_translator_inpaint_subtitle_mask_expand`** / BGR→RGB toggles. Poor mask coverage here hurts **every** inpainter equally.

---

## 2. Inpainter inventory (files + levers)

| Key | Module | Mechanism | Faster | Better quality |
|-----|--------|-----------|--------|----------------|
| **opencv-tela** / **opencv-telea** | `base.py` | `cv2.INPAINT_NS` / `INPAINT_TELEA` | Lower `inpaint_radius`, `mask_dilate_px`, `inpaint_passes`; smaller images | Higher radius, dilate, 2–3 passes; fix detection mask |
| **patchmatch** | `base.py` + `patch_match.py` | PatchMatch (DLL) | Smaller images; `patch_size` tradeoff (already 7) | Stronger masks; may need larger patch on textured BG |
| **lama_mpe** | `base.py` + `lama.py` | PyTorch LaMa (MPE ckpt), stride-64 resize, square pad | Lower **`inpaint_size`** (256–512 for subs); GPU; **`bf16`** where supported | Higher `inpaint_size` for huge bubbles; per-block + good mask; avoid huge full-image masks |
| **lama_large_512px** | `base.py` + `lama.py` | Larger LaMa arch | Same as lama_mpe + lower `inpaint_size` | **`mask_dilation`**, **`inpaint_enlarge_ratio`**, **`precision bf16`** |
| **lama_onnx** | `inpaint_lama_onnx.py` | ONNXRuntime, fixed **512²** internal, **letterbox** | **CUDA EP**; smaller crops via per-block; batch N/A | Correct mask; optional larger `inpaint_size` only helps downscale-then-letterbox path |
| **lama_manga_onnx** | `inpaint_lama_manga_onnx.py` | ONNX manga LaMa | Same family as lama_onnx | Tuned for line art; may beat general LaMa on manga |
| **aot** | `base.py` + `aot.py` | PyTorch AOT, resize to `inpaint_size`, composite | Lower `inpaint_size` (1024 vs 2048); GPU | Full mask coverage; higher `inpaint_size` reduces resize blur on thin strokes |
| **simple_lama** | `inpaint_simple_lama.py` | External pip, **`inpaint_by_block = False`** | Always full-image → slow on big pages; use only when needed | Often strong LaMa quality at canonical resolution |
| **mat** | `inpaint_mat.py` | Subprocess to MAT repo, **512×512** | Avoid subprocess overhead = use neural built-ins | 512 cap; good for large holes, not fastest |
| **repaint**, **SD2**, **SDXL**, **Kandinsky**, **dreamshaper**, **fluently**, **cuhk_manga**, **flux_fill**, **qwen_image_edit** | respective `inpaint_*.py` | Diffusion / heavy GenAI | **Fewer steps**, smaller `inpaint_size` / `max_size`, CPU offload only if necessary | **More steps**, better prompts, larger resolution (VRAM tradeoff) |

---

## 3. Research & industry themes (online / literature)

### 3.1 LaMa family (your main neural erasers)

- **Resolution vs receptive field:** LaMa is trained on **fixed-ish resolutions**; at very high res the effective context per pixel shrinks. Papers and community notes suggest **multi-scale / coarse-to-fine** refinement for huge images ([LaMa project](https://advimman.github.io/lama-project/), [arXiv:2206.13644](https://arxiv.org/pdf/2206.13644)).
- **Practical inference:** IOPaint and similar UIs document LaMa as a **general eraser**; quality is **mask-sensitive** ([IOPaint LaMa](https://www.iopaint.com/models/erase/lama)).
- **BT alignment:** You already **resize/pad to stride 64** and use **per-block crops** — consistent with “process at model-friendly resolution, then paste back.”

### 3.2 PyTorch inference speedups

- **`torch.compile`:** PyTorch and Hugging Face docs report **~20–40%** inference gains on many CV/diffusion workloads, with **first-run compile cost** and possible **recompiles on shape change**. For LaMa/AOT, **`dynamic=True`** may help variable crop sizes ([torch.compile tutorial](https://docs.pytorch.org/tutorials/intermediate/torch_compile_full_example.html), [Diffusers + torch.compile](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/)).
- **Mixed precision:** You already use **`autocast` + bf16** on CUDA for LaMa paths where enabled — keep **fp32 fallback** on older GPUs without stable bf16.

### 3.3 ONNX Runtime

- **Execution providers:** `lama_onnx` uses **`CUDAExecutionProvider`** when available (`inpaint_lama_onnx.py`). Optional next steps in code: **`SessionOptions.graph_optimization_level`**, **`enable_mem_pattern`**, **`enable_cpu_mem_arena`** (profile per GPU).
- **TensorRT EP:** Can win or lose vs CUDA EP depending on graph and GPU; ONNX Runtime issues note **slower** cases when partitions fall back to CPU ([OR issue #17434](https://github.com/microsoft/onnxruntime/issues/17434)). **Treat as experimental** — benchmark before shipping defaults.
- **FP16 ONNX:** Some models **NaN** in FP16; validate per export ([OR WebGPU inpaint discussion](https://github.com/microsoft/onnxruntime/issues/22983)).

### 3.4 Diffusion inpainters (FLUX, SD, RePaint, …)

- **Steps vs quality:** Standard tradeoff — **`num_inference_steps`** is the main quality knob; use schedulers that support **fewer-step** presets where available ([Diffusers perf](https://huggingface.co/docs/diffusers/optimization/fp16)).
- **torch.compile + Diffusers:** Same blog as above — biggest wins when shapes are stable or **`dynamic=True`** is set correctly.

### 3.5 Classical (Telea / NS / PatchMatch)

- OpenCV inpaint is **local**; **radius** controls how far information propagates — too small → **unfilled halos** (you added tunable radius + mask dilate in `base.py`).
- PatchMatch quality depends on **patch size** and **structure** of the background; large flat regions are easy, periodic textures can show seams.

---

## 4. Prioritized improvements (for this repo)

### A. No architecture change (users can do now)

1. **Tighten masks:** Detector dilations, `mask_dilation` on `lama_large_512px`, video `video_translator_inpaint_subtitle_mask_expand`, OpenCV **`mask_dilate_px`**.
2. **Per-block context:** Increase **`inpaint_enlarge_ratio`** ( **`lama_large_512px`** UI); for **`lama_mpe`** the ratio is currently **fixed 2.0** unless you add the same param to that class.
3. **Speed:** Lower **`inpaint_size`** for subtitle-sized crops; enable **`bf16`** on capable GPUs; for video, keep **per-block** inpaint unless debugging.
4. **Disable misleading fast paths:** If video subs look “partially erased,” try disabling median skip (`check_need_inpaint` behavior) — today **`lama_large_512px`** sets `check_need_inpaint = False`; **`lama_mpe`** inherits **True** from base.

### B. Small code changes (high value)

1. **`lama_mpe`:** **`inpaint_enlarge_ratio`** is exposed (same as `lama_large_512px`). Module config **`inpaint_torch_compile`** enables **`torch.compile`** for **`lama_mpe`**, **`lama_large_512px`**, and **AOT** (CUDA only; first run compiles).
2. **ONNX inpainters (`lama_onnx`, `lama_manga_onnx`):** Session options via **`inpaint_onnx_ort_graph_optimization_level`** (`all` / `extended` / `basic` / `disable`), **`inpaint_onnx_ort_enable_mem_pattern`**, **`inpaint_onnx_ort_enable_cpu_mem_arena`**, **`inpaint_onnx_ort_intra_op_num_threads`** (0 = default). See `onnxruntime_utils.build_inpaint_onnx_session_options`.
3. **Future:** Optional IO binding for fixed 512² LaMa ONNX inputs; benchmark TensorRT EP per GPU.
4. **Batch crops:** If multiple non-overlapping blocks share the same size after pad-to-64, batched inference could improve GPU utilization (non-trivial; validate LaMa/AOT support).

### C. Larger research / features

1. **Two-stage LaMa:** Coarse pass at ~512, upsample + refine band around mask (literature direction for HR inpainting).
2. **Temporal video inpainting:** Reuse previous frame’s background where mask unchanged (optical flow / SSIM gating) — not in BT today; big win for video **speed + stability**.
3. **Replace subprocess MAT** with in-process inference if MAT is still desired.

---

## 5. Quick “if it looks bad / slow” checklist

| Symptom | Check |
|--------|--------|
| Ghost text / halos | Mask too tight → dilate (detector, video expand, `mask_dilation`, OpenCV `mask_dilate_px`) |
| Smudged whole frame | `inpaint_full_image` or huge mask ratio → use per-block; shrink mask |
| Slow LaMa on GPU | Lower `inpaint_size`; smaller pages; try `bf16`; avoid unnecessary full-image |
| Slow diffusion | Reduce `num_inference_steps`, `max_size`; enable compile if stable |
| ONNX slower than expected | Confirm CUDA EP; try graph opts; avoid CPU fallback |
| AOT partial fill | Raise `inpaint_size`; improve mask; compare to LaMa on same mask |

---

## 6. References (external)

- LaMa: [project page](https://advimman.github.io/lama-project/), [paper (arXiv)](https://arxiv.org/pdf/2206.13644)  
- IOPaint (LaMa erase): [iopaint.com](https://www.iopaint.com/models/erase/lama)  
- PyTorch `torch.compile`: [tutorial](https://docs.pytorch.org/tutorials/intermediate/torch_compile_full_example.html)  
- Diffusers performance: [HF docs](https://huggingface.co/docs/diffusers/optimization/fp16), [torch.compile + Diffusers](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/)  
- ONNX Runtime: execution providers & perf discussions on [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime/issues)  

---

*Last updated from a full pass over `modules/inpaint/*.py` and `InpainterBase` in `base.py`.*

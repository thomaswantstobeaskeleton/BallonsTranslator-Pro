# Inpainter options and VRAM usage

If **lama_large_512px** uses too much VRAM, use one of the lighter options below. All are suitable for text removal in manga/comic bubbles unless noted.

---

## Lighter alternatives (less VRAM than lama_large_512px)

| Inpainter | VRAM / device | Notes |
|-----------|----------------|--------|
| **lama_mpe** | Lower (same family, smaller model) | Same LaMa architecture but **9 blocks** instead of 18. Good quality, less VRAM. Uses `data/models/lama_mpe.ckpt` (auto-download). |
| **lama_manga_onnx** | Low (ONNX, CPU or GPU) | LaMa manga ONNX; often uses less VRAM than PyTorch. Set **inpaint_size** to 512 or 768 to save more. Needs `pip install onnxruntime`. Model: `data/models/lama_manga.onnx` (see mayocream/lama-manga-onnx). |
| **lama_onnx** | Lowest neural option | General LaMa ONNX, fixed 512×512. Very light. Model: `data/models/inpainting_lama_2025jan.onnx` (opencv/inpainting_lama). |
| **opencv-tela** / **opencv-telea** | **0 VRAM** (CPU only) | Built-in OpenCV; no model download. Fast, but quality is lower than LaMa. |
| **patchmatch** | **0 VRAM** (CPU only) | PatchMatch algorithm; no neural network. Good for simple fills; may show seams on complex bubbles. |
| **aot** | Moderate | Manga-image-translator AOT inpainter. Different model; may use less VRAM than large LaMa on some setups. |

---

## If you keep lama_large_512px

- Lower **inpaint_size** (e.g. 512 or 768) to reduce VRAM; description says: *"lower saves VRAM"*.
- Use **bf16** precision if your GPU supports it (already the default when supported).

---

## Optional (may have dependency conflicts)

- **simple_lama** – LaMa via `simple-lama-inpainting` pip package. Can be lighter but has Pillow version conflicts; see [OPTIONAL_DEPENDENCIES.md](OPTIONAL_DEPENDENCIES.md).

---

## Summary

**Best first try for less VRAM:** **lama_mpe** (smaller LaMa, no extra deps).  
**Lowest VRAM with good quality:** **lama_manga_onnx** or **lama_onnx** (ONNX).  
**Zero VRAM:** **opencv-tela**, **opencv-telea**, or **patchmatch** (quality trade-off).

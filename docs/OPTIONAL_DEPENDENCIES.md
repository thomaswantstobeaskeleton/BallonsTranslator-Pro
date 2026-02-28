# Optional module dependencies and conflicts

Some optional detectors and inpainters have dependency version constraints that conflict with the main `requirements.txt`. This document explains the conflicts and workarounds.

---

## 1. CRAFT detector (`craft_det`)

**Conflict:** The PyPI package **craft-text-detector** (0.4.3) requires:

- `opencv-python>=3.4.8.29,<4.5.4.62`

The project’s main requirements use **opencv-python>=4.8.1.78** (for compatibility with other components). So with the default install, **craft-text-detector** is incompatible.

**What happens:**  
If you install `craft-text-detector` in the same environment, pip may report a dependency conflict. If you then upgrade opencv to satisfy the main requirements, **craft_det** may fail at runtime (import or inference).

**Workarounds:**

1. **Don’t use `craft_det`**  
   Use **easyocr_det** (includes CRAFT) or **mmocr_det** (e.g. TextSnake) instead.

2. **Separate environment for CRAFT**  
   In a dedicated venv, install an older opencv and craft-text-detector:
   ```bash
   pip install "opencv-python>=4.2,<4.5.4.62" craft-text-detector torch
   ```
   Run only the CRAFT-related workflow in this env (or accept running the full app there with older opencv).

3. **Keep main env as-is**  
   Do not install `craft-text-detector`. The **craft_det** option will appear in the UI but will show an error when selected; use another detector.

---

## 2. Simple LaMa inpainter (`simple_lama`)

**Conflict:** The PyPI package **simple-lama-inpainting** (0.1.2) requires:

- `pillow>=9.5.0,<10.0.0`

The project typically uses **Pillow 10.x** (e.g. 10.4.0). So with the default install, **simple-lama-inpainting** is incompatible.

**What happens:**  
Pip may report a dependency conflict. With Pillow 10.x installed, **simple_lama** may fail at import or runtime.

**Workarounds:**

1. **Don’t use `simple_lama`**  
   Use **lama_large_512px**, **lama_mpe**, **lama_onnx**, or **lama_manga_onnx** instead.

2. **Downgrade Pillow (only if you need simple_lama)**  
   ```bash
   pip install "pillow>=9.5,<10"
   ```
   This may affect other features that expect Pillow 10.x.

3. **Separate environment**  
   Use a dedicated venv with `pillow>=9.5,<10` and `simple-lama-inpainting` for the simple_lama inpainter only.

---

## Summary

| Optional module   | Conflicting package        | Conflict                          | Prefer instead                          |
|-------------------|----------------------------|-----------------------------------|-----------------------------------------|
| **craft_det**     | craft-text-detector        | opencv-python &lt;4.5.4.62 vs ≥4.8 | easyocr_det, mmocr_det, ctd             |
| **simple_lama**   | simple-lama-inpainting     | pillow &lt;10 vs 10.x               | lama_large_512px, lama_onnx, lama_manga_onnx |

The main application and all other detectors/inpainters work with the versions in `requirements.txt`. These notes only apply if you explicitly want **craft_det** or **simple_lama**.

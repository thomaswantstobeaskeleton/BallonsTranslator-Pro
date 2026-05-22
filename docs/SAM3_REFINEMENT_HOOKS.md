# SAM 3 refinement hooks

This document describes the first SAM 3 integration layer for BallonsTranslator-Pro.

The goal is **not** to replace CTD, YOLO, Paddle, EasyOCR, or other production detectors. The intended pipeline is:

1. Run an existing detector first, such as `ctd`, `hf_object_det`, `ysgyolo`, or `paddle_det`.
2. Optionally run SAM 3 with prompts such as `speech bubble` or `text`.
3. Filter SAM 3 candidates against the original detector mask.
4. Merge the selected SAM 3 candidates with the detector mask.
5. Expose the refined mask and safe-area metadata to inpainting, auto-layout, cleanup-only mode, and live translation high-quality mode.

## Added utility module

`utils/sam3_refinement.py` is dependency-safe. It does not import `transformers`, `torch`, or SAM 3 directly. Instead, callers provide a `mask_runner(image, prompt)` or `box_runner(image, prompt)` callback.

This keeps normal installs working even when:

- SAM 3 dependencies are missing.
- the user has not accepted the SAM 3 Hugging Face gated model terms.
- the user is offline.
- the app is running in a lightweight/headless test environment.

## Core helpers

- `SAM3RefinementOptions`
- `SAM3RefinementResult`
- `refine_mask_with_sam3(...)`
- `sam3_safe_areas_from_mask(...)`
- `should_use_sam3_for_live_translation(...)`
- `cleanup_only_mode_plan(...)`

## Bubble mask refiner

Use `refine_mask_with_sam3(...)` after an existing detector has produced an initial mask.

Recommended settings:

```python
from utils.sam3_refinement import SAM3RefinementOptions, refine_mask_with_sam3

opts = SAM3RefinementOptions(
    enabled=True,
    prompt="speech bubble",
    fallback_prompt="text",
    merge_mode="union",
    min_iou_with_initial=0.02,
    max_mask_area_ratio=0.75,
)
result = refine_mask_with_sam3(image, detector_mask, opts, mask_runner=sam3_mask_runner)
refined_mask = result.mask
```

For strict cleanup, use `merge_mode="intersect"` to avoid expanding the mask too far. For whole-bubble cleanup, use `merge_mode="replace_if_nonempty"` or `union`.

## Auto-layout safe-area detector

`sam3_safe_areas_from_mask(...)` turns a refined SAM/bubble mask into conservative safe-area metadata. The current implementation returns a stable merged safe rectangle for the contour. Future UI/API wiring can attach these safe areas per text block.

This is meant to feed the auto-layout engine so text can fit inside a real bubble contour instead of a rectangular detector box.

## Cleanup-only mode

`cleanup_only_mode_plan(...)` returns a serializable plan for a future UI/API preset:

- run detection
- skip OCR
- skip translation
- run inpainting
- skip render/typesetting
- export clean raws

Default preset:

```python
cleanup_only_mode_plan(
    detector="sam_text_det",
    sam_prompt="speech bubble",
    inpainter="lama_manga_onnx",
)
```

## Live translation helper

SAM 3 is too heavy for every live frame. Use `should_use_sam3_for_live_translation(...)` to gate SAM 3 so it only runs in high-quality/slower live profiles.

Examples that enable SAM 3:

- `high_quality_slower`
- `hq_slow`
- `{ "use_sam3": true, "quality_mode": "high_quality" }`

Examples that do not enable SAM 3:

- `fast`
- `low_latency`
- `{ "use_sam3": true, "quality_mode": "low_latency" }`

## Next wiring PRs

Recommended follow-up work:

1. Add config fields under `pcfg.module` for SAM 3 refinement:
   - `sam3_refine_enabled`
   - `sam3_refine_prompt`
   - `sam3_refine_fallback_prompt`
   - `sam3_refine_merge_mode`
   - `sam3_refine_min_iou`
2. Add an optional post-detector hook in the pipeline after CTD/YOLO/Paddle detection.
3. Add UI controls under detection/cleanup settings.
4. Attach `result.safe_areas` metadata to text blocks or page metadata for auto-layout.
5. Add a cleanup-only command/menu action using `cleanup_only_mode_plan(...)`.
6. Add realtime profile wiring so SAM 3 is only used for high-quality/slower live translation.

## Tests

See `tests/test_sam3_refinement.py`.

The tests use synthetic masks and fake runners. They do not require SAM 3, Hugging Face, PyTorch, OpenCV, or GPU access.

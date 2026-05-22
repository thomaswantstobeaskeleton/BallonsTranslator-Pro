# SAM 3 refinement hooks

This document describes the SAM 3 mask-refinement integration for BallonsTranslator-Pro.

The goal is not to replace CTD, YOLO, Paddle, EasyOCR, or other production detectors. The intended pipeline is:

1. Run an existing detector first, such as `ctd`, `hf_object_det`, `ysgyolo`, `paddle_det`, or `paddle_det_v5`.
2. Run SAM 3 with prompts such as `speech bubble` or `text`.
3. Filter SAM 3 candidates against the original detector mask.
4. Merge the selected SAM 3 candidates with the detector mask.
5. Expose the refined mask and safe-area metadata to inpainting, auto-layout, cleanup-only mode, and high-quality live translation profiles.

## UI entry point

The feature is wired as a normal detector module named `sam3_refiner`.

Open Settings -> DL Module -> Text Detection, then set Detector to `sam3_refiner`.

Because this is a normal detector module, it uses the same module selector, parameter editor, config save/load path, and per-module parameter persistence as other detectors.

## Persistence

The selected detector is stored through the normal config field:

- `pcfg.module.textdetector = "sam3_refiner"`

The refiner parameters are stored through the normal detector parameter path:

- `pcfg.module.textdetector_params["sam3_refiner"]`

The selected base detector also keeps its own normal settings, such as `pcfg.module.textdetector_params["ctd"]` or `pcfg.module.textdetector_params["paddle_det"]`.

This means a user can tune CTD, YOLO, or Paddle exactly as before, then select `sam3_refiner` and choose that detector as the first pass.

## Curated model choices

The refinement-model selector intentionally exposes only model choices expected to help SAM-style mask refinement:

- `facebook/sam3`
- `Custom SAM 3 model id`

OCR models, LLMs, inpainters, translation models, and YOLO detectors are intentionally excluded from this selector because they are not SAM-style mask-refinement models.

Use `Custom SAM 3 model id` only for SAM 3-compatible Hugging Face models. If the custom ID is empty, the code falls back to `facebook/sam3`.

## Curated base detectors

The first-pass detector list is curated to detectors that make sense before SAM 3 refinement:

- `ctd`
- `hf_object_det`
- `ysgyolo`
- `paddle_det`
- `paddle_det_v5`
- `easyocr_det`
- `craft_det`

These detectors create useful coarse text or bubble regions that SAM 3 can refine.

## Added utility module

`utils/sam3_refinement.py` is dependency-safe. It does not import `transformers`, `torch`, or SAM 3 directly. Instead, callers provide a mask or box runner callback.

This keeps normal installs working even when SAM 3 dependencies are missing, the user has not accepted gated model access, the user is offline, or tests are running in a lightweight environment.

## Added detector module

`modules/textdetector/detector_sam3_refiner.py` registers `sam3_refiner`.

It behaves like a normal text detector. Internally it:

1. Loads the selected base detector and its normal saved parameters.
2. Runs the base detector on the page.
3. Loads `sam_text_det` with the selected SAM 3 model ID and prompt.
4. Runs SAM 3 as a refinement candidate source.
5. Filters candidates by area and IoU with the base detector mask.
6. Merges masks using `union`, `intersect`, or `replace_if_nonempty`.
7. Attaches `region_inpaint_dict["sam3_refinement"]` metadata to detected blocks when safe-area attachment is enabled.

Current limitation: the first UI-wired implementation uses the existing `sam_text_det` detector output, which exposes SAM detections as boxes/blocks. These are converted into masks for refinement. A later PR can expose raw SAM pixel masks from `sam_text_det` for more precise contour refinement.

## Cleanup-only mode

`cleanup_only_mode_plan(...)` returns a serializable plan for a cleanup-only UI/API preset. The default detector is now `sam3_refiner`, with `speech bubble` as the SAM prompt and `lama_manga_onnx` as the inpainter.

In the current UI, users can approximate this by selecting `sam3_refiner` as the detector, enabling detection and inpaint, and disabling OCR/translation/render paths where available.

## Live translation helper

SAM 3 is too heavy for every live frame. Use `should_use_sam3_for_live_translation(...)` to gate SAM 3 so it only runs in high-quality/slower live profiles.

## Next wiring PRs

Recommended follow-up work:

1. Expose raw SAM pixel masks from `sam_text_det` so `sam3_refiner` can refine with true SAM contours instead of block rectangles.
2. Add auto-layout consumption of `region_inpaint_dict["sam3_refinement"]`.
3. Add a dedicated cleanup-only menu/API action using `cleanup_only_mode_plan(...)`.
4. Add realtime profile wiring so SAM 3 is only used for high-quality/slower live translation.
5. Add a setup-health warning when `sam3_refiner` is selected but SAM 3 dependencies/model access are unavailable.

## Tests

See `tests/test_sam3_refinement.py`.

The tests use synthetic masks and fake runners. They do not require SAM 3, Hugging Face, PyTorch, OpenCV, or GPU access.

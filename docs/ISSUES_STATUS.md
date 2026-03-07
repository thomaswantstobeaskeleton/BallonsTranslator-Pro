# Recent issues status (BallonsTranslator-Pro)

Quick reference for [GitHub issues](https://github.com/thomaswantstobeaskeleton/BallonsTranslator-Pro/issues). Verify on your build; close if resolved.

| # | Title | Status | Notes |
|---|--------|--------|-------|
| **17** | Ctrl+A not working on text blocks | **Fixed** | Canvas: app-level event filter for Ctrl+A → `set_blkitems_selection(True)`. Edit menu: "Select all text boxes". Shortcut `canvas.select_all` (Ctrl+A). |
| **18** | Inpaint/OCR/Translate panels showing when disabled | **Fixed** | `setPipelineVisible()` and `on_enable_module()` set selector visibility from `pcfg.module.enable_detect/ocr/translate/inpaint`. Pipeline menu toggles update visibility. |
| **16** | Translation ComboBox too small | **Fixed** | Translator dropdown `setMinimumWidth(220)`. Pipeline in horizontal `QScrollArea` so it doesn't get squashed. |
| **19** | text-generation-webui AssertionError | **Fixed** | `modules/base.py`: `updateParam()` now skips unknown `param_key` (no assert). Prevents crash when UI sends param not in module's `params`. |
| **15** | Unused models still being downloaded | **Addressed** | Model package selector on first launch (maintainer reply). Users can choose packages; uninstall via Tools → Manage models. |
| **14** | Re-run "Auto fit scales font size" for another font | Open | Feature request: re-run auto-fit for selected block with different font. |
| **13** | Bug | Open | No text; images only — needs reproduction steps. |
| **12** | Spell check - how working? | Open | Documentation / UX. |
| **11** | GoogleTranslate | Open | Unclear; needs details. |
| **10** | OneOcr - each letter alternates with space | Open | OCR module bug. |
| **9** | Delete and Recover removed text not properly working | Open | Delete/recover + mask manual edit flow. |
| **8** | merge font size - not working | **Wired** | CTD detector sets `model.merge_fntsize_tol_hor/ver`; `ctd/inference.py` passes them to `group_output()`. If still not working, may be tolerance value or merge logic in `utils/textblock.py`. |

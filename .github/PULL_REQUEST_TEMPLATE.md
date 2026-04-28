## Summary
- What changed?
- Why is this needed?

## Required Checks

### README parity (EN + ZH)
- [ ] I reviewed [`README.md`](../README.md) and [`README_zh_CN.md`](../README_zh_CN.md) and confirmed they are in parity for any user-facing changes in this PR.
- [ ] If one README was updated without the other, I explained why.

### Startup script verification
- [ ] Verified startup behavior and entry points as applicable:
  - [ ] [`launch.py`](../launch.py)
  - [ ] [`launch_win.bat`](../launch_win.bat)
  - [ ] [`launch_win_with_autoupdate.bat`](../launch_win_with_autoupdate.bat)
  - [ ] [`launch_win_amd_nightly.bat`](../launch_win_amd_nightly.bat)
- [ ] Included a brief verification result for each script touched by this PR.

### First-run model-picker verification
- [ ] Verified first-run model package selection flow end-to-end where applicable:
  - [ ] [`utils/model_packages.py`](../utils/model_packages.py)
  - [ ] [`ui/model_package_selector_dialog.py`](../ui/model_package_selector_dialog.py)
- [ ] Confirmed behavior for first launch, model package listing, selection, and persistence.
- [ ] Included before/after screenshots for any UI change to the first-run model dialog.

## UX / Behavior Validation
- [ ] For UX-facing changes, I documented a clear **manual verification path** (steps + expected result).
- [ ] If defaults changed, I explained the **migration impact** for existing users/configurations.

## Risks / Compatibility
- Potential regressions:
- Rollback notes (if needed):

## Additional Notes
- Environment limitations (if any):

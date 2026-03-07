# Updating UI translations

The app uses Qt Linguist (.ts / .qm) for UI language. Display language is set in **Config → General → Display language** or **View → Display Language**. After changing the language, the app restarts to load the new locale.

## Fixing missing or incomplete translations (e.g. 简体中文)

1. **Edit the translation source**  
   - `translate/zh_CN.ts` is the source for Simplified Chinese.  
   - Open it in Qt Linguist or a text editor. Each `<message>` has `<source>` (English) and `<translation>`. Fill in or fix the `<translation>` for any missing or wrong strings.

2. **Add new strings from code**  
   - New UI strings are picked up by running the update script, which scans `ui/**/*.py` for `self.tr("...")` and merges into the .ts file:  
     `python scripts/update_translation.py`  
   - This uses `pylupdate6` (or `pylupdate5`). Install Qt dev tools if needed.  
   - After running, open the .ts file and add translations for any new entries (they appear with an empty or unfinished `<translation>`).

3. **Compile .ts to .qm**  
   - The app loads **.qm** files, not .ts. After editing `.ts`, compile it:  
     `python scripts/compile_translation.py`  
   - Or run **lrelease** directly if you have Qt Linguist / Qt SDK in PATH:  
     `lrelease translate/zh_CN.ts`  
   - Or: `lrelease translate/zh_CN.ts -qm translate/zh_CN.qm`  
   - On Windows, **lrelease** is in the Qt bin folder (e.g. with PyQt6, you may need to install [Qt SDK](https://www.qt.io/download) or use the one bundled with your Qt install). Alternatively, `pip install pyqt6-tools` and run `python -m pylrelease6 translate/zh_CN.ts -qm translate/zh_CN.qm`.

4. **Restart the app**  
   - Set display language to 简体中文 (or the locale you edited) and restart so the new .qm is loaded.

## QFont warning when switching to 简体中文

If you see `QFont::setPointSize: Point size <= 0 (-1)` in the log when using 简体中文 (or other locales), the app now sanitizes the default font in `launch.py` and in `utils/widget.py` so both `pointSize()` and `pointSizeF()` are valid. If it still appears, ensure you are on the latest commit that includes the font sanitization fix.

## Files involved

| File / path | Role |
|-------------|------|
| `translate/zh_CN.ts` | Source translations for Simplified Chinese (edit this). |
| `translate/zh_CN.qm` | Compiled locale loaded at runtime (generated from .ts). |
| `scripts/update_translation.py` | Scans UI Python files and updates .ts with new/updated source strings. |
| `utils/shared.py` | `DISPLAY_LANGUAGE_MAP`, `DEFAULT_DISPLAY_LANG`. |
| `launch.py` | Loads translator and .qm for the configured display language. |

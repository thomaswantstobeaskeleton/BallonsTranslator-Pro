# Release v1.5.1 — PyQt6 text eraser fix, CTD box shrink, README updates

Use this for the **GitHub release**: title, tag, and description body.

---

## Tag

```
v1.5.1
```

Create the tag **after** committing all changes and **before** or when creating the release:

```bash
git tag -a v1.5.1 -m "v1.5.1 — PyQt6 text eraser fix, CTD box shrink, README updates"
git push origin v1.5.1
```

---

## Release title

```
v1.5.1 — PyQt6 text eraser fix, CTD box shrink, README updates
```

---

## Release description (paste into GitHub)

```markdown
**BallonsTranslatorPro** extended fork — patch release with a PyQt6 compatibility fix, CTD box-size controls, and documentation updates.

---

## What's new in v1.5.1

### Fixes
- **Text eraser (PyQt6):** Fixed crash when finishing a stroke. `QRectF.intersects()` in PyQt6 expects a `QPolygonF`; the text eraser now passes the correct type so the tool works on Qt 6.

### New / improved
- **CTD (ComicTextDetector) — smaller boxes:** If CTD produces boxes that are too large:
  - **box_shrink_px** (Config → Text detection → CTD): shrink each box inward by 4–12 px (0 = off). Applied before box_padding.
  - **box_padding:** set to **0** (or 1–2) to stop adding extra margin.
  - **merge font size tolerance:** lower it (e.g. 2.0 or 1.5) so fewer lines are merged per box.
- **README:** Dual detection recommended combo (Paddle v5 + HF ogkalu, `labels_include`), and §7.1 CTD settings updated with "boxes too large" tips.

---

## Quick start (zip)

1. **Extract** the zip to a folder.
2. **First run:** `python launch.py` — installs base deps and downloads default models into `data/`.
3. **Settings:** Open the settings panel → choose **Text detection**, **OCR**, **Inpainting**, **Translation**. New modules appear when dependencies are installed.
4. **Docs:** See [README](https://github.com/thomaswantstobeaskeleton/BallonsTranslator-Pro/blob/main/README.md) and `docs/` for quality rankings, manhua settings, and optional dependencies.

---

## Requirements

- **Python:** 3.10 or 3.11 (avoid Microsoft Store Python).
- **Models:** If the first-run download fails, use the links in the README (MEGA / Google Drive) and place the `data` folder in the project root.

---

**Repository:** [BallonsTranslator-Pro](https://github.com/thomaswantstobeaskeleton/BallonsTranslator-Pro) · **Upstream:** [BallonsTranslator](https://github.com/dmMaze/BallonsTranslator)
```

---

## Optional: bump version in code before release

If you want the app to show **1.5.1** in the UI and logs, update `launch.py`:

```python
VERSION = '1.5.1'
```

Then commit that change with the rest before tagging.

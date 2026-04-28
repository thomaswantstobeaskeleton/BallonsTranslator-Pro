# Model manifest maintenance

The model manifest lives at `data/model_manifest.json`.

## What it controls

- First-launch package choices (`core`, `advanced_ocr`, etc.).
- Package labels/descriptions in the package selector dialog.
- Model metadata shown in **Tools → Manage models** tooltips:
  - size estimate,
  - required dependencies,
  - support tier.

If the manifest file is missing or invalid, the app falls back to built-in hardcoded package definitions.

## Schema (v1)

Top-level keys:

- `schema_version` (integer)
- `modules` (list)
- `packages` (list)

### `modules[]` required fields

- `module_key` (string)
- `category` (string)
- `size_estimate` (string)
- `required_deps` (list of strings)
- `support_tier` (string)

### `packages[]` required fields

- `id` (string)
- `label` (string)
- `description` (string)
- `modules` (list of `{ "category": str, "module_key": str|null }`)

Notes:

- `module_key` may be `null` only for special pseudo entries such as `_optional_onnx`.
- Every non-null package module key must exist in `modules[]`.

## Add a model

1. Add a new entry in `modules[]` with all required metadata.
2. Add it to one or more package `modules` lists.
3. If needed, update package label/description text.
4. Run:
   - `python -m compileall -f -q utils/model_packages.py ui/model_manager_dialog.py utils/model_manager.py launch.py`

## Remove a model

1. Remove the model from all package `modules` lists.
2. Remove its entry from `modules[]`.
3. Run the same compile check command.

## Startup validation

At app startup, `utils.model_packages.validate_manifest_on_startup()` validates schema and logs errors.
On validation failure, the app keeps running with fallback package data.

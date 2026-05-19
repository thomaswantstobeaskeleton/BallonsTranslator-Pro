# Local Automation API and MCP Command Surface

Last updated: 2026-05-19

BallonsTranslator-Pro exposes a localhost-only automation API when `automation_api_enabled` is enabled in config. The API is designed for headless helpers, MCP-style tools, and external workflow scripts while keeping existing UI progress behavior intact.

## Discovery and health

- `GET /health` returns service status plus the same route catalog as `/routes`.
- `GET /routes` returns stable sorted POST routes, GET routes, MCP-compatible command routes, job routes, and the SSE event-stream template.
- `GET /events?job_id=<job_id>` returns a Server-Sent Events snapshot for a job. Events now include SSE `id` (job `updated_at`) and `event` (current job status such as `running` / `succeeded` / `failed` / `cancelled`) plus a bounded status/log snapshot payload; this keeps clients forward-compatible with future live streaming loops.

If `automation_api_key` is set, send it as `X-API-Key` for every request, including discovery and events.

### Route discovery payload fields

`GET /routes` and `GET /health` share the same discovery payload contract:

- `routes`: sorted list of POST handler names.
- `methods.GET`: sorted GET endpoints (`events`, `health`, `routes` by default).
- `methods.POST`: same sorted list as `routes`.
- `mcp_routes`: subset of `routes` that map to MCP-compatible commands.
- `job_routes`: subset of `routes` for job lifecycle endpoints (`job_*`, `jobs_list`).
- `event_stream`: stable SSE template path (`/events?job_id=<job_id>`).


## MCP-friendly commands

The route catalog marks these commands as MCP-compatible when the UI registers them:

- `open_project`
- `project_status`
- `list_pages`
- `apply_edit`
- `run_pipeline`
- `render`
- `export`
- `undo`
- `redo`

Aliases such as `project_open`, `pipeline_run`, `scene_edit`, and `render_page` remain supported for compatibility.

## Long-running jobs

Use `POST /job_start` for long-running operations. Supported task names and aliases normalize to a small stable set:

| Canonical task | Accepted aliases |
|---|---|
| `run_pipeline` | `run_pipeline`, `pipeline_run` |
| `render_page` | `render`, `render_page`, `render_current_page` |
| `export` | `export` |
| `batch_export` | `batch_export`, `export_archive` |
| `proof_pack` | `proof_pack`, `export_lettering_proof`, `lettering_proof` |

Job lifecycle routes:

- `POST /job_status` with `{"job_id": "..."}`
- `POST /job_cancel` with `{"job_id": "..."}`
- `POST /job_logs` with `{"job_id": "...", "offset": 0}` for incremental log polling
- `POST /job_result` with `{"job_id": "..."}`
- `POST /jobs_list` with optional `{"limit": 50}`

The job status payload now includes `stage` and `progress` (`0.0..1.0`) for stable automation progress reporting.

Cancellation is cooperative. A queued job can be cancelled immediately; a running job exposes `cancel_requested` so deeper pipeline stages can progressively add cancellation checkpoints.

## Example curl flow

```bash
curl http://127.0.0.1:39542/routes
curl -X POST http://127.0.0.1:39542/open_project \
  -H 'Content-Type: application/json' \
  -d '{"path":"/path/to/project-or-folder"}'
curl -X POST http://127.0.0.1:39542/job_start \
  -H 'Content-Type: application/json' \
  -d '{"task":"run_pipeline","payload":{}}'
curl http://127.0.0.1:39542/events?job_id=job_123
```

## Current limitations

- The event stream currently returns a status/log snapshot rather than an infinite stream.
- Job cancellation is exposed through the API but still depends on individual task implementations to check cancellation during expensive work.
- The API remains bound to `127.0.0.1` by design; do not expose it directly on untrusted networks.


### Scene edit payload

`POST /apply_edit` (alias: `/scene_edit`) accepts either one operation object or a batch payload:

```json
{
  "strict": true,
  "ops": [
    {"op": "add_textbox", "page": "001.png", "text": "Hello"},
    {"op": "update_textbox", "page": "001.png", "block_id": "tbx_...", "text": "Updated"},
    {"op": "delete_textbox", "page": "001.png", "index": 3},
    {"op": "undo"},
    {"op": "redo"}
  ]
}
```

Notes:
- `update_textbox` and `delete_textbox` require **either** `index` **or** `block_id`.
- Responses include stable `block_id` values for successful add/update/delete operations.
- When `strict=false`, the server continues batch execution and returns per-operation errors in `errors`.
- When `strict=true` (default), the first operation failure returns an API error.


## Headless batch CLI contract

The `scripts/batch_translate.py` runner now exposes an explicit stage selector for automation callers:

- `--stages detect,ocr,translate,inpaint` to run only a stable subset of stages.
- `--save-text-json <path|dir>` exports translation JSON after processing.
- `--load-text-json <path>` imports translation JSON before processing each project.
- Empty `--stages` keeps default behavior (all enabled pipeline stages from config, subject to `--no-*` flags).
- Invalid stage names fail fast with `HeadlessExitCode.INPUT_ERROR`.

Summary JSON payloads include:

- `stages`: normalized sorted stage list requested by the caller.
- `warnings`: non-fatal contract warnings (for example, when no stages remain enabled after config + flag filtering).


For non-GUI runs, `scripts/batch_translate.py` now returns stable non-zero exit codes:

- `0`: success
- `2`: config error (e.g. missing config file)
- `3`: input error (no valid input dirs)
- `4`: runtime error (all requested dirs failed)
- `5`: partial failure (some dirs succeeded, some failed)

Optional output:

- `--summary-json <path>` writes a machine-readable run summary (`requested/processed/skipped/failed`, plus `exit_code`).


## Credential migration note

Phase 2A migration support is now active for sensitive API keys used by automation/layout-review/video-flow-fixer settings.

- When OS keyring is available, existing plaintext secrets are migrated and (by default) scrubbed from future config saves.
- `credential_use_plaintext_fallback=true` keeps plaintext values for environments where keyring integration is intentionally bypassed.
- UI now shows credential backend status in layout-review provider config.


## Provider connection checks

Phase 2B setup now exposes a connection test path in the Layout Review provider settings dialog:

- Endpoint presets auto-fill common providers (OpenAI, OpenRouter, Google-compatible, LM Studio, Ollama).
- **Test connection** performs a lightweight model-list probe and shows success/failure detail.
- Requests use runtime timeout settings (`runtime_http_timeout_sec`).


## XLIFF interchange

Phase 3 interchange now includes XLIFF 1.2 export/import hooks:

- API export: `POST /export_xliff` (or `POST /export` with `{"kind":"xliff"}`)
- API import: `POST /import_xliff` with `{"path":".../file.xliff"}`
- UI: Tools → Export → Export XLIFF / Import XLIFF

Current mapping is stable per page + block index (`page::index`) and returns match diagnostics for missing/unmatched pages.


## Translation JSON interchange

Added Phase 3 roundtrip translation JSON hooks for deterministic external editing:

- API export: `POST /export_translation_json` (or `POST /export` with `{"kind":"translation_json"}`)
- API import: `POST /import_translation_json` with `{"path":".../translation_export.json"}`
- UI: Tools → Export → Export translation JSON / Import translation JSON

Format includes stable `block_id` plus `index` fallback for resilient re-import.


## Translation CSV interchange

Added Phase 3 translation CSV roundtrip hooks:

- API export: `POST /export_translation_csv` (or `POST /export` with `{"kind":"translation_csv"}`)
- API import: `POST /import_translation_csv` with `{"path":".../translation_export.csv"}`
- UI: Tools → Export → Export translation CSV / Import translation CSV

Columns: `page,index,block_id,source,translation` with stable `block_id` matching and index fallback.


## Translation memory (TM) API

Phase 4 TM groundwork is now available via local automation routes:

- `POST /tm_build_from_project` — build project TM from current source/translation pairs
- `POST /tm_query` with `{"source":"...","min_score":0.65,"limit":5}`
- `POST /tm_export` writes `translation_memory.json`
- `POST /tm_import` imports TM JSON (`merge=true` by default)

TM entries are persisted in project JSON as `translation_memory` and include `source`, `target`, `page`, and `block_id`.


## Translation QA and prompt profiles API

Phase 4 now exposes profile-aware translation QA endpoints:

- `POST /translation_prompt_profiles` → available profile IDs + current default
- `POST /translation_qa_report` with optional `page`, `profile`, `retry_issue_threshold`

The QA report returns per-block issues and `retry_candidates` so automation can optionally re-run poor-quality blocks.


## Batch find/replace API

Phase 5 editor UX now includes previewable batch find/replace routes:

- `POST /batch_find_replace_preview` with `pattern`, `replacement`, optional `pages`, `use_regex`, `case_sensitive`
- `POST /batch_find_replace_apply` with either the same fields or a prior `preview` payload

`preview` returns match samples without mutating the project; `apply` mutates translations and returns changed rows.


## Concordance and glossary API

- `POST /concordance_query` with `{query, in_target=true, limit=50}` returns project-wide source/target matches with `page`, `index`, and `block_id` provenance.
- `POST /glossary_export` with optional `{format: "json"|"csv", path}` exports project glossary.
- `POST /glossary_import_preview` with `{path, mode: "merge"|"replace"}` returns non-destructive preview counts (`added/skipped/result`).
- `POST /glossary_import` with `{path, mode: "merge"|"replace"}` imports glossary entries from JSON or CSV.


## OCR crop inspector API

- `POST /ocr_rerun_block` with `{page?, index, engine?}` reruns OCR on one textbox only and returns updated OCR text, confidence, and selected engine.

- `POST /ocr_compare_block` with `{page?, index, secondary_engine}` compares current OCR text against a secondary OCR engine for one block (non-destructive).
- `POST /ocr_apply_compare_choice` with `{page?, index, text, engine?}` applies a chosen OCR text to one block.

- `POST /renderer_diagnostics` returns renderer backend capability diagnostics (Qt default renderer, optional shaping modules, warnings).


## Cleanup-only workflow API

- `POST /cleanup_only` with optional `{pages, out_dir}` runs detect + inpaint only (no OCR/translation) and optionally exports clean images.


## Parent/child batch planning API

- `POST /batch_parent_enumerate` with `{parent_path}` discovers child projects from nested image folders and `.cbz`/`.zip` archives.
- `POST /batch_parent_save_state` with `{parent_path, state_path, statuses?}` writes resumable parent-batch state JSON (`pending/done/failed/...`).
- `POST /batch_parent_load_state` with `{state_path}` loads saved parent-batch state payload.
- `POST /batch_parent_update_status` with `{state_path, input_path, status}` updates one child status.
- `POST /batch_parent_next_pending` with `{state_path}` returns next pending child item.


## Data path manager API

- `POST /data_path_status` with optional `{path, min_free_gb}` returns resolved path, existence, free space, and threshold check.
- `POST /data_path_migrate` with `{source, dest, dry_run=true}` previews or performs data-path migration for top-level entries.


## Docker / server mode quickstart

For headless/API usage, run Pro with automation API enabled and query the helper route:

- `POST /server_mode_info` returns health/routes URLs, path hints, Docker mount hints, and ready-to-run curl/Python snippets.

Example curl:

```bash
curl -X POST http://127.0.0.1:39542/server_mode_info -H 'Content-Type: application/json' -d '{}'
```

Typical container mount hints from the payload:

- `/app/data/models` for model/cache volume
- `/app/projects` for project input/output volume
- `/app/config/config.json` for persisted configuration


## Import translated image alignment API

- `POST /import_translated_image_align` with `{page, translated_image, min_iou?}` detects/OCRs the translated image and maps translations back to raw blocks by IoU.

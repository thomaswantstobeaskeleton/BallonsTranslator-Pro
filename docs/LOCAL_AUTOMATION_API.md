# Local Automation API and MCP Command Surface

Last updated: 2026-05-19

BallonsTranslator-Pro exposes a localhost-only automation API when `automation_api_enabled` is enabled in config. The API is designed for headless helpers, MCP-style tools, and external workflow scripts while keeping existing UI progress behavior intact.

## Discovery and health

- `GET /health` returns service status plus the same route catalog as `/routes`.
- `GET /routes` returns stable sorted POST routes, GET routes, MCP-compatible command routes, job routes, and the SSE event-stream template.
- `GET /events?job_id=<job_id>` returns a Server-Sent Events snapshot for a job. The current implementation sends a bounded status/log snapshot; future work can extend this to a live streaming loop without changing the endpoint.

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

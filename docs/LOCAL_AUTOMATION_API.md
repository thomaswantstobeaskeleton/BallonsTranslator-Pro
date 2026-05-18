# Local Automation API and MCP Command Surface

Last updated: 2026-05-18

BallonsTranslator-Pro exposes a localhost-only automation API when `automation_api_enabled` is enabled in config. The API is designed for headless helpers, MCP-style tools, and external workflow scripts while keeping existing UI progress behavior intact.

## Discovery and health

- `GET /health` returns service status plus the same route catalog as `/routes`.
- `GET /routes` returns stable sorted POST routes, GET routes, MCP-compatible command routes, job routes, and the SSE event-stream template.
- `GET /events?job_id=<job_id>` returns a Server-Sent Events snapshot for a job. The current implementation sends a bounded status/log snapshot; future work can extend this to a live streaming loop without changing the endpoint.

If `automation_api_key` is set, send it as `X-API-Key` for every request, including discovery and events.

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
- `POST /job_logs` with `{"job_id": "..."}`
- `POST /job_result` with `{"job_id": "..."}`
- `POST /jobs_list` with optional `{"limit": 50}`

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

# Feature Parity Matrix (Working Draft)

Last updated: 2026-05-19

This matrix tracks BallonsTranslator-Pro parity progress against major alternatives.

## Phase status snapshot

- **Phase 0 (Audit & safety baseline):** ✅ **in progress / mostly done baseline**
  - Audit plan documented in `docs/ALTERNATIVES_FEATURE_GAP_IMPLEMENTATION_PLAN.md`.
  - Route discovery and export/proof/save-state tests are in place and being expanded.
  - Baseline verification rerun on 2026-05-19; route/export/save-state suites passed, with a known model-picker test import stub failure documented for follow-up.
- **Phase 1 (Automation/headless/MCP):** 🟨 **actively in progress**
  - Added typed edit-op validation and batch scene-edit surface.
  - Added job lifecycle routes (`job_start/status/cancel/logs/result/list`).
  - Added MCP-friendly aliases, deterministic route discovery tests, and an SSE-compatible `/events` job snapshot endpoint.
  - Remaining: deeper task cancellation hooks, CLI/headless runner contract, live SSE/WS streaming loop.
- **Phase 2+ (Secure providers, interchange, CAT, OCR UX, renderer fidelity, cleanup, batch parity):** ❌ **not started in implementation yet**

## Relative parity (high-level)

| Capability Area | Upstream BallonsTranslator | BallonsTranslator-Pro (current) | manga-image-translator | Koharu | ImageTrans | manga-translator-ui | SickZil-Machine |
|---|---|---|---|---|---|---|---|
| Multi-engine OCR/translate/inpaint | Partial | Strong | Strong | Strong | Partial | Partial | Partial |
| Typography QA / layout review | Limited | Strong | Partial | Partial | Partial | Limited | Limited |
| Local automation API | Limited | Strong (expanding) | Partial | Strong | Limited | Partial | Limited |
| Job lifecycle for long tasks | Limited | Partial (implemented base lifecycle) | Partial | Strong | Limited | Limited | Limited |
| Scene edit API schema | Limited | Partial (typed validation + batch ops) | Partial | Strong | Limited | Limited | Limited |
| Provider secure onboarding | Limited | Partial | Partial | Partial | Partial | Partial | Limited |
| Professional interchange (XLIFF/DOCX/etc.) | Limited | Partial | Partial | Partial | Strong | Limited | Limited |
| CAT TM/termbase/concordance | Limited | Partial | Limited | Partial | Strong | Limited | Limited |
| Advanced shaping backend | Limited | Partial | Partial | Partial | Partial | Partial | Limited |
| Cleanup-only mask/inpaint workflow | Limited | Partial | Partial | Partial | Partial | Limited | Strong |

> Notes:
> - "Partial" here means at least one significant building block exists, but full acceptance criteria for the target phase are not yet complete.
> - This file is intentionally conservative and will be updated after each phase PR.

## Immediate next milestones

1. **Phase 1 next:** complete cooperative cancellation wiring deeper into long-running tasks and add endpoint tests around job lifecycle paths.
2. **Phase 1 next:** formalize headless CLI pipeline invocation contract and exit codes.
3. **Phase 2 start:** introduce keyring-backed provider secret storage with migration tests.

## 2026-05-20 Phase 0/1 checkpoint
- Realtime mode and Translation Assist were re-audited before implementation updates.
- Existing realtime/API scaffolding was extended (not duplicated) for first milestone route wiring and UI entry points.
- Next implementation slices remain phased to avoid startup/project compatibility regressions.

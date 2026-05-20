# Translation Assist Plan (Phase 0/1 Baseline)

## Status Legend
- ✅ already implemented
- 🟨 partially implemented
- ⛔ not implemented

## Current baseline audit
- 🟨 Translation memory, glossary, concordance primitives already exist in project/API utilities.
- ⛔ Dedicated per-block Translation Assist dock with multi-provider candidate comparison is not complete.
- ⛔ SFX dictionary suggestions are not integrated into a dedicated assist surface yet.

## First PR scope (this series)
1. Add Translation Assist planning doc and route contract wiring.
2. Add Translation Assist placeholder dock (manual open, no auto-overwrite behavior).
3. Add local automation API namespace endpoints:
   - `translation_assist_block`
   - `translation_assist_candidates`
   - `translation_assist_apply_candidate`
4. Add tests proving assist flow does not mutate project text unless candidate apply is explicitly called.
5. Add namespace placeholders for TM/glossary/concordance/SFX route wiring so `/routes` discovery is stable for automation clients.
6. Wire TM/glossary/concordance/SFX assist endpoints to existing project data/query helpers (non-destructive by default).

## Current implementation progress (2026-05-20 follow-up)
- ✅ `translation_assist_block` now resolves real project page/block and returns source/target text from actual text blocks.
- ✅ `translation_assist_tm` uses existing translation-memory fuzzy matching.
- ✅ `translation_assist_glossary` uses existing project glossary entries for term hits.
- ✅ `translation_assist_concordance` uses existing concordance project search.
- ✅ `translation_assist_sfx` uses default + project SFX dictionary search.
- ✅ Translation Assist apply path now uses dedicated undo-command integration in editor flow (apply from dock pushes an undoable command).
- ✅ Translation Assist dock now supports user-controlled assist source toggles (TM/Glossary/SFX/Concordance) and max-candidate cap persisted via config fields.
- ✅ Candidate aggregation now de-duplicates repeated suggestions across TM/glossary/SFX/concordance and enforces max-candidate caps.
- ✅ Assist dock now auto-refreshes on in-canvas block selection changes (when dock is visible and feature is enabled).
- ✅ Assist dock now supports one-click “Add selected candidate to TM” and “Add selected candidate to Glossary” actions.
- ✅ Assist panel shows lightweight per-block QA warnings (empty target, untranslated risk, suspiciously short target).
- ✅ Added `translation_assist_qa` API route to expose per-block guardrail warnings for automation clients and dock display.
- ✅ Assist dock supports edit/merge-before-apply workflow via explicit “Apply Edited Text” action and `translation_assist_apply_text` API route.
- ✅ Assist candidate cache added (per normalized source + profile/providers) with explicit clear action in dock/API.
- ✅ Candidate rows now include lightweight provider telemetry metadata (source + latency) for comparison context.
- ✅ Added `translation_assist_compare` route with low-latency/high-quality presets for provider-group comparison orchestration.
- ✅ Added provider-warning normalization + candidate cost telemetry helpers, and synthetic external-provider fan-out placeholders for compare API output shape.
- ✅ Compare flow now supports scope switching (`translator`, `ocr`, `detector`, `inpainter`) from the dock, so users can compare/apply module choices from the same assist surface.
- ✅ Right-side text editor list now exposes a context menu with direct compare actions (translator/OCR/detector/inpainter), plus quick open/jump actions so compare is reachable without opening the dock first.
- ✅ Compare apply flow now uses a dedicated table dialog (provider, candidate, latency, source, current-engine marker) instead of numeric index prompts, improving usability and reducing wrong-pick risk.
- ✅ Applying a compare candidate can now switch active OCR/detector/inpainter module configuration (saved as project dirty state) in addition to text candidate application.

## Risk / migration
- Low risk: additive UI + additive API namespace.
- No config migration required in this slice.
- Existing editor, save/load, export, and startup paths remain unchanged.

## Tests to add in later slices
- multi-engine comparison cache behavior
- TM fuzzy match scoring in assist panel
- glossary violation checks + one-click safe apply
- SFX lookup integration in assist panel
- provider failure isolation

## User-facing docs needed later
- `docs/TRANSLATION_ASSIST.md`
- `docs/SFX_DICTIONARY.md`
- README / README_zh_CN parity section for Translation Assist workflow

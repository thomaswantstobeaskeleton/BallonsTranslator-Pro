# Upstream BallonsTranslator Dev-Branch Sync

_Last reviewed: 2026-05-08. Source: `git fetch upstream-base dev --depth=100`, then `git log --oneline --max-count=30 upstream-base/dev` and `git log --name-status --max-count=20 upstream-base/dev`._

| Commit | Summary | Changed files | Category | Relevant to Pro? | Already implemented in Pro? | Ported this pass? | Porting notes | Deferred reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [`6649de1`](https://github.com/dmMaze/BallonsTranslator/commit/6649de101d61046353b434383d88bfa6718b47c7) | update project save state logic #1178 | `ui/mainwindow.py` | Bugfix; Export/project workflow | yes | partial | **yes** | Adapted manually: final render/save now updates saved undo step and save-state only after `saveImg` is reached successfully, preserving Pro's upscale/colorization/export behavior. | — |
| [`c80eb81`](https://github.com/dmMaze/BallonsTranslator/commit/c80eb81d7c65ca6ecb48551bd9d70c2d416f836c) | fix #1179 | `requirements.txt` | Dependency/platform compatibility | yes | partial | no | Reviewed. Upstream pins/bumps `transformers` narrowly. | Deferred because Pro intentionally keeps a broader `transformers>=4.56` range for GLM-OCR/HF module compatibility; pin needs full dependency matrix validation. |
| [`04c3414`](https://github.com/dmMaze/BallonsTranslator/commit/04c3414b7d741349e7c73260808565c516b6ea22) | pin transformers version, close #1177 | `requirements.txt`, `ui/textedit_area.py` | Dependency/platform compatibility; UI robustness | yes | partial | no | Reviewed. The `pynput` fallback idea conflicts with this repo's explicit no-new try/catch import rule and Pro still ships `keyboard`-based lookup. | Deferred pending a cleaner cross-platform lookup abstraction without import-time try/catch. |
| [`88d4969`](https://github.com/dmMaze/BallonsTranslator/commit/88d49695872a67e1f62e53a9f2a95400f9465d44) | add missing gguf dependency, close #1175 | `requirements.txt` | Dependency/platform compatibility | yes | yes | no | Pro requirements already include `gguf>=0.10.0`. | Already covered. |
| [`4c14019`](https://github.com/dmMaze/BallonsTranslator/commit/4c14019) | optimize replace-all/render-all UI freeze and save bug (#1170) | UI/render workflow files | Bugfix; Batch/project workflow | yes | partial | no | Reviewed at summary/name-status level. | Larger UI-flow port deferred until replace-all code paths are audited against Pro batch/export changes. |
| [`1958f66`](https://github.com/dmMaze/BallonsTranslator/commit/1958f66) | support Ollama as translation and OCR provider (#1167) | translator/OCR/provider files | OCR/detection/inpainting/translation module update | yes | unknown | no | High-value provider setup item. | Deferred because provider systems need safe Pro-specific integration and settings UI validation. |
| [`10f5e8e`](https://github.com/dmMaze/BallonsTranslator/commit/10f5e8e) | add option to translate each text block individually (#1165) | translation workflow files | UI/UX improvement; Translation workflow | yes | partial | no | Relevant for stability with LLM APIs. | Deferred to avoid duplicating existing Pro batch/API translation controls without a full provider/settings audit. |
| [`ec3ef9d`](https://github.com/dmMaze/BallonsTranslator/commit/ec3ef9d) | add low vram mode for llm_api (#1161) | LLM API config/module files | Performance/runtime/GPU | yes | unknown | no | Relevant to Pro runtime profiles. | Deferred for runtime/settings pass. |
| [`54ed6ae`](https://github.com/dmMaze/BallonsTranslator/commit/54ed6ae) / [`91f5d13`](https://github.com/dmMaze/BallonsTranslator/commit/91f5d13) | headless continuous mode | launch/headless files | Automation/API/headless | yes | partial | no | Pro already has local automation APIs and headless workflow pieces. | Deferred until headless continuous project queue can be integrated with Pro's API/events safely. |
| [`d677d1b`](https://github.com/dmMaze/BallonsTranslator/commit/d677d1b) / [`02a3a60`](https://github.com/dmMaze/BallonsTranslator/commit/02a3a60) | fix progress widget always on top #1104 | UI progress widget files | UI/UX improvement; Crash/regression fix | yes | unknown | no | Relevant polish item. | Deferred pending UI inspection against Pro's custom progress/frameless window code. |

## Ported / verified in this pass

- Re-reviewed `d677d1b`/`02a3a60` (#1104), `13fc383`, and `36fa134`; equivalent Pro behavior is already present in current code paths, so no duplicate port was applied.
- Added API health/routes discovery as a Pro-specific automation improvement adjacent to upstream headless/progress work.
- `6649de1` / upstream #1178 save-state fix was manually adapted in `ui/mainwindow.py` without cherry-picking, preserving Pro-specific final upscale/colorization behavior.

## Next dev-sync candidates

1. Inspect and safely adapt `4c14019` replace-all/render-all freeze + save bug.
2. Validate `c80eb81`/`04c3414` dependency constraints against Pro's OCR/LLM modules.
3. Port provider/runtime improvements from `1958f66`, `10f5e8e`, and `ec3ef9d` if Pro lacks equivalent controls.
4. Review progress-widget z-order fixes `d677d1b`/`02a3a60` against Pro frameless windows.
5. Evaluate headless continuous commits for integration with the existing local automation API.

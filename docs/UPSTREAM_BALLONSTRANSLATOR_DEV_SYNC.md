# Upstream BallonsTranslator Dev Sync

_Refreshed: 2026-05-14. Source: `dmMaze/BallonsTranslator` `dev`, fetched with `git fetch upstream-base dev --depth=150`._

## Implemented / ported in this pass

- Reviewed `6649de1`, `c80eb81`, `04c3414`, `88d4969`, `4c14019`, `1958f66`, `a390d4c`, `09e2e5b`, and adjacent recent dev commits. This pass did not cherry-pick blindly; it adapted workflow/export behavior and ported the safe shortcut/dependency compatibility portion of `a390d4c` while keeping Pro-specific shortcut systems.
- Prior port retained: `6649de1` save-state logic for #1178 remains documented as already in Pro; `09e2e5b` palette PNG handling was already present in Pro.

## Recent commit review

| Commit | Summary | Changed files | Category | Relevant to Pro | Already implemented | Ported this pass | Notes / deferred reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `6649de1` | (upstream-base/dev) update project save state logic #1178 | ui/mainwindow.py | Bugfix | Yes | Partial | Yes (adapted) | Reviewed for safe manual adaptation; no blind cherry-pick. |
| `c80eb81` | fix #1179 | requirements.txt | Bugfix | Yes | No/unknown | No | Reviewed for safe manual adaptation; no blind cherry-pick. |
| `04c3414` | pin transformers version, close #1177 | requirements.txt, ui/textedit_area.py | Dependency/platform compatibility | Yes | No/unknown | No | Deferred until Pro module compatibility matrix is tested. |
| `a05ef01` | remove deprecated deeplx | modules/translators/trans_deeplx.py, requirements.txt | Bugfix | Yes | No/unknown | No | Reviewed for safe manual adaptation; no blind cherry-pick. |
| `88d4969` | add missing gguf dependency, close #1175 | requirements.txt | Dependency/platform compatibility | Yes | No/unknown | No | Deferred until Pro module compatibility matrix is tested. |
| `640c4bf` | add reference to Ballonstranslator-Pro, close #1151 | README.md, README_EN.md | OCR/detection/inpainting/translation module update | Yes | No/unknown | No | Reviewed for safe manual adaptation; no blind cherry-pick. |
| `e1c93f2` | remove redundant code | utils/torch_utils.py | Bugfix | Yes | No/unknown | No | Reviewed for safe manual adaptation; no blind cherry-pick. |
| `485bbe8` | add flux inpaint pipeline #1171 (#1173) | launch.py, modules/base.py, modules/inpaint/base.py, modules/inpaint/flux_inpaint_pipeline.py, requirements.txt… | OCR/detection/inpainting/translation module update | Yes | No/unknown | No | Deferred until Pro module compatibility matrix is tested. |
| `a390d4c` | remove deprecated requirement keyboard | requirements.txt, ui/textedit_area.py, utils/structures.py | Dependency/platform compatibility | Yes | No | Yes (adapted) | Ported manually: removed hard `keyboard` dependency/import, added `pynput`-first SalaDict shortcut backend with fallback, and added annotation compatibility helper without reverting Pro shortcut systems. |
| `64a5713` | fix #1172 | modules/translators/trans_google.py | Bugfix | Yes | No/unknown | No | Reviewed for safe manual adaptation; no blind cherry-pick. |
| `4c14019` | 优化全部替换并渲染全部文件界面卡顿问题和 修复使用 全部替换并重新渲染 会导致图片切换不保存图片的bug (#1170) | ui/global_search_widget.py, ui/mainwindow.py | UI/UX improvement | Yes | Partial | Reviewed; adapted lower-risk workflow/export support | Relevant UI freeze/save-state fix; direct port deferred due Pro batch/report differences. This pass adds workflow planning/navigation and render manifest without touching replace internals. |
| `1958f66` | 支持Ollama作为翻译和OCR的服务提供 / support Ollama as trans & OCR provider (#1167) | modules/ocr/ocr_llm_api.py, modules/translators/trans_llm_api.py | OCR/detection/inpainting/translation module update | Yes | No/unknown | No | Deferred until Pro module compatibility matrix is tested. |
| `10f5e8e` | add option to translate each text block individually, close #1165 | modules/translators/base.py, modules/translators/trans_google.py, ui/module_manager.py, ui/module_parse_widgets.py, utils/config.py | Bugfix | Yes | No/unknown | No | Reviewed for safe manual adaptation; no blind cherry-pick. |
| `ec3ef9d` | add low vram mode for llm_api, close #1161 | modules/translators/trans_llm_api.py | Dependency/platform compatibility | Yes | No/unknown | No | Deferred until Pro module compatibility matrix is tested. |
| `13fc383` | fix stop translation | ui/module_manager.py | Bugfix | Yes | No/unknown | No | Reviewed for safe manual adaptation; no blind cherry-pick. |
| `c382df5` | 🚧 增加 HIP SDK 说明 | README.md | Dependency/platform compatibility | Yes | No/unknown | No | Deferred until Pro module compatibility matrix is tested. |
| `900031b` | 🚧 增加警告信息 | README.md | Bugfix | Yes | No/unknown | No | Reviewed for safe manual adaptation; no blind cherry-pick. |
| `acf993b` | 🚧 更新 AMD 现在 2026.1.1 驱动支持 Windows系统 时的配置和说明 | README.md, launch.py | Bugfix | Yes | No/unknown | No | Reviewed for safe manual adaptation; no blind cherry-pick. |
| `54ed6ae` | Added launch arg headless_continous and added the required checks | launch.py | Export/project workflow | Yes | No/unknown | No | Reviewed for safe manual adaptation; no blind cherry-pick. |
| `91f5d13` | Added headless continous by reprompting the user for new directories | ui/mainwindow.py | Export/project workflow | Yes | No/unknown | No | Reviewed for safe manual adaptation; no blind cherry-pick. |
| `00cd512` | Added headless continous check to headless checks | utils/message.py | Export/project workflow | Yes | No/unknown | No | Reviewed for safe manual adaptation; no blind cherry-pick. |
| `fc609c3` | Added headless continous check to headless checks | ui/framelesswindow/fw_qt5/utils/linux_utils.py, ui/framelesswindow/linux_utils.py, ui/framelesswindow/mac_utils.py, ui/framelesswindow/win32_utils.py | Export/project workflow | Yes | No/unknown | No | Reviewed for safe manual adaptation; no blind cherry-pick. |
| `5b5c21f` | Added Headless Continous variable | utils/shared.py | Export/project workflow | Yes | No/unknown | No | Reviewed for safe manual adaptation; no blind cherry-pick. |
| `16e491c` | close #1144 | requirements.txt | Bugfix | Yes | No/unknown | No | Reviewed for safe manual adaptation; no blind cherry-pick. |
| `09e2e5b` | fix #1133 | utils/io_utils.py | Bugfix | Yes | No/unknown | No | Reviewed for safe manual adaptation; no blind cherry-pick. |

## Next batch candidates

1. Inspect upstream `4c14019` replace-and-render changes file-by-file and adapt if Pro still has the freeze/save regression.
2. Evaluate `c80eb81`/`04c3414`/`88d4969` dependency pins against Pro install profiles.
3. Port safe pieces of `1958f66` Ollama OCR/translation only after provider settings conflict audit.
4. Review `10f5e8e` per-block translation UX against Pro selected-block translation actions.
5. Continue watching upstream export/project corruption fixes before major PSD writer work.

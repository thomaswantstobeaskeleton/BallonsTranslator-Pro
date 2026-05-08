# Upstream BallonsTranslator Issue Backlog for BallonsTranslator-Pro

_Last refreshed: 2026-05-08 via GitHub REST API against `dmMaze/BallonsTranslator`; 800 all-state issues/PRs scanned (#1181 through #308). This backlog is used to port/adapt upstream fixes without overwriting Pro-specific systems._


## Newly implemented / reviewed in this pass (2026-05-08 lettering proof + API discovery)

| Issue | Title | Category | Relevant labels | Maps to BT-Pro? | Already implemented in Pro? | Priority | Implementation notes | Deferred reason if not implemented |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [#1104](https://github.com/dmMaze/BallonsTranslator/issues/1104) | Progress widget topmost/crash regression | UI/UX editor workflow; Bugs/regressions/crashes | bug | yes | **already ported / verified** | P0 | Re-reviewed upstream progress-window dev commits; Pro already has non-topmost progress flags and faster thread-stop polling. API health/proof export adds safer long-workflow observability. | — |
| [#1178](https://github.com/dmMaze/BallonsTranslator/issues/1178) | Project save-state logic | Export/project workflow; Bugs/regressions/crashes | bug | yes | **ported previously / verified** | P0 | Current branch keeps the adapted save-state-after-success render fix. | — |
| [#995](https://github.com/dmMaze/BallonsTranslator/issues/995) / [#818](https://github.com/dmMaze/BallonsTranslator/issues/818) | Full-width/vertical text conversion and detector orientation | Text rendering / typography / fonts; Vertical CJK / RTL / punctuation | bug | yes | **advanced** | P1 | Vertical proof metrics and SVG/PSD handoff now preserve resolved writing mode, punctuation diagnostics, and vertical cell placement for review/export. | Detector orientation metadata port remains deferred. |

| Issue | Title | Category | Relevant labels | Maps to BT-Pro? | Already implemented in Pro? | Priority | Implementation notes | Deferred reason if not implemented |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [#1178](https://github.com/dmMaze/BallonsTranslator/issues/1178) | Bug Report: Inpaint output contains non-grayscale pixels for grayscale input | Bugfix; export/project workflow | bug | yes | ported in this pass | P0 | Adapted upstream save-state logic so failed render/save no longer marks the project saved. | — |
| [#1179](https://github.com/dmMaze/BallonsTranslator/issues/1179) | 最新版本的PaddleOCRVLManga执行报错 | Compatibility/dependency/platform issues | bug | yes | reviewed/deferred | P0 | Reviewed upstream dependency commits; Pro has broader dependency ranges/custom optional dependency docs. | Direct pin/dependency change deferred to avoid downgrading Pro modules. |
| [#1177](https://github.com/dmMaze/BallonsTranslator/issues/1177) | Bug Report: failed to set modules | Compatibility/dependency/platform issues | bug | yes | reviewed/deferred | P0 | Reviewed upstream dependency commits; Pro has broader dependency ranges/custom optional dependency docs. | Direct pin/dependency change deferred to avoid downgrading Pro modules. |
| [#1175](https://github.com/dmMaze/BallonsTranslator/issues/1175) | Bug Report: Flux inpainter fails to load GGUF model (flux-2-klein-4b-Q4_K_M.gguf) | Compatibility/dependency/platform issues | bug | yes | reviewed/deferred | P0 | Reviewed upstream dependency commits; Pro has broader dependency ranges/custom optional dependency docs. | Direct pin/dependency change deferred to avoid downgrading Pro modules. |
| [#818](https://github.com/dmMaze/BallonsTranslator/issues/818) | Bug Report: 用ysgyolo 文本檢測，原本书写方向是立会变成横的 | Vertical CJK / RTL / punctuation | bug | yes | advanced in this pass | P0 | Auto-polish resolves tall CJK OCR/detected boxes to vertical RL instead of leaving them horizontal. | Detector-side orientation metadata still deferred. |
| [#995](https://github.com/dmMaze/BallonsTranslator/issues/995) | Bug Report: full-width converted to half-width characters | Bugs/regressions/workflow | bug | yes | reviewed/deferred | P1-P2 | Relevant to Pro and kept in backlog for future safe porting. | Requires module-specific reproduction or conflicts with Pro additions. |
| [#997](https://github.com/dmMaze/BallonsTranslator/issues/997) | Bug Report: Bounding box shrinks when translation fails | Bugs/regressions/workflow | bug | yes | reviewed/deferred | P1-P2 | Relevant to Pro and kept in backlog for future safe porting. | Requires module-specific reproduction or conflicts with Pro additions. |
| [#1042](https://github.com/dmMaze/BallonsTranslator/issues/1042) | Bug 反馈:字体加载异常 | Text rendering / typography / fonts | bug | yes | advanced in this pass | P1 | Typography polish can apply configured fallback chain to selected/page boxes when glyphs are missing. | OS font discovery UI remains future work. |
| [#972](https://github.com/dmMaze/BallonsTranslator/issues/972) | Bug 反馈:应用字体之后点击格式会报错跳出 | Bugs/regressions/workflow | bug | yes | reviewed/deferred | P1-P2 | Relevant to Pro and kept in backlog for future safe porting. | Requires module-specific reproduction or conflicts with Pro additions. |
| [#1104](https://github.com/dmMaze/BallonsTranslator/issues/1104) | Bug 反馈: 發現更新後會出現閃退&加載表出現頂置狀態 | Bugs/regressions/workflow | bug | yes | reviewed/deferred | P1-P2 | Relevant to Pro and kept in backlog for future safe porting. | Requires module-specific reproduction or conflicts with Pro additions. |
| [#1165](https://github.com/dmMaze/BallonsTranslator/issues/1165) | Feature Request: 新增「逐個氣泡翻譯」式 | Bugs/regressions/workflow | — | yes | reviewed/deferred | P1-P2 | Relevant to Pro and kept in backlog for future safe porting. | Requires module-specific reproduction or conflicts with Pro additions. |
| [#1167](https://github.com/dmMaze/BallonsTranslator/pull/1167) | 支持Ollama作为翻译和OCR的服务提供 / support Ollama as trans & OCR provider | Bugs/regressions/workflow | — | yes | reviewed/deferred | P1-P2 | Relevant to Pro and kept in backlog for future safe porting. | Requires module-specific reproduction or conflicts with Pro additions. |
| [#1161](https://github.com/dmMaze/BallonsTranslator/issues/1161) | Why "low vram mode" missing in LLM_API_Translator? | Bugs/regressions/workflow | — | yes | reviewed/deferred | P1-P2 | Relevant to Pro and kept in backlog for future safe porting. | Requires module-specific reproduction or conflicts with Pro additions. |
| [#1154](https://github.com/dmMaze/BallonsTranslator/issues/1154) | Bug 反馈:KeyError程序试图从一个字典中获取一个叫 'ctd' 的键，但这个键在字典中并不存在。 | Bugs/regressions/workflow | bug | yes | reviewed/deferred | P1-P2 | Relevant to Pro and kept in backlog for future safe porting. | Requires module-specific reproduction or conflicts with Pro additions. |
| [#1120](https://github.com/dmMaze/BallonsTranslator/issues/1120) | Bug 反馈:使用zip部署最新的翻译器会报找不到ctd | Bugs/regressions/workflow | bug | yes | reviewed/deferred | P1-P2 | Relevant to Pro and kept in backlog for future safe porting. | Requires module-specific reproduction or conflicts with Pro additions. |
| [#1023](https://github.com/dmMaze/BallonsTranslator/issues/1023) | Bug Report: dot (.) change into E+00 in some config (eg. ChatGPT) | Bugs/regressions/workflow | bug | yes | reviewed/deferred | P1-P2 | Relevant to Pro and kept in backlog for future safe porting. | Requires module-specific reproduction or conflicts with Pro additions. |
| [#1098](https://github.com/dmMaze/BallonsTranslator/issues/1098) | [Bug] Translation gets stuck (infinite loading) with DeepSeek API | Bugs/regressions/workflow | bug | yes | reviewed/deferred | P1-P2 | Relevant to Pro and kept in backlog for future safe porting. | Requires module-specific reproduction or conflicts with Pro additions. |
| [#1181](https://github.com/dmMaze/BallonsTranslator/issues/1181) | Feature Request: Add 4 main module checkboxes in the left panel | Bugs/regressions/workflow | — | yes | reviewed/deferred | P1-P2 | Relevant to Pro and kept in backlog for future safe porting. | Requires module-specific reproduction or conflicts with Pro additions. |

## Next upstream issue candidates

1. #1177/#1179 dependency compatibility once Pro module constraints are revalidated.
2. #1175 GGUF/Flux loading diagnostics if Pro exposes the same model path.
3. #1165 per-block translation mode if not superseded by Pro batch/style APIs.
4. #1167 Ollama OCR/translation provider parity if Pro provider setup lacks it.
5. #1098 stuck translation cancellation/retry telemetry.
6. #1023 numeric config serialization regression audit.
7. #997 bounding-box preservation on failed translation.

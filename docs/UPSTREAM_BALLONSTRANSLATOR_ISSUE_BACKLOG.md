# Upstream BallonsTranslator Issue Backlog

_Refreshed: 2026-05-14. Source: `dmMaze/BallonsTranslator` GitHub REST API, 407 all-state issues scanned across paginated results._

## Implemented or advanced in this pass

- 2026-05-14 follow-up: upstream #1128/#1132/#1169/#1077 informed real tate-chu-yoko vertical layout, precise bounds diagnostics, selected textbox fixes, and batch workflow surfacing.
- 2026-05-14 follow-up: upstream packaging/shortcut compatibility from dev `a390d4c` was ported by removing the hard `keyboard` dependency/import and adding a `pynput`-first fallback backend.
- Upstream #1169/#1077/#1128/#1132/#1122 inspired the page lettering workflow and navigation over font/layout warnings; existing Pro smart-fit/font fallback is now packaged as a current-page command/API route.
- Upstream #1178 / dev `6649de1` remains ported from the prior pass; this pass keeps the save-state-safe render/export path and adds a manifest for API-rendered pages.
- Upstream #1170 was reviewed as relevant to replace-and-render UI freezing/save-state; this pass defers direct porting because Pro has separate batch/report infrastructure, but adds lower-risk workflow planning/navigation around rendering QA.

## Issue backlog

| Issue | Title | Category | Labels | Maps to Pro | Implemented in Pro | Priority | Notes / deferred reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#1185](https://github.com/dmMaze/BallonsTranslator/issues/1185) | Bug Report: Flux | Bugs/regressions/crashes | bug | Yes | Partial | Medium | Tracked for future pass. |
| [#1184](https://github.com/dmMaze/BallonsTranslator/issues/1184) | Bug 反馈: 无法调用Flux2-Klein图像修复 | Bugs/regressions/crashes | bug | Yes | Partial | Medium | Tracked for future pass. |
| [#1181](https://github.com/dmMaze/BallonsTranslator/issues/1181) | Feature Request: Add 4 main module checkboxes in the left panel | Feature requests/enhancements | none | Yes | Partial | Medium | Tracked for future pass. |
| [#1180](https://github.com/dmMaze/BallonsTranslator/issues/1180) | Feature Request: Add spacebar panning in Image-editting mod(画板模式) | Feature requests/enhancements | none | Yes | Partial | Medium | Tracked for future pass. |
| [#1179](https://github.com/dmMaze/BallonsTranslator/issues/1179) | 最新版本的PaddleOCRVLManga执行报错 | OCR/detection/inpainting | bug | Yes | Partial | Medium | Tracked for future pass. |
| [#1178](https://github.com/dmMaze/BallonsTranslator/issues/1178) | Bug Report: Inpaint output contains non-grayscale pixels for grayscale input | OCR/detection/inpainting | bug | Yes | Partial/deferred | Medium | Advanced by smart-fit workflow/API, next issue selection, fallback diagnostics, and render manifest; direct upstream UI freeze patch reviewed/deferred pending Pro conflict audit. |
| [#1177](https://github.com/dmMaze/BallonsTranslator/issues/1177) | Bug Report: failed to set modules | Bugs/regressions/crashes | bug | Yes | Partial | Medium | Tracked for future pass. |
| [#1175](https://github.com/dmMaze/BallonsTranslator/issues/1175) | Bug Report: Flux inpainter fails to load GGUF model (flux-2-klein-4b-Q4_K_M.gguf) | OCR/detection/inpainting | bug | Yes | Partial | Medium | Tracked for future pass. |
| [#1174](https://github.com/dmMaze/BallonsTranslator/issues/1174) | Bug 反馈: | Bugs/regressions/crashes | bug | Yes | Partial | Medium | Tracked for future pass. |
| [#1172](https://github.com/dmMaze/BallonsTranslator/issues/1172) | Bug Report: | Bugs/regressions/crashes | bug | Yes | Partial | Medium | Tracked for future pass. |
| [#1171](https://github.com/dmMaze/BallonsTranslator/issues/1171) | Feature Request: Add support for FLUX.2-klein-4B inpainting | OCR/detection/inpainting | none | Yes | Partial | Medium | Tracked for future pass. |
| [#1169](https://github.com/dmMaze/BallonsTranslator/issues/1169) | Feature Request: Font size based on block size | Text rendering / typography / fonts | none | Yes | Yes/advanced this pass | High | Advanced by smart-fit workflow/API, next issue selection, fallback diagnostics, and render manifest; direct upstream UI freeze patch reviewed/deferred pending Pro conflict audit. |
| [#1168](https://github.com/dmMaze/BallonsTranslator/issues/1168) | 添加了字体集功能并尝试修复了“仅识别fonts内字体” | Text rendering / typography / fonts | none | Yes | Partial | High | Tracked for future pass. |
| [#1166](https://github.com/dmMaze/BallonsTranslator/issues/1166) | PaddleOCR无论如何都安装不上，即使我安装CPU版本，也会显示不出来 | OCR/detection/inpainting | none | Yes | Partial | Medium | Tracked for future pass. |
| [#1165](https://github.com/dmMaze/BallonsTranslator/issues/1165) | Feature Request: 新增「逐個氣泡翻譯」式 | Feature requests/enhancements | none | Yes | Partial | Medium | Tracked for future pass. |
| [#1164](https://github.com/dmMaze/BallonsTranslator/issues/1164) | Feature Request: add parameter top k on ChatGpt/llm_api translator | Automation/API/headless/MCP | none | Yes | Partial | Medium | Tracked for future pass. |
| [#1162](https://github.com/dmMaze/BallonsTranslator/issues/1162) | Feature Request:可否改进Intel GPU launch脚本 | Compatibility/dependency/platform issues | none | Yes | Partial | Medium | Tracked for future pass. |
| [#1161](https://github.com/dmMaze/BallonsTranslator/issues/1161) | Why "low vram mode" missing in LLM_API_Translator? | Automation/API/headless/MCP | none | Yes | Partial | Low | Tracked for future pass. |
| [#1159](https://github.com/dmMaze/BallonsTranslator/issues/1159) | 我修改了项目的json文件的font_size，如何让项目应用修改后的json文件？ | Text rendering / typography / fonts | none | Yes | Partial | High | Tracked for future pass. |
| [#1158](https://github.com/dmMaze/BallonsTranslator/issues/1158) | Bug Report: | Bugs/regressions/crashes | bug | Yes | Partial | Medium | Tracked for future pass. |
| [#1155](https://github.com/dmMaze/BallonsTranslator/issues/1155) | Bug Report: | Bugs/regressions/crashes | bug | Yes | Partial | Medium | Tracked for future pass. |
| [#1154](https://github.com/dmMaze/BallonsTranslator/issues/1154) | Bug 反馈:KeyError程序试图从一个字典中获取一个叫 'ctd' 的键，但这个键在字典中并不存在。 | Bugs/regressions/crashes | bug | Yes | Partial | Medium | Tracked for future pass. |
| [#1151](https://github.com/dmMaze/BallonsTranslator/issues/1151) | Feature Request: Propose BallonsTranslator-Pro as experimental branch | Translation/provider/model setup | none | Yes | Partial | Medium | Tracked for future pass. |
| [#1150](https://github.com/dmMaze/BallonsTranslator/issues/1150) | Feature Request: Add community fork (BallonsTranslator-Pro) to README | Translation/provider/model setup | none | Yes | Partial | Medium | Tracked for future pass. |
| [#1148](https://github.com/dmMaze/BallonsTranslator/issues/1148) | Bug Report:FATAL: kernel `fmha_cutlassF_f32_aligned_64x64_rf_sm80` is for sm80-sm100, but was built for sm37 | Bugs/regressions/crashes | bug | Yes | Partial | Medium | Tracked for future pass. |
| [#1145](https://github.com/dmMaze/BallonsTranslator/issues/1145) | Bug 反馈:软件判定语言的标准是时区而不是系统语言 | Bugs/regressions/crashes | bug | Yes | Partial | Medium | Tracked for future pass. |
| [#1144](https://github.com/dmMaze/BallonsTranslator/issues/1144) | ysg yolo26 | Feature requests/enhancements | none | Yes | Partial | Low | Tracked for future pass. |
| [#1143](https://github.com/dmMaze/BallonsTranslator/issues/1143) | Feature Request:Stroke size and color for global settings | Settings/config/keybinds | none | Yes | Partial | Medium | Tracked for future pass. |
| [#1142](https://github.com/dmMaze/BallonsTranslator/issues/1142) | Feature Request:Local VLM and more context for better translation | Feature requests/enhancements | none | Yes | Partial | Medium | Tracked for future pass. |
| [#1141](https://github.com/dmMaze/BallonsTranslator/issues/1141) | Feature Request:Support of the acbf format, the cbz multilangual | Feature requests/enhancements | none | Yes | Partial | Medium | Tracked for future pass. |
| [#1140](https://github.com/dmMaze/BallonsTranslator/issues/1140) | Bug Report: when i use gemini api to transulate blank | Automation/API/headless/MCP | bug | Yes | Partial | Medium | Tracked for future pass. |
| [#1138](https://github.com/dmMaze/BallonsTranslator/issues/1138) | Feature Request: adding “Text on Path" | Feature requests/enhancements | none | Yes | Partial | Medium | Tracked for future pass. |
| [#1137](https://github.com/dmMaze/BallonsTranslator/issues/1137) | Feature Request: Manual Text Detection | UI/UX/editor workflow | none | Yes | Partial | Medium | Tracked for future pass. |
| [#1134](https://github.com/dmMaze/BallonsTranslator/issues/1134) | Feature Request: Export or save image as JPG or PNG without translation after manually inpainted and translated!! | PSD/export/layers | none | Yes | Yes/advanced this pass | High | Advanced by smart-fit workflow/API, next issue selection, fallback diagnostics, and render manifest; direct upstream UI freeze patch reviewed/deferred pending Pro conflict audit. |
| [#1133](https://github.com/dmMaze/BallonsTranslator/issues/1133) | Bug 反馈:PNG图片无法打开 | Bugs/regressions/crashes | bug | Yes | Partial | Medium | Tracked for future pass. |
| [#1132](https://github.com/dmMaze/BallonsTranslator/issues/1132) | Bug Report: Adding latin text after the original language | Bugs/regressions/crashes | bug | Yes | Yes/advanced this pass | Medium | Advanced by smart-fit workflow/API, next issue selection, fallback diagnostics, and render manifest; direct upstream UI freeze patch reviewed/deferred pending Pro conflict audit. |
| [#1131](https://github.com/dmMaze/BallonsTranslator/issues/1131) | 用vl视觉模型代替文字识别和OCR | OCR/detection/inpainting | none | Yes | Partial | Medium | Tracked for future pass. |
| [#1128](https://github.com/dmMaze/BallonsTranslator/issues/1128) | Bug Report: Vertical Text not working | Text rendering / typography / fonts | bug | Yes | Yes/advanced this pass | High | Advanced by smart-fit workflow/API, next issue selection, fallback diagnostics, and render manifest; direct upstream UI freeze patch reviewed/deferred pending Pro conflict audit. |
| [#1127](https://github.com/dmMaze/BallonsTranslator/issues/1127) | 安裝時出現問題 | Bugs/regressions/crashes | bug | Yes | Partial | Medium | Tracked for future pass. |
| [#1126](https://github.com/dmMaze/BallonsTranslator/issues/1126) | 安裝時出現問題 | Bugs/regressions/crashes | bug | Yes | Partial | Medium | Tracked for future pass. |
| [#1122](https://github.com/dmMaze/BallonsTranslator/issues/1122) | Bug Report:No font on my side? | Text rendering / typography / fonts | bug | Yes | Yes/advanced this pass | High | Advanced by smart-fit workflow/API, next issue selection, fallback diagnostics, and render manifest; direct upstream UI freeze patch reviewed/deferred pending Pro conflict audit. |
| [#1121](https://github.com/dmMaze/BallonsTranslator/issues/1121) | Feature Request: Open Import Translation From.... default change | Feature requests/enhancements | none | Yes | Partial | Medium | Tracked for future pass. |
| [#1120](https://github.com/dmMaze/BallonsTranslator/issues/1120) | Bug 反馈:使用zip部署最新的翻译器会报找不到ctd | Bugs/regressions/crashes | bug | Yes | Partial | Medium | Tracked for future pass. |
| [#1119](https://github.com/dmMaze/BallonsTranslator/issues/1119) | Bug Report: YSGYOLO setting not translated? | Translation/provider/model setup | bug | Yes | Partial | Medium | Tracked for future pass. |
| [#1118](https://github.com/dmMaze/BallonsTranslator/issues/1118) | Feature Request: 仅运行本页 | Feature requests/enhancements | none | Yes | Partial | Medium | Tracked for future pass. |
| [#1111](https://github.com/dmMaze/BallonsTranslator/issues/1111) | Feature Request:是否可以增加同時一次性處理多個圖片的功能 | Feature requests/enhancements | none | Yes | Partial | Medium | Tracked for future pass. |
| [#1109](https://github.com/dmMaze/BallonsTranslator/issues/1109) | Bug 反馈:无法停止运行＆翻译出错导致文本框压缩 | Bugs/regressions/crashes | bug | Yes | Partial | Medium | Tracked for future pass. |
| [#1108](https://github.com/dmMaze/BallonsTranslator/issues/1108) | Bug 反馈: 拉小文本框自动放大 | Bugs/regressions/crashes | bug | Yes | Partial | Medium | Tracked for future pass. |
| [#1107](https://github.com/dmMaze/BallonsTranslator/issues/1107) | Feature Request:尊敬的dmMaze你好，和各位ballonstranslator的贡献者 | Translation/provider/model setup | none | Yes | Partial | Medium | Tracked for future pass. |
| [#1104](https://github.com/dmMaze/BallonsTranslator/issues/1104) | Bug 反馈: 發現更新後會出現閃退&加載表出現頂置狀態 | Bugs/regressions/crashes | bug | Yes | Partial | Medium | Tracked for future pass. |
| [#1100](https://github.com/dmMaze/BallonsTranslator/issues/1100) | Bug 反馈:无法编辑阿尔法通道的字体阴影 | Bugs/regressions/crashes | bug | Yes | Partial | Medium | Tracked for future pass. |
| [#1098](https://github.com/dmMaze/BallonsTranslator/issues/1098) | [Bug] Translation gets stuck (infinite loading) with DeepSeek API | Automation/API/headless/MCP | bug | Yes | Partial | Medium | Tracked for future pass. |
| [#1096](https://github.com/dmMaze/BallonsTranslator/issues/1096) | Bug 反馈: 我把3.10版python反安装后重新安装3.11版python 出现问题 | Bugs/regressions/crashes | bug | Yes | Partial | Medium | Tracked for future pass. |
| [#1095](https://github.com/dmMaze/BallonsTranslator/issues/1095) | Bug 反馈:更新后缺失库？ | Bugs/regressions/crashes | bug | Yes | Partial | Medium | Tracked for future pass. |
| [#1094](https://github.com/dmMaze/BallonsTranslator/issues/1094) | Feature Request:基于cmd命令批量翻译模式的批量翻译Python小程序 | Feature requests/enhancements | none | Yes | Partial | Medium | Tracked for future pass. |
| [#1093](https://github.com/dmMaze/BallonsTranslator/issues/1093) | Feature Request: Text Mesh Warp & Eraser (Masking) Tools | Feature requests/enhancements | none | Yes | Partial | Medium | Tracked for future pass. |
| [#1092](https://github.com/dmMaze/BallonsTranslator/issues/1092) | Bug Report: | Bugs/regressions/crashes | bug | Yes | Partial | Medium | Tracked for future pass. |
| [#1090](https://github.com/dmMaze/BallonsTranslator/issues/1090) | Feature Request:deepseekocr实现文本检测+ocr+颜色提取 | OCR/detection/inpainting | none | Yes | Partial | Medium | Tracked for future pass. |
| [#1089](https://github.com/dmMaze/BallonsTranslator/issues/1089) | Bug 反馈:彩云和百度的翻译器，在正确输出id和密钥后也无法实现，一直显示密钥是否正确，无法使用 | Bugs/regressions/crashes | bug | Yes | Partial | Medium | Tracked for future pass. |
| [#1088](https://github.com/dmMaze/BallonsTranslator/issues/1088) | Bug 反馈:cmd批量翻译模式下ocr问题 | OCR/detection/inpainting | bug | Yes | Partial | Medium | Tracked for future pass. |
| [#1083](https://github.com/dmMaze/BallonsTranslator/issues/1083) | Feature Request:paddle-vl一键包 | Feature requests/enhancements | none | Yes | Partial | Medium | Tracked for future pass. |
| [#1082](https://github.com/dmMaze/BallonsTranslator/issues/1082) | 問題:Ocr_paddle更新後無法開啟 | OCR/detection/inpainting | bug | Yes | Partial | Medium | Tracked for future pass. |
| [#1080](https://github.com/dmMaze/BallonsTranslator/issues/1080) | Bug 反馈: 检测到amdgpu时无法正确安装依赖 | Compatibility/dependency/platform issues | bug | Yes | Partial | Medium | Tracked for future pass. |
| [#1077](https://github.com/dmMaze/BallonsTranslator/issues/1077) | Feature Request: Option to automatically adjust the size of the text based on its text box' dimensions + more font sizes in the list box. | Text rendering / typography / fonts | none | Yes | Yes/advanced this pass | High | Advanced by smart-fit workflow/API, next issue selection, fallback diagnostics, and render manifest; direct upstream UI freeze patch reviewed/deferred pending Pro conflict audit. |
| [#1076](https://github.com/dmMaze/BallonsTranslator/issues/1076) | 问个问题，我将YSGYolo放在data/model中，没有显示新的检测器，是哪里搞错了吗？ | Translation/provider/model setup | none | Yes | Partial | Medium | Tracked for future pass. |
| [#1075](https://github.com/dmMaze/BallonsTranslator/issues/1075) | Bug 反馈:图片反色了 | Bugs/regressions/crashes | bug | Yes | Partial | Medium | Tracked for future pass. |
| [#1074](https://github.com/dmMaze/BallonsTranslator/issues/1074) | Bug Report: `Optional[str]): argument 1 has unexpected type 'dict'` | Bugs/regressions/crashes | bug | Yes | Partial | Medium | Tracked for future pass. |
| [#1073](https://github.com/dmMaze/BallonsTranslator/issues/1073) | Feature Request:希望可以對文本检测時的次序右到左的排列直接編輯 | Feature requests/enhancements | none | Yes | Partial | Medium | Tracked for future pass. |
| [#1072](https://github.com/dmMaze/BallonsTranslator/issues/1072) | Feature Request:请添加paddleocr-vl功能 | OCR/detection/inpainting | none | Yes | Partial | Medium | Tracked for future pass. |
| [#1071](https://github.com/dmMaze/BallonsTranslator/issues/1071) | Bug 反馈:oneocr行顺序错误 | OCR/detection/inpainting | bug | Yes | Partial | Medium | Tracked for future pass. |

## Next batch candidates

1. Native editable PSD text layer writer validation for #558/#587 and upstream export requests.
2. Batch lettering workflow over selected pages with progress/cancel UI.
3. Full provider/model onboarding wizard for OCR/inpaint/LLM setup diagnostics.
4. Visual regression proofs for Qt vertical punctuation painting at Windows DPI scales.
5. Direct audit of upstream #1170 replace-and-render freeze patch against Pro batch export/report code.

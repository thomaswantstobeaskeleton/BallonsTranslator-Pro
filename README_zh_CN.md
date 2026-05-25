> [!IMPORTANT]
> **如打算公开分享本工具的机翻结果，且没有有经验的译者进行过完整的翻译或校对，请在显眼位置注明机翻。**

<div align="center">

# BallonsTranslator-Pro

**AI 驱动的漫画/条漫翻译工具箱 —— [dmMaze/BallonsTranslator](https://github.com/dmMaze/BallonsTranslator) 的 Pro 分支，新增 50+ 模块与生产级工作流工具。**

[English](/README.md) | 简体中文 | [pt-BR](doc/README_PT-BR.md) | [Русский](doc/README_RU.md) | [日本語](doc/README_JA.md) | [Indonesia](doc/README_ID.md) | [Tiếng Việt](doc/README_VI.md) | [한국어](doc/README_KO.md) | [Español](doc/README_ES.md) | [Français](doc/README_FR.md)

[![GitHub Stars](https://img.shields.io/github/stars/thomaswantstobeaskeleton/BallonsTranslator-Pro?style=for-the-badge&logo=github&color=gold)](https://github.com/thomaswantstobeaskeleton/BallonsTranslator-Pro/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/thomaswantstobeaskeleton/BallonsTranslator-Pro?style=for-the-badge&logo=github&color=blue)](https://github.com/thomaswantstobeaskeleton/BallonsTranslator-Pro/network/members)
[![License](https://img.shields.io/github/license/thomaswantstobeaskeleton/BallonsTranslator-Pro?style=for-the-badge&color=green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg?style=for-the-badge)](docs/GPU_ACCELERATION.md)

</div>

<p align="center">
  <img src="doc/src/1111.png" width="100%">
</p>

## 项目简介

BallonsTranslator-Pro 是 [dmMaze/BallonsTranslator](https://github.com/dmMaze/BallonsTranslator) 的增强分支，专为**严肃的漫画/条漫汉化工作流**设计。在保留原版全部功能的基础上，新增：

- **90+ AI 模块** —— 20+ 文本检测器、30+ OCR 引擎、25+ 翻译器、15+ 修复器
- **生产级排版工具** —— 自动排版、溢出检测、形状感知安全区、密度感知字体缩放
- **实时屏幕翻译** —— 始终置顶的穿透式悬浮窗，可直接在阅读器上实时翻译
- **Translation Assist 面板** —— 逐块候选、术语表/记忆库、多引擎对比、QA 警告
- **批处理队列** —— 多文件夹队列，支持暂停/恢复/取消
- **自动化 API** —— 本地 REST API + MCP 风格命令面，支持无头/脚本化工作流
- **Docker / 服务端模式** —— 无需 GUI 即可运行，适合 CI 或远程部署

---

## 快速开始

| 系统 | 启动方式 |
|---|---|
| ![Windows](https://img.shields.io/badge/Windows-0078D6?logo=windows&logoColor=white) | 双击 `launcher.bat` 或 `launch_win.bat` |
| ![macOS](https://img.shields.io/badge/macOS-000000?logo=apple&logoColor=white) | `python launch.py` |
| ![Linux](https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black) | `python launch.py` |

```bash
# 或克隆后运行
git clone https://github.com/thomaswantstobeaskeleton/BallonsTranslator-Pro
cd BallonsTranslator-Pro
python launch.py
```

> GPU 配置帮助（AMD RX 9070/9060/7900、NVIDIA、Apple Silicon）：[docs/GPU_ACCELERATION.md](docs/GPU_ACCELERATION.md)

---

## 功能亮点

### 一键翻译流水线
<p align="center"><img src="doc/src/run.gif" width="80%"></p>

全链路自由组合 **90+ 模块**，告别一刀切默认方案。

### 实时屏幕翻译


始终置顶、鼠标穿透的悬浮窗。可选任意屏幕区域或跟随窗口。默认隐私优先：不保存、不记录。

### Translation Assist 面板


逐块候选建议、术语表查询、多提供商并排对比、块级 QA 警告。

### 生产级排版与自动布局
<p align="center"><img src="doc/src/multisel_autolayout.gif" width="80%"></p>

形状感知安全区、平衡换行、密度感知字体缩放、溢出安全检查。

### 批处理队列


多文件夹统一队列，支持暂停、恢复与取消。

---

## 模块目录

<details>
<summary><b>文本检测器（20+）</b> —— 点击展开</summary>

| 模块 | 类型 | 适用场景 |
|---|---|---|
| `ctd` (ComicTextDetector) | 内置 | 默认；日/英漫画 |
| `ysgyolo` | Pro | 过滤拟声词；lh5426 模型 |
| `animetext_yolo` | Pro | 复杂动漫场景（AnimeText 数据集） |
| `mangalens_bubble_segmentation` | Pro | 气泡分割 |
| `paddle_det` / `paddle_det_v5` | Pro | 中/英/日 |
| `pp_doclayout_v3` | Pro | 复杂版面、弯曲页面、混合栏 |
| `easyocr_det` | Pro | 多语言场景文本 |
| `craft_det` | Pro | 弯曲/方向文本 |
| `mmocr_det` | Pro | 文档/漫画文本（多边形） |
| `rapidocr_det` | Pro | 轻量 ONNX 检测 |
| `dptext_detr` | Pro | 动态点查询 |
| `hf_object_det` | Pro | 气泡/文本目标检测 |
| `hunyuan_ocr_det` | Pro | 全图 spotting |
| `magi_det` | Pro | 漫画分镜+阅读顺序（CVPR 2024） |
| `surya_det` | Pro | 90+ 语言，行级检测 |
| `sam_text_det` / `sam3_refiner` | Pro | SAM 提示分割 |
| `swintextspotter_v2` | Pro | 端到端 text spotting |
| `textmamba_det` | Pro | 弯曲文本（Mamba SSM） |
| `stariver_ocr` | Pro | 星河云 OCR |

</details>

<details>
<summary><b>OCR 引擎（30+）</b> —— 点击展开</summary>

| 模块 | 类型 | 适用场景 |
|---|---|---|
| `mit_32px` / `mit_48px` / `mit_48px_ctc` | 内置 | manga-image-translator 模型 |
| `manga_ocr` | 内置 | 日语漫画（kha-white） |
| `paddle_ocr` / `paddle_vl` | Pro | 中/英/日 |
| `easyocr_ocr` | Pro | 多语言 |
| `rapid_ocr` | Pro | 轻量 ONNX |
| `florence2_ocr` | Pro | Microsoft 视觉模型 |
| `got_ocr2` | Pro | 统一 plain/scene/formatted |
| `hunyuan_ocr` | Pro | 100+ 语言，spotting |
| `glm_ocr` | Pro | 0.9B 文档 OCR |
| `internvl2_ocr` / `internvl3_ocr` | Pro | 文档/图表 OCR |
| `lighton_ocr` | Pro | 1B 参数 OCR |
| `chandra_ocr` | Pro | 9B 文档 OCR（版面、表格、公式） |
| `deepseek_ocr` | Pro | 重量级文档 OCR |
| `docowl2_ocr` | Pro | 无 OCR 文档理解 |
| `bing_ocr` | Pro | Bing 图像 OCR API |
| `google_vision` | Pro | Google Cloud Vision |
| `google_lens_exp` | Pro | 实验性 Google Lens API |
| `macos_ocr` | Pro | Apple Vision（macOS 原生） |
| `callisto_ocr` / `qwen2vl_ocr` | Pro | 2B VLM OCR |
| `vlm_ocr` (通用 HF) | Pro | 任意 Hugging Face VLM |
| `donut` | Pro | 无 OCR 文档理解 |
| `llm_ocr` | Pro | LLM API OCR |
| `lens_proto` | Pro | Google Lens Protobuf |

</details>

<details>
<summary><b>翻译器（25+）</b> —— 点击展开</summary>

| 模块 | 类型 | 适用场景 |
|---|---|---|
| `google` | 内置 | 免费、快速 |
| `deepl` / `deeplx` / `deeplx_api` | Pro | 高质量 NMT |
| `sugoi` | Pro | 日译英（离线） |
| `sakura` | Pro | 日译中（Sakura-13B） |
| `chatgpt` / `chatgpt_exp` / `openai` | Pro | GPT-4 / GPT-3.5 |
| `gemini_neverliie` / `mistral_neverliie` | Pro | Gemini / Mistral（neverliie SDK） |
| `cohere_command_r` | Pro | Cohere Command R+ |
| `qwen_mt` | Pro | 阿里 Qwen 翻译 |
| `hy_mt_1_5_7b` | Pro | 腾讯 Hunyuan MT |
| `hunyuan_mt_chimera_7b` | Pro | 多源 ensemble + Chimera 精修 |
| `chimera` | Pro | 多源 ensemble |
| `ensemble` | Pro | 3 翻译器 + LLM 裁判 |
| `chain` | Pro | 链式翻译（如日→英→中） |
| `mbart50` | Pro | Meta 50 语言 NMT |
| `nllb200` | Pro | Meta 200 语言 NMT |
| `opus_mt` | Pro | Helsinki NLP 按语言对模型 |
| `t5_mt` | Pro | 基于提示的 T5 翻译 |
| `m2m100` / `m2m100_hf` | Pro | Meta 多对多翻译 |
| `manual` | Pro | JSON 提示词工作流 |
| `llm_api_translator` | Pro | 通用 LLM API |
| `eztrans` | Pro | 韩语游戏翻译 |
| `text-generation-webui` | Pro | 本地 TGW 后端 |
| `translatorspack` | Pro | 聚合（google、bing、baidu 等） |
| `caiyun` / `baidu` / `papago` | Pro | 中文云 API |

</details>

<details>
<summary><b>图像修复器（15+）</b> —— 点击展开</summary>

| 模块 | 类型 | 适用场景 |
|---|---|---|
| `aot` | 内置 | manga-image-translator 默认 |
| `patchmatch` | 内置 | 非深度学习（类 PS） |
| `lama_mpe` | 内置 | 微调 LaMa |
| `cuhk_manga_inpaint` | Pro | CUHK Seamless Manga（SIGGRAPH 2021） |
| `lama_onnx` / `lama_manga_onnx` | Pro | ONNX LaMa（通用/漫画） |
| `simple_lama` | Pro | pip 安装 LaMa |
| `mat` | Pro | Mask-Aware Transformer（CVPR 2022） |
| `opencv-tela` / `opencv-classic` | Pro | OpenCV 修复 |
| `diffusers_sd_inpaint` / `diffusers_sd2_inpaint` / `diffusers_sdxl_inpaint` | Pro | Stable Diffusion 系列 |
| `dreamshaper_inpaint` | Pro | DreamShaper 8 |
| `kandinsky_inpaint` | Pro | Kandinsky 2.1 |
| `fluently_v4_inpaint` | Pro | 动漫/漫画调优 |
| `repaint` | Pro | DDPM 修复 |
| `flux_fill` | Pro | FLUX.1 Fill（12B，高质量） |
| `qwen_image_edit` | Pro | Qwen-Image-Edit 语义填充 |

</details>

---

## Pro 与原版对比一览

| 功能 | 原版 | Pro |
|---|---|---|
| 文本检测器 | 3 个 | **20+** |
| OCR 引擎 | 5 个 | **30+** |
| 翻译器 | 10 个 | **25+** |
| 图像修复器 | 3 个 | **15+** |
| 实时屏幕悬浮窗 | 无 | **有** |
| Translation Assist 面板 | 无 | **有** |
| 批处理队列（多文件夹） | 无 | **有** |
| 版式审阅与自动排版 | 基础 | **高级** |
| 自动化 API / 无头模式 | 无 | **有** |
| Docker / 服务端模式 | 无 | **有** |
| 应用内 Google Fonts 安装 | 无 | **有** |
| 环境诊断工具 | 无 | **有** |
| 模型管理器 | 无 | **有** |
| PSD 导出 | 无 | **有** |
| InDesign LPTXT 工作流 | 无 | **有** |

---

## 基础工作流

1. **打开页面** —— `File → Open Folder / Open Images`
2. **选择模块** —— 挑选检测器 + OCR + 翻译器 + 修复器
3. **运行** —— 一键执行完整流水线
4. **检查与编辑** —— 富文本编辑器，支持撤销/重做、多选、样式预设
5. **导出** —— `Tools → Export all pages`（PNG、CBZ、PSD、InDesign LPTXT）

---

## 系统要求

- **Python** 3.10.2+（避免 3.10.0/3.10.1 —— PyInstaller 崩溃 bug）
- **网络** 首次安装及下载模型需要联网
- **磁盘** 约 10–50 GB，视所选模型而定
- **GPU** 可选（CUDA、ROCm、Apple Silicon 或纯 CPU）

```bash
python --version
```

---

## 文档导航

| 入门与运行 | 质量与排版 | 自动化与规划 |
|---|---|---|
| [TROUBLESHOOTING](docs/TROUBLESHOOTING.md) | [QUALITY_RANKINGS](docs/QUALITY_RANKINGS.md) | [LOCAL_AUTOMATION_API](docs/LOCAL_AUTOMATION_API.md) |
| [GPU_ACCELERATION](docs/GPU_ACCELERATION.md) | [MODELS_REFERENCE](docs/MODELS_REFERENCE.md) | [REALTIME_TRANSLATION_MODE_PLAN](docs/REALTIME_TRANSLATION_MODE_PLAN.md) |
| [DOCS_HIGHLIGHTS](docs/DOCS_HIGHLIGHTS.md) | [TRANSLATION_CONTEXT_AND_GLOSSARY](docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md) | [TRANSLATION_ASSIST_PLAN](docs/TRANSLATION_ASSIST_PLAN.md) |
| | [INDESIGN_LPTXT_WORKFLOW](docs/INDESIGN_LPTXT_WORKFLOW.md) | [FEATURE_PARITY_MATRIX](docs/FEATURE_PARITY_MATRIX.md) |

- 完整更新历史：[docs/CHANGELOG.md](docs/CHANGELOG.md)

---

## 漫画/生肉下载器

BallonsTranslator-Pro 内置基于注册表的漫画源下载器，支持 MangaDex、Comick、GOMANGA、MangaForFree、ToonGod、Mangakakalot、NaruRaw 等。可通过 `utils/manga_sources/provider_base.py` 添加新源，无需重建 UI。

- `docs/RAW_SOURCE_PROVIDER_EXPANSION_PLAN.md` —— 路线图
- `docs/MANGA_SOURCE_PLUGIN_API.md` —— 提供者接口

```bash
python scripts/test_manga_sources.py
pytest -q tests/test_manga_provider_base.py
```

---

## 许可证

[GPL-3.0](LICENSE) —— Forked from [dmMaze/BallonsTranslator](https://github.com/dmMaze/BallonsTranslator).

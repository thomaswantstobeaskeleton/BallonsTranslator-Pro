> [!IMPORTANT]  
> **如打算公开分享本工具的机翻结果，且没有有经验的译者进行过完整的翻译或校对，请在显眼位置注明机翻。**

# BallonsTranslator-Pro
简体中文 | [English](/README_EN.md) | [pt-BR](doc/README_PT-BR.md) | [Русский](doc/README_RU.md) | [日本語](doc/README_JA.md) | [Indonesia](doc/README_ID.md) | [Tiếng Việt](doc/README_VI.md) | [한국어](doc/README_KO.md) | [Español](doc/README_ES.md) | [Français](doc/README_FR.md)

BallonsTranslator-Pro 是 [dmMaze/BallonsTranslator](https://github.com/dmMaze/BallonsTranslator) 的增强分支，面向严肃漫画/条漫翻译工作流。


从整体流程看，BallonsTranslator-Pro 可以帮助你：

1. 检测气泡/文本区域
2. OCR 识别文字
3. 翻译文本
4. 修复原图文字区域（inpaint）
5. 编辑并导出页面

- 完整更新历史: [docs/CHANGELOG.md](docs/CHANGELOG.md)

---

## 快速启动（先点什么）

克隆/下载并打开项目目录后：

- **Windows（推荐）：** 双击 `launcher.bat`，使用一个菜单完成启动、设置、自动更新、AMD/NVIDIA/CPU GPU 模式。
- **Windows 快速启动：** 双击 `launch_win.bat`，使用自动 GPU 检测立即启动。
- **跨平台（手动）：** 终端运行 `python launch.py`。
- **GPU 帮助：** 查看 [docs/GPU_ACCELERATION.md](docs/GPU_ACCELERATION.md)，尤其是 AMD Radeon RX 9070/9060/7900 用户。

如果 Windows SmartScreen 阻止 `.bat`，可右键批处理文件 → **以管理员身份运行**（或选择“更多信息 → 仍要运行”）。

---

## 安装 Google Fonts（应用内原生）

现在可直接在应用内安装 Google Fonts：

1. 打开 **工具 → 模型 → 安装 Google Font...**
2. 输入字体家族名（例如：`Bangers`、`Noto Sans JP`、`Comic Neue`）
3. 程序会自动下载并注册字体

已安装字体会保存到 `fonts/google/`，并自动出现在字体选择器中。

---

## 克隆 / 下载方式

### 方式 A — 下载 ZIP（最简单）

1. 打开 GitHub 仓库页面。
2. 点击 **Code** → **Download ZIP**。
3. 解压到普通目录（例如：`Documents/BallonsTranslator-Pro`）。
4. 打开该目录。
5. Windows 用 `launch_win.bat` 启动；或终端运行 `python launch.py`。

### 方式 B — Git clone（终端）

```bash
git clone https://github.com/thomaswantstobeaskeleton/BallonsTranslator-Pro
cd BallonsTranslator-Pro
python launch.py
```

### 方式 C — GitHub Desktop

1. 安装 GitHub Desktop。
2. **File → Clone repository**。
3. 粘贴仓库 URL 并选择本地路径。
4. 打开本地目录。
5. 启动 `launch_win.bat`（Windows）或 `python launch.py`。

---

## 运行要求

- Python 3.10+
- 首次安装/下载模型需要联网
- 需为模型预留足够磁盘空间

检查 Python：

```bash
python --version
```

---

## 为什么是 Pro

- 面向真实项目的模块组合
- 更好的批处理与长章节流程支持
- 基于 LLM 的审校链（质量/一致性）
- 默认配置更友好，上手更快
- 提供本地自动化/API 文档，便于无头模式与 MCP 风格工作流（见 [Local Automation API](docs/LOCAL_AUTOMATION_API.md)）
- Docker/服务器模式快速说明与示例 curl/Python 客户端也在同一文档中。

---

## 功能总览（快速索引）

### 核心生产流程

- **流程模块**：检测器、OCR、翻译器、修复器可自由组合，并可叠加排版/版式优化步骤。
- **版式审阅与自动排版**：内置 Layout Review、自动适配/原子气泡适配、溢出检测、阅读顺序辅助与排版 QA，适合正式汉化流程。
- **文本编辑能力**：支持富文本气泡编辑、撤销/重做、多选变换、查找替换、样式预设、形变/路径控制，以及按块 OCR/翻译检查。
- **质量优先后处理**：提供块级 QA 提示、审阅导向工具和一致性辅助，帮助减少明显翻译/排版回归。

### 翻译质量能力栈

- **Translation Assist（测试版）**：按文本块提供 TM/术语表/SFX/语料候选，支持显式应用、先编辑后应用、加入 TM/术语表、块级 QA、提供商遥测显示与缓存控制。
- **多提供商/模块对比工作流**：支持按预设（`low_latency`、`high_quality`）与范围（`translator`、`ocr`、`detector`、`inpainter`）进行同面板对比。
- **提示词档位与偏好控制**：可按项目配置 Assist 提示词档位与提供商偏好，便于可控质量调优。
- **质量文档配套**：质量与模型参考可见 [docs/QUALITY_RANKINGS.md](docs/QUALITY_RANKINGS.md) 与 [docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md](docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md)。

### 实时、自动化与工程能力

- **实时屏幕翻译（实验性）**：无需建项目，直接对选定屏幕区域实时 OCR+翻译（工具菜单进入），默认隐私优先。
- **自动化 API**：提供本地路由发现（`/routes`）与 MCP 风格命令面，便于脚本化和无头控制（含 realtime / translation-assist 命名空间）。
- **批处理与导出**：批队列、归档/CBZ 流程、结构化 OCR/翻译互换、校对/交付导出链路。
- **生肉下载器**：基于注册表的源插件体系，保留合规/安全边界并支持后续扩展。
- **诊断与排障**：环境体检、启动诊断、可选依赖说明、GPU/运行时排障指引。

### 平台与易用性亮点

- **Windows 优先启动体验**：`launcher.bat` / `launch_win.bat` 一键流程覆盖安装与快速启动。
- **应用内 Google Fonts 安装**：无需离开程序即可下载并注册字体。
- **模块兼容弹性**：可根据资源/隐私/成本选择本地离线或云端 OCR/MT/LLM 组件。

---

## 基础工作流

1. **打开页面**：File → Open Folder / Open Images
2. **选择模块**：Detector / OCR / Inpaint / Translator
3. 点击 **Run**
4. 检查并编辑文本块
5. 在 **Tools → Export all pages** 导出

---

## 仓库内常用路径

- 启动入口：`launch.py`
- Windows 一键启动：`launch_win.bat`
- 配置模板：`config/config.example.json`
- 当前配置：`config/config.json`
- 模型目录：`data/models/`
- 字体目录：`fonts/`

---

## 文档（按场景）

### 入门与运行
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
- [docs/GPU_ACCELERATION.md](docs/GPU_ACCELERATION.md)
- [docs/DOCS_HIGHLIGHTS.md](docs/DOCS_HIGHLIGHTS.md) — 一页式文档导览，说明每份核心文档的用途与适用场景。

### 质量、翻译与排版
- [docs/QUALITY_RANKINGS.md](docs/QUALITY_RANKINGS.md) — 常见模块组合的质量/速度取舍参考。
- [docs/MODELS_REFERENCE.md](docs/MODELS_REFERENCE.md) — 模型/模块能力与选型参考。
- [docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md](docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md) — 上下文与术语表组织策略、一致性建议。
- [docs/INDESIGN_LPTXT_WORKFLOW.md](docs/INDESIGN_LPTXT_WORKFLOW.md) — 面向专业后期排版的交接流程说明。

### 自动化、实时与规划
- [docs/LOCAL_AUTOMATION_API.md](docs/LOCAL_AUTOMATION_API.md) — 本地自动化路由与客户端示例。
- [docs/REALTIME_TRANSLATION_MODE_PLAN.md](docs/REALTIME_TRANSLATION_MODE_PLAN.md) — 实时模式分期与当前约束。
- [docs/TRANSLATION_ASSIST_PLAN.md](docs/TRANSLATION_ASSIST_PLAN.md) — Translation Assist 能力清单与路线图。
- [docs/ALTERNATIVES_FEATURE_GAP_IMPLEMENTATION_PLAN.md](docs/ALTERNATIVES_FEATURE_GAP_IMPLEMENTATION_PLAN.md) — 功能差距计划与阶段记录。
- [docs/FEATURE_PARITY_MATRIX.md](docs/FEATURE_PARITY_MATRIX.md) — 跨功能区的对齐/覆盖状态矩阵。


## Manga / Comic Raw Downloader

BallonsTranslator-Pro includes a registry-backed manga/raw source downloader. Existing sources (MangaDex, Comick, GOMANGA, Manhwa Reader, MangaForFree, ToonGod, Mangakakalot, NaruRaw, ManhwaRaw, 1kkk, generic chapter URLs, and local folders) are preserved, and new providers can be added through `utils/manga_sources/provider_base.py` plus `utils/manga_sources/registry.py` without manually rebuilding the UI source list.

Current first-batch registry providers include MangaSee/MangaSee123, ReadManga-style pages, and Manganato/Manganelo compatibility. Providers must use public pages/APIs only, respect request delays, and must not bypass DRM, paywalls, login restrictions, private APIs, CAPTCHA, Cloudflare, or other access controls. Browser rendering remains the explicit user-controlled Playwright option.

Developer docs:

- `docs/RAW_SOURCE_PROVIDER_EXPANSION_PLAN.md` — audit and roadmap.
- `docs/MANGA_SOURCE_PLUGIN_API.md` — provider interface, registry metadata, legal/ToS expectations, and tests.
- `docs/EXTERNAL_SOURCE_AUDIT.md` — generated local audit report for optional external downloader repositories.

Smoke tests:

```bash
python scripts/test_manga_sources.py
python scripts/test_manga_sources.py --stable-registry-only
pytest -q tests/test_manga_provider_base.py
```

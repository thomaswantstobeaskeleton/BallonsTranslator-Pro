# BallonsTranslator-Pro

基于 [BallonsTranslator](https://github.com/dmMaze/BallonsTranslator) 的社区分支，面向漫画翻译流程，提供更丰富的检测/OCR/修复可选模块与批处理工具。

- English README (same structure): [README.md](README.md)
- 完整更新历史: [docs/CHANGELOG.md](docs/CHANGELOG.md)

**What’s new (short):** v1.7.0 重点更新自动排版行为与首次运行模型下载（主界面先启动、后台下载）。详细内容见 [docs/CHANGELOG.md](docs/CHANGELOG.md)。

## 模块分级与兼容矩阵

| 分级 | 含义 |
|---|---|
| **Stable** | 默认与首跑预设优先使用，验证最充分。 |
| **Beta** | 大多数场景可用，但跨环境验证少于 Stable。 |
| **Experimental** | 新功能/占位实现/高波动模块。 |
| **External dependency heavy** | 依赖较重（大模型、额外仓库或外部服务）。 |

统一兼容矩阵请见：**[docs/MODULE_COMPATIBILITY_MATRIX.md](docs/MODULE_COMPATIBILITY_MATRIX.md)**。

## 简介
## Overview

BallonsTranslator-Pro 保留上游 detect → OCR → translate → inpaint 的核心流程，并扩展了可选模块、模型管理和漫画图源能力。

另见：
- 故障排查: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
- 模型参考: [docs/MODELS_REFERENCE.md](docs/MODELS_REFERENCE.md)
- 质量分级: [docs/QUALITY_RANKINGS.md](docs/QUALITY_RANKINGS.md)

## Install

1. 克隆仓库：
   ```bash
   git clone https://github.com/thomaswantstobeaskeleton/BallonsTranslator-Pro.git
   cd BallonsTranslator-Pro
   ```
2. 直接启动（首次运行会自动安装基础依赖）：
   ```bash
   python launch.py
   ```
3. 可选安装脚本：
   - Windows: `setup.bat`
   - Linux/macOS: `./setup.sh`

## First Run

### Quick start (5 steps) / 快速开始（5 步）

1. **Run app / 启动应用**: `python launch.py`
2. **Pick model package / 选择模型包**: 建议先选 Core only，启动更快。
3. **Open pages / 打开页面**: 文件 → 打开文件夹（或打开图片）。
4. **Select modules / 选择模块**: 设置面板 → 检测/OCR/修复/翻译。
5. **Run pipeline / 运行流程**: 点击底栏 **Run**。

### First-run model picker (explanation) / 首次运行模型选择器说明

- Trigger condition / 触发条件: 当 `config/config.json` 不存在（全新安装）时出现。
- Behavior / 行为:
  - 主窗口先打开。
  - 选中的模型包在后台下载。
  - 若下载失败，使用 **工具 → 模型 → 重试下载模型**。
- Paths / 路径:
  - 配置模板: `config/config.example.json`
  - 运行配置: `config/config.json`
  - 模型下载目录: `data/models/`
  - HF 缓存（可选清理）: `.btrans_cache/hub`
- Screenshot references / 截图路径:
  - UI 设置面板示例: `doc/src/configpanel.png`
  - 本地补充首次模型选择器截图建议路径: `docs/images/first_run_model_picker.png`

## Model Packages

首次启动后可通过 **Tools → Manage models** 增删可选模型包。

建议基础组合：
- Detector: `ctd`
- OCR: `manga_ocr`
- Inpaint: `aot`
- Translator: `google`

推荐配置见：
- [docs/MANHUA_BEST_SETTINGS.md](docs/MANHUA_BEST_SETTINGS.md)
- [docs/QUALITY_RANKINGS.md](docs/QUALITY_RANKINGS.md)

## Startup Scripts

- `python launch.py`: 主入口。
- `setup.bat` / `setup.sh`: 一次性环境初始化。
- `Launch BallonsTranslator.bat`（Windows）: 使用仓库 venv 的 Python 启动。

提示：便携迁移时，建议保持 `data/` 与 `config/` 在 `launch.py` 同级目录。

## Core Workflows

1. **Project workflow**: 打开文件夹/图片 → 检测/OCR/翻译/修复 → 保存项目。
2. **Batch workflow**: Tools → Batch queue 批量处理多文件夹。
3. **Source workflow**: Tools → Manga/Comic source 搜索和下载章节。
4. **Export workflow**: Tools → Export all pages（含 PDF 导出）。

| 阶段 | 推荐 | 分级 | 关键设置 |
|------|------|------|----------|
| **检测** | CTD | **Stable** | detect_size 1280，box score 0.42–0.48，box_padding 4–6 |
| **OCR** | Surya OCR | **Beta** | 语言：简体中文，Fix Latin misread：True，crop_padding 6–8 |
| **修复** | lama_large_512px | **Beta** | mask_dilation 1–2，inpaint_size 1024 |
相关文档：
- 翻译上下文/术语表: [docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md](docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md)
- InDesign LPtxt 导出: [docs/INDESIGN_LPTXT_WORKFLOW.md](docs/INDESIGN_LPTXT_WORKFLOW.md)

## Troubleshooting

主文档： [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

常见问题方向：
- GPU/显存限制与降级方案
- 可选依赖冲突
- HuggingFace/网络下载失败
- OCR/文本框对齐调参

## Contribution

- 先阅读: [CONTRIBUTING.md](CONTRIBUTING.md)
- 提交 issue 时请附可复现信息（系统、GPU/CPU、模块名、日志、示例页）。
- 保持提交小而聚焦。

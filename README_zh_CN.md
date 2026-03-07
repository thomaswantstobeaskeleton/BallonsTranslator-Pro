# BallonsTranslatorPro

**仓库：** [https://github.com/thomaswantstobeaskeleton/BallonsTranslator-Pro](https://github.com/thomaswantstobeaskeleton/BallonsTranslator-Pro) · **版本：** 1.7.0

基于 [BallonsTranslator](https://github.com/dmMaze/BallonsTranslator) 的社区分支，在保留原有行为与默认设置的前提下增加多项功能。

---

## 简介

| 项目 | 说明 |
|------|------|
| **功能** | 20+ 文本检测器（含双检测）、30+ OCR 引擎、15+ 图像修复模型、翻译上下文与术语表、文字橡皮擦、批量队列、**漫画/图源**（MangaDex 含生肉/原语言、GOMANGA、Manhwa Reader、Comick、MangaForFree、ToonGod、Mangakakalot、NaruRaw、ManhwaRaw、1kkk、通用章节 URL、本地文件夹）、**批量导出 PDF**、**重复/重叠块检查**、370+ 字体。完整文档与推荐设置见英文 README。 |
| **上游** | [BallonsTranslator](https://github.com/dmMaze/BallonsTranslator) — 原项目 |
| **参与** | 可作为独立实验分支合并回上游，见 [CONTRIBUTING.md](CONTRIBUTING.md)。 |

**界面语言：** 在 **视图 → 帮助 → 界面语言** 中选择 **简体中文** 或 **English**。本仓库提供完整中文界面翻译。

---

## 快速开始

1. **克隆并运行：**  
   `git clone https://github.com/thomaswantstobeaskeleton/BallonsTranslator-Pro.git && cd BallonsTranslator-Pro && python launch.py`

2. **首次运行：**  
   自动安装基础依赖。若不存在 `config.json`（全新安装），会弹出 **模型包选择** 对话框：选择要下载的模型包（仅核心包，或核心 + 高级 OCR / 高级修复等）。**主窗口会立即打开**，模型包在后台下载；若下载失败或中断，无需重启，使用 **工具 → 模型 → 重试下载模型** 即可。下载成功后，检测器/OCR/修复/翻译会恢复为核心默认（ctd、manga_ocr、aot、google）。之后可通过 **工具 → 管理模型** 下载更多模型。若已有 `config.json`，则按当前配置下载；缺省仅下载核心包，避免占用多余磁盘（见 [Issue #15](https://github.com/thomaswantstobeaskeleton/BallonsTranslator-Pro/issues/15)）。

3. **配置：** 打开设置面板 → 在 **文本检测**、**OCR**、**图像修复**、**翻译** 下拉框中选择模块。

4. **新模块** 会自动出现；仅需安装你实际使用的模块依赖。

5. **更新：** 使用 **视图 → 帮助 → 从 GitHub 更新** 拉取最新代码，本地配置与数据不会被覆盖。*仅在使用 git 克隆时可用；若通过 ZIP 下载，请重新下载最新 ZIP 替换文件夹。*

---

## 便携 / 一键安装

- **Windows：** 可运行 `setup.bat` 创建 venv 并安装依赖，或直接运行 `python launch.py`（会自动安装基础依赖与 PyTorch）。
- **Linux / macOS：** 运行 `./setup.sh` 或 `bash setup.sh`。
- **Torch：** `launch.py` 会自动检测 GPU 并安装对应 PyTorch（NVIDIA CUDA、AMD ROCm 或 CPU）。可通过环境变量 **TORCH_COMMAND** 覆盖。
- **字体：** 将 `.ttf` / `.otf` / `.ttc` / `.pfb` 放入 **`fonts/`** 目录，启动时自动加载。
- **模型：** 默认下载到 **`data/`**（如 `data/models/`）。可整体复制 `data/` 和 `config/` 做备份或迁移。
- **故障排除：** 见 **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)**（GPU OOM、HuggingFace 门控模型、API 密钥、依赖冲突等）。

---

## 功能概览

- **文本检测：** CTD、Paddle、EasyOCR、YSG（淫书馆）/ 其他 YOLO、HF 目标检测、MMOCR、Surya、Magi、CRAFT、DPText-DETR 等 20+ 种；多数支持 **框 padding** 4–6 px 减少裁字。
- **OCR：** Paddle、manga_ocr、Surya、TrOCR、GOT-OCR2、Ocean、InternVL2/3、HunyuanOCR 等 30+ 种；多种支持 **裁剪 padding**。
- **图像修复：** lama_large_512px（可调 mask 膨胀）、Simple LaMa、Diffusers、LaMa ONNX、MAT、Fluently v4 等。
- **翻译上下文：** 术语表、前页上下文、系列级存储；LLM_API_Translator 支持接近长度限制时的 **上下文摘要**。
- **界面：** 画布右键 30+ 操作（复制/粘贴、合并、拼写检查、文字橡皮擦、流程阶段等）；**漫画/图源** 多站点搜索与下载；**批量导出 PDF**；**检查项目**（缺失文件、无效 JSON、重叠块）；可自定义快捷键与关键词替换（OCR、机翻前/后）。
- **配置面板：** 逻辑 DPI、深色模式、界面语言、WebP 无损、排版默认、双文本检测（主 + 副检测器）。

**YSG（淫书馆）说明：** 仅指 [YSGforMTL/YSGYoloDetector](https://huggingface.co/YSGforMTL/YSGYoloDetector) 系列模型，由作者 lhj5426 独立开发（数据标注至训练约 19 个月）。其他 YOLO 模型（如 ogkalu 漫画气泡检测）不属于「YSG 系列」，请勿混淆。详见 [docs/MODELS_REFERENCE.md](docs/MODELS_REFERENCE.md)。

---

## 国漫 / 简体中文推荐设置

| 阶段 | 推荐 | 关键设置 |
|------|------|----------|
| **检测** | CTD | detect_size 1280，box score 0.42–0.48，box_padding 4–6 |
| **OCR** | Surya OCR | 语言：简体中文，Fix Latin misread：True，crop_padding 6–8 |
| **修复** | lama_large_512px | mask_dilation 1–2，inpaint_size 1024 |

更多质量分级与设置见 [docs/MANHUA_BEST_SETTINGS.md](docs/MANHUA_BEST_SETTINGS.md)、[docs/QUALITY_RANKINGS.md](docs/QUALITY_RANKINGS.md)。

---

## 免责声明

本分支新增大量可选模块（检测、OCR、修复），**并非全部在每种环境**（Windows/Linux/macOS、CPU/CUDA、所有语言对）下都经过完整测试。可能遇到依赖冲突、显存不足（OOM）或模型特有问题。遇到问题时请尝试同类别其他模块，或提交 Issue 并注明系统、设备与配置。已知依赖冲突见 **docs/OPTIONAL_DEPENDENCIES.md**。

---

## 更多文档

- **完整英文说明与教程：** [README.md](README.md)
- **故障排除：** [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
- **翻译上下文与术语表：** [docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md](docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md)
- **模型参考与质量排名：** [docs/MODELS_REFERENCE.md](docs/MODELS_REFERENCE.md)、[docs/QUALITY_RANKINGS.md](docs/QUALITY_RANKINGS.md)

> [!IMPORTANT]  
> **如打算公开分享本工具的机翻结果，且没有有经验的译者进行过完整的翻译或校对，请在显眼位置注明机翻。**

# BallonsTranslator-Pro

BallonsTranslator-Pro 是 [dmMaze/BallonsTranslator](https://github.com/dmMaze/BallonsTranslator) 的增强分支，面向严肃漫画/条漫翻译工作流。


从整体流程看，BallonsTranslator-Pro 可以帮助你：

1. 检测气泡/文本区域
2. OCR 识别文字
3. 翻译文本
4. 修复原图文字区域（inpaint）
5. 编辑并导出页面

- English README（与本页同结构）: [README.md](README.md)
- 完整更新历史: [docs/CHANGELOG.md](docs/CHANGELOG.md)

---

## 快速启动（先点什么）

克隆/下载并打开项目目录后：

- **Windows（推荐）：** 双击 `launch_win.bat`
- **Windows（夜版更新 + 启动）：** 双击 `launch_win_amd_nightly.bat`
- **Windows（启动 + 自动更新）：** 双击 `launch_win_with_autoupdate.bat`
- **跨平台（手动）：** 终端运行 `python launch.py`

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

## 文档

- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
- [docs/QUALITY_RANKINGS.md](docs/QUALITY_RANKINGS.md)
- [docs/MODELS_REFERENCE.md](docs/MODELS_REFERENCE.md)
- [docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md](docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md)
- [docs/INDESIGN_LPTXT_WORKFLOW.md](docs/INDESIGN_LPTXT_WORKFLOW.md)

# 硬编码 UI 文本审计（未进入 `.ts`/`.qm` 翻译链路）

本审计用于定位当前在 Python 代码中硬编码、未走 Qt 翻译机制的用户可见文案。

## 范围与方法

- 扫描 `ui/*.py` 中常见 Qt 文本 API（`QLabel`、`QPushButton`、`QCheckBox`、`QGroupBox`、`setText`、`setToolTip`、`setPlaceholderText` 等）的字符串字面量。
- 重点关注用户可见英文 UI 文案。
- 纯日志文本不作为 UI 翻译阻塞项。

## 高优先级硬编码位置

### `ui/merge_dialog.py`
存在较多英文硬编码（窗口标题、分组标题、按钮、占位符、提示文案等），应统一改为 `self.tr(...)`。

### `ui/mainwindowbars.py`
品牌文本 `BallonsTranslatorPro`：若作为品牌名可保留，否则建议将周边说明文案本地化。

### `ui/configpanel.py`
组合框可见枚举值存在英文字面量：`never`、`all_pages`、`current_page`。建议“显示文本本地化 + 内部值稳定映射”拆分。

### `ui/subtitle_file_translator_dialog.py`
格式项中存在字面量键值（`auto`/`srt`/`txt`），若用户可见应提供本地化标签。

### `ui/video_translator_dialog.py`
若干占位符是英文字面量（语言示例、编码器、API key 示例等）。
- 技术 token 可保留；
- 说明性占位符建议本地化。

## 为什么重要

只要用户可见文案没有经过 `self.tr(...)`，切换中文后就会出现中英混杂，影响“完整中文界面”的一致性目标。

## 建议修复顺序

1. 先完成 `ui/merge_dialog.py`（硬编码面最大）。
2. 规范 `ui/configpanel.py` 与 `ui/subtitle_file_translator_dialog.py` 的“显示值/内部值”分离。
3. 逐项审查 `ui/video_translator_dialog.py` 占位符并本地化说明性文本。
4. 在具备 Qt Linguist 工具（`lrelease`）环境中更新 `.ts` 并编译 `.qm`。

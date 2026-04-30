# 视频翻译：字幕导出（SRT / ASS / VTT）

本文说明项目如何导出外挂字幕，以兼容常见播放器、YouTube 与 FFmpeg。  
关于视频编码（FPS、码率、兼容性）请见：`VIDEO_TRANSLATOR_FPS_AND_BITRATE_zh_CN.md`。

## 业界常见做法（参考）

- `subtitle-translator`：支持 SRT/ASS 与批量导出。
- `SubZilla` / `subtitle_utils`：统一 UTF-8 编码、格式转换。
- `pyVideoTrans`：SRT 合并/封装，MP4 常见 `mov_text + language`。
- FFmpeg / SubRip 规范：
  - SRT 时间戳毫秒使用逗号（`,`）
  - WebVTT 使用点号（`.`）
  - ASS 在定位场景建议写 PlayResX/PlayResY。

## 本项目当前实现

### 1) 文件编码

- 所有字幕文件均使用 **UTF-8** 写出，保证中日韩等多语言兼容。

### 2) SRT

- 时间格式：`HH:MM:SS,mmm`（毫秒分隔符为逗号）。
- 结构：序号 → 时间区间 → 文本 → 空行。
- 文本换行会被替换为空格，保证单条字幕单行输出。
- 用途：外挂导出；以及在“Mux SRT into video”开启时作为软字幕封装入 MP4。

### 3) WebVTT

- 时间格式：`HH:MM:SS.mmm`（毫秒分隔符为点号）。
- 文件头：`WEBVTT` + 空行。
- Cue 布局与 SRT 类似（序号、时间段、文本）。

### 4) ASS

- Script Info：若有视频宽高，会写入 `PlayResX/PlayResY`，使缩放与定位更稳定。
- 默认样式：Arial、字号 20、白字、描边、底部居中（alignment=2）。
- 转义规则：
  - `\` → `\\`
  - `{` → `{{`
  - `}` → `}}`
  - 文本换行写为 `\N`
- 时间戳格式：`H:MM:SS.cc`（厘秒）。

### 5) MP4 封装 SRT（软字幕）

当启用“仅修复（不烧录）+ 封装 SRT 到视频”时，使用 FFmpeg 以 `mov_text` 写入字幕流，并设置 `language=eng` 元数据，提升播放器识别兼容性。

## 参考

- SubRip（SRT）：<https://en.wikipedia.org/wiki/SubRip>
- WebVTT：<https://www.w3.org/TR/webvtt1/>
- ASS：<https://wiki.videolan.org/SubStation_Alpha/>
- FFmpeg 字幕实践：<https://www.ffmpeg.media/articles/subtitles-burn-in-soft-subs-format-conversion>
- pyVideoTrans 字幕说明：<https://pyvideotrans.com/en/blog/ffmpeg-subtitles>

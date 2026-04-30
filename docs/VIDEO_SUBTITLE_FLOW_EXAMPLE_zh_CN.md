# 视频字幕流示例（中文）

本文给出视频翻译器在“检测 → OCR/ASR → 翻译 → 输出字幕”流程中的典型数据流，帮助排查字幕时序、断句和对齐问题。

## 典型流程

1. 读取输入视频并逐帧处理。
2. 根据配置选择：
   - 视觉字幕路径（检测+OCR）；或
   - 音频字幕路径（ASR）。
3. 生成时间片段（cue）：`start/end/text`。
4. 将 cue 批量送入翻译器（可按 chunk 分批）。
5. 将翻译结果写回 cue。
6. 导出 SRT / ASS / VTT，或烧录回视频。

## 最常见的错位来源

- FPS 与时长推导不一致（建议优先 ffprobe）。
- 断句策略不同导致 cue 切分变化（标点合并 / LLM 断句）。
- 翻译返回行数与输入行数不一致（需做数量校验/回退）。

## 排查建议

- 先导出 SRT，使用播放器核对时轴是否正确。
- 再开启烧录，检查是否只是样式/换行问题。
- 对长视频先用较小样本区间验证配置，再全量运行。

## 相关文档

- `VIDEO_SUBTITLE_EXPORT_zh_CN.md`
- `VIDEO_TRANSLATOR_FPS_AND_BITRATE_zh_CN.md`
- `TRANSLATION_CONTEXT_AND_GLOSSARY.md`

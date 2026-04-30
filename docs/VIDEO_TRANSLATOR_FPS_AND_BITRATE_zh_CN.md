# 视频翻译：保持与源视频一致的 FPS 与码率

本文说明视频翻译器如何在输出时尽量保持与源视频一致的帧率与码率，参考了 FFmpeg 文档与同类项目实践。

## 为什么重要

- **FPS 错误**：如果源视频是 24fps，却按 30fps 封装（帧数不变），输出会更短、播放更快。
- **码率错误**：过低会糊、方块明显；过高会导致文件异常增大。重编码时尽量贴近源码率，质量/体积更可预期。

## 同类项目常见做法

- 对 **rawvideo**（管道喂帧）来说：
  - `-framerate`（放在 `-i` 前）设输入帧率；
  - `-r`（放在 `-i` 后）设输出帧率。
- 仅封装软字幕时，常用 `-c:v copy -c:a copy`，避免重编码，FPS/码率天然不变。
- 必须重编码（如去字后重写）时：
  - FPS 优先从 `ffprobe` 的 `avg_frame_rate` / `r_frame_rate` 读取（24、23.976、25、29.97、30 等保持原值）；
  - 码率从 `ffprobe` 读取后传给编码器（`-b:v/-minrate/-maxrate` 等），或改用 CRF 质量模式。

## 本项目当前策略

1. **FPS**
   - 优先调用 `_get_source_fps_ffprobe(video_path)` 读取流帧率（如 `24/1`、`30000/1001`）。
   - 失败回退：OpenCV FPS → `frame_count/duration` 推导 → 默认 24。
   - 不再把合法 24fps“强制改 30fps”。
   - FFmpeg rawvideo 编码时同时设置输入 `-framerate` 与输出 `-r`。

2. **码率**
   - 当 OpenCV 码率缺失或异常时，使用 `ffprobe`（如 `_get_source_bitrate_kbps`）读取。
   - 若读到有效源码率，传给 FFmpeg（`-b:v/-minrate/-maxrate/-bufsize`）实现近似 CBR。
   - 若拿不到可靠码率，使用 CRF，让编码器按质量分配码率。

3. **帧数保持**
   - 读取端每次 `cap.read()` 读一帧，写出端每轮写一帧。
   - 即：输出帧数 = 输入帧数；在正确 FPS 下时长保持一致。

4. **OpenCV 回退路径**
   - 关闭“Use FFmpeg”时，使用 OpenCV `VideoWriter` 并沿用同 FPS。
   - OpenCV 在时间戳/时长上存在已知差异；若你在意严格一致，建议开启 FFmpeg。

## 兼容性（播放器/流媒体）

为兼容浏览器、手机与老播放器（如 QuickTime、WMP），遵循以下实践：

1. **像素格式 `yuv420p`**：H.264 的通用兼容格式，避免部分播放器无法解码 `yuv444p`。
2. **`-movflags +faststart`**：将 MP4 的 moov 元数据前置，支持边下边播。
3. **可选 profile/level**：极老设备可用 baseline/3.0；当前默认 libx264（通常 High）以获得更好压缩率。

## 参考链接

- FFmpeg 文档：<https://ffmpeg.org/ffmpeg.html>
- Stack Overflow（输入/输出帧率选项区别）：<https://stackoverflow.com/questions/51143100/framerate-vs-r-vs-filter-fps>
- SuperUser（跨设备兼容编码参数）：<https://superuser.com/questions/859010/what-ffmpeg-command-line-produces-video-more-compatible-across-all-devices>
- Apple SE（QuickTime 与 yuv420p）：<https://apple.stackexchange.com/questions/166553/why-wont-video-from-ffmpeg-show-in-quicktime-imovie-or-quick-preview>
- 同类项目：
  - <https://github.com/YaoFANGUK/video-subtitle-remover>
  - <https://github.com/jianchang512/pyvideotrans>
  - <https://github.com/cchuang/subtitle-burner>

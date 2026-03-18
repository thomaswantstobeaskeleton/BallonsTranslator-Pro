# Video translator: matching source FPS and bitrate

How the Video translator makes the output video match the source framerate and bitrate, following practices used by similar projects and FFmpeg documentation.

## Why it matters

- **Wrong FPS**: If we encode at 30 fps while the source is 24 fps (but we keep the same number of frames), the output file is **shorter** and plays **faster** than the original. If we used to force 30 fps for “suspicious” values, 24 fps sources were corrupted that way.
- **Wrong bitrate**: Too low → blocky; too high → huge files. Matching the source (when re-encoding) keeps quality and size predictable.

## How similar projects do it

- **FFmpeg docs / Stack Overflow**: For **raw video** (e.g. piping frames), use **`-framerate`** before `-i` to set the **input** framerate; use **`-r`** after `-i` for the **output** framerate. Duration = (number of frames) / framerate.
- **Copy streams**: When only adding soft subs, many tools use `-c:v copy -c:a copy` so video/audio are not re-encoded and FPS/bitrate stay identical.
- **Re-encode**: When we must re-encode (e.g. after inpainting), we get FPS and bitrate from the source and pass them explicitly:
  - **FPS**: Prefer **ffprobe** (`avg_frame_rate` or `r_frame_rate`) so 24, 23.976, 25, 29.97, 30 etc. are preserved. Do **not** clamp valid rates (e.g. 24) to 30.
  - **Bitrate**: Use **ffprobe** to read stream/format bitrate, then pass `-b:v`, `-minrate`, `-maxrate` (and optionally `-bufsize`) to the encoder when we want CBR-like output; otherwise use CRF.

## What we do in this project

1. **FPS**
   - **ffprobe first**: We call `_get_source_fps_ffprobe(video_path)` to read the video stream’s `avg_frame_rate` (e.g. `24/1`, `30000/1001`). That value is used for both reading (OpenCV returns one frame per read) and encoding.
   - **Fallback**: If ffprobe fails, we use OpenCV’s reported FPS, or FPS derived from (frame_count / duration), or 24 as a safe default. We **never** force 30 fps for valid 24 fps sources.
   - **FFmpeg rawvideo**: When using FFmpeg for encoding, we pass **`-framerate`** for the rawvideo **input** (so the pipe is interpreted at the correct rate) and **`-r`** for the **output** so the container has the same framerate.

2. **Bitrate**
   - We use **ffprobe** (e.g. `_get_source_bitrate_kbps`) when OpenCV’s bitrate is missing or too low. If we have a valid source bitrate, we pass it to FFmpeg (`-b:v`, `-minrate`, `-maxrate`, `-bufsize`) for CBR-style encoding. If not, we use **CRF** so the encoder chooses bitrate by quality.

3. **Frame count**
   - We read one frame per `cap.read()` and write one frame per iteration. So **output frame count = input frame count**. With FPS set from ffprobe (or fallback), **output duration = input duration**.

4. **OpenCV fallback**
   - When “Use FFmpeg” is off, we use OpenCV’s `VideoWriter` with the same FPS. OpenCV has known issues with timestamps/duration; using FFmpeg is recommended when exact FPS/duration matter.

## Compatibility (players & streaming)

So the output plays reliably in browsers, phones, and older players (e.g. QuickTime, Windows Media Player), we follow practices from FFmpeg docs and Super User/Stack Overflow:

1. **Pixel format: `yuv420p`** — We pass `-pix_fmt yuv420p` to libx264. Some builds default to yuv444p, which many players (including Apple QuickTime/iMovie) cannot play. YUV420P is the standard for H.264 and works everywhere.
2. **Web/streaming: `-movflags +faststart`** — For MP4 output we add `-movflags +faststart`. This moves the moov atom (metadata) to the start so playback can begin without downloading the whole file.
3. **Optional: profile/level** — For maximum compatibility with very old devices, some projects use `-profile:v baseline -level 3.0`. We use libx264 defaults (typically High) for better compression; an option could be added later for baseline.

## References

- FFmpeg: [Options for raw video / image input](https://ffmpeg.org/ffmpeg.html) — use `-framerate` for input where the format has no inherent framerate.
- Stack Overflow: [Framerate vs r vs Filter fps](https://stackoverflow.com/questions/51143100/framerate-vs-r-vs-filter-fps) — input vs output options.
- Super User: [What ffmpeg command line produces video more compatible across all devices?](https://superuser.com/questions/859010/what-ffmpeg-command-line-produces-video-more-compatible-across-all-devices) — baseline, yuv420p, faststart.
- Apple Stack Exchange: [Why won't video from ffmpeg show in QuickTime?](https://apple.stackexchange.com/questions/166553/why-wont-video-from-ffmpeg-show-in-quicktime-imovie-or-quick-preview) — need yuv420p for H.264.
- Similar projects: [video-subtitle-remover](https://github.com/YaoFANGUK/video-subtitle-remover), [pyvideotrans](https://github.com/jianchang512/pyvideotrans), [subtitle-burner](https://github.com/cchuang/subtitle-burner).

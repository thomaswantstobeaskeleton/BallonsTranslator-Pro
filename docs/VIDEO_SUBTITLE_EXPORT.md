# Video translator: subtitle export (SRT, ASS, VTT)

How we export sidecar subtitles so they match common practice and work in players, YouTube, and FFmpeg. For video encoding (FPS, bitrate, compatibility), see [VIDEO_TRANSLATOR_FPS_AND_BITRATE.md](VIDEO_TRANSLATOR_FPS_AND_BITRATE.md).

## Practices from similar projects

- **rockbenben/subtitle-translator**: SRT and ASS support, batch export.
- **SubZilla, subtitle_utils**: UTF-8 encoding; SRT/ASS/VTT format conversion.
- **pyVideoTrans**: SRT for merge/mux; `-c:s mov_text` and `language=` for MP4 embedding.
- **FFmpeg / SubRip spec**: SRT uses **comma** for decimal in timestamps (e.g. `00:00:01,000`); WebVTT uses **period**; ASS needs PlayResX/PlayResY for correct scaling when positioning is used.

## What we do

### Encoding

- All subtitle files are written with **UTF-8** (`encoding="utf-8"`), so players and platforms (YouTube, Vimeo, etc.) can handle any language.

### SRT

- **Timestamp format**: `HH:MM:SS,mmm` (comma between seconds and milliseconds, as per SubRip). We use `replace(".", ",")` so we never output a period in the timecode.
- **Structure**: Index, blank line, `start --> end`, text, blank line. Newlines in text are replaced with a space so each subtitle is one line.
- Used for: sidecar export, and for muxing into MP4 as a soft subtitle stream when “Mux SRT into video” is on.

### WebVTT

- **Timestamp format**: `HH:MM:SS.mmm` (period for decimal, as per WebVTT).
- **Header**: First line is `WEBVTT`, then blank line, then cues.
- Same cue layout as SRT (index, time range, text).

### ASS (Advanced SubStation Alpha)

- **Script Info**: We write `PlayResX` and `PlayResY` when we have video width/height (OCR, ASR, existing-subs paths). This matches the video resolution so players and FFmpeg’s `subtitles` filter scale and position correctly.
- **Styles**: Default style with Arial, font size 20, white text, outline, alignment 2 (bottom center).
- **Escaping**: Backslash and curly braces in text are escaped (`\` → `\\`, `{` → `{{`, `}` → `}}`). Line breaks in text are written as `\N`.
- **Timestamps**: ASS format `H:MM:SS.cc` (centiseconds).

### Muxing SRT into MP4 (inpaint-only / soft subs)

- When “Inpaint only” and “Mux SRT into video” are enabled, we run FFmpeg to add the SRT as a **mov_text** subtitle stream and set **language=eng** so players show the track as English. Command pattern: `-i video -i srt -c:v copy -c:s mov_text -metadata:s:s:0 language=eng output.mp4`.

## References

- [SubRip (SRT) format](https://en.wikipedia.org/wiki/SubRip) — comma for milliseconds.
- [WebVTT](https://www.w3.org/TR/webvtt1/) — period for milliseconds.
- [ASS format](https://wiki.videolan.org/SubStation_Alpha/) — PlayResX/PlayResY, styles, escaping.
- [FFmpeg subtitles (burn-in, soft subs)](https://www.ffmpeg.media/articles/subtitles-burn-in-soft-subs-format-conversion).
- [pyVideoTrans: soft vs hard subs](https://pyvideotrans.com/en/blog/ffmpeg-subtitles) — mov_text, language metadata.

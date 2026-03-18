# Video translator: frame pacing and timing

This doc summarizes how we preserve frame count and timing when adding text, doing OCR, and encoding, and what issues can still occur.

## What we do (no slowdown/speedup)

- **Strict 1:1 frame loop**: For every `cap.read()` we call `write_frame(...)` exactly once (either the processed frame or the original). So the output has the **same number of frames** as the input; we never drop or duplicate frames.
- **Timing**: We pass the source FPS to the writer (OpenCV `VideoWriter` or FFmpeg `-r` on the raw pipe). So output duration is `frame_count / fps`, matching the intended playback length.
- **FFmpeg pipe**: We send raw frames as fast as we produce them. FFmpeg interprets each frame as 1/FPS seconds. So even if processing is slow in wall-clock time, the resulting file has correct duration and plays at the right speed.
- **FPS fallback**: OpenCV `CAP_PROP_FPS` can be wrong (0, huge, or bogus) for some codecs and VFR. We:
  - Normalize FPS: if reported FPS is ≤0 we use 25; if &lt;15 we use 24 (unless we have a better value).
  - When reported FPS is suspicious (0, &gt;200, or &lt;15), we try **FPS = frame_count / duration** by seeking to the end of the file and reading `CAP_PROP_POS_MSEC`, then seeking back. That gives a more reliable FPS for many problematic files.

## Possible issues

1. **Variable frame rate (VFR)**  
   We treat the source as constant frame rate: each frame is assumed to last 1/FPS seconds. For true VFR (e.g. some phone recordings), frame timings in the original may not be uniform; our output will have uniform timing. If you need frame-accurate VFR, you’d need to pass timecodes to FFmpeg (e.g. with `setpts` or a timecode file) or use a different pipeline; that is not implemented.

2. **OpenCV FPS wrong**  
   For some containers/codecs, `CAP_PROP_FPS` is unreliable even after our fallback (e.g. no duration, or wrong duration). If the output video “feels” too fast or too slow, the source FPS was likely wrong; try re-encoding the source to a known CFR (e.g. `ffmpeg -i in.mp4 -r 30 -c:v libx264 out.mp4`) and use that as input.

3. **Processing time vs playback**  
   How long the job runs (wall-clock) does not change the number of frames or the output duration. Only the FPS and frame count determine playback length.

4. **Temporal smoothing**  
   We blend between the previous keyframe result and the current one. That only affects which pixels we write for each frame; we still write one frame per input frame, so pacing is unchanged.

## References

- OpenCV `CAP_PROP_FPS` issues: [opencv/opencv#21006](https://github.com/opencv/opencv/issues/21006), [opencv/opencv#24000](https://github.com/opencv/opencv/issues/24000).
- FFmpeg raw video input: `-r` sets the input frame rate for the pipe; each frame is given that duration in the output.

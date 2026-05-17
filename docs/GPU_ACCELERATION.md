# GPU acceleration setup (NVIDIA + AMD)

BallonsTranslator-Pro now uses one launcher flow instead of separate GPU-specific start scripts.

## Recommended Windows entrypoint

Double-click **`launcher.bat`** and choose one of these options:

| Menu option | Use when | What it installs/uses |
| --- | --- | --- |
| **1. Start (auto GPU)** | Most users | NVIDIA → CUDA PyTorch; AMD RX 7000/9000 + Python 3.12 → AMD ROCm preview; other AMD Windows GPUs → DirectML; otherwise CPU. |
| **4. Force AMD ROCm preview** | Radeon RX 7000/9000 users who are on Windows + Python 3.12 and want the AMD preview stack | AMD ROCm/PyTorch preview wheels from AMD. |
| **5. Force AMD DirectML** | Any AMD Windows user who wants the broad fallback, or whose ROCm wheel does not match Python | `torch-directml`. |
| **6. CPU-only safe mode** | Troubleshooting crashes or VRAM issues | CPU PyTorch. |

`launch_win.bat` still works and auto-detects by default. Old specialized scripts such as `launch_win_amd_nightly.bat` and `launch_win_with_autoupdate.bat` are now compatibility wrappers that call the unified launcher path.

## Command-line overrides

```bat
launch_win.bat --gpu-profile auto
launch_win.bat --gpu-profile nvidia-cuda
launch_win.bat --gpu-profile amd-rocm-preview
launch_win.bat --gpu-profile amd-directml
launch_win.bat --gpu-profile cpu
```

You can also set an environment variable before launching:

```bat
set BT_GPU_PROFILE=amd-directml
launch_win.bat
```

For advanced users, `TORCH_COMMAND` still overrides the exact PyTorch install command.

## AMD Radeon RX 9070 XT / RX 9070 notes

Official references: AMD lists ROCm/PyTorch Windows support for Radeon RX 9070 XT-class GPUs in the Radeon/Ryzen ROCm compatibility docs, and Microsoft documents DirectML as the broad Windows PyTorch backend for AMD/Intel/NVIDIA GPUs.


- On Windows with **Python 3.12**, auto mode picks the AMD ROCm Windows preview wheels for RX 9000/RDNA4 cards.
- If you are on Python 3.10/3.11 or the ROCm wheel says it is not supported on your platform, use **AMD DirectML** instead: `launch_win.bat --gpu-profile amd-directml`.
- After launching, open **Tools → Diagnostics → Runtime resource summary** or **Copy startup diagnostics**. Look for `Detected backend: privateuseone` (DirectML), `Detected backend: cuda` (NVIDIA/ROCm-style torch builds), or available device selectors containing `privateuseone:0` / `cuda`.
- If the app still runs on CPU, run `launch_win.bat --gpu-report` and include that output in a bug report.

## NVIDIA notes

Auto mode checks for `nvidia-smi`/NVIDIA display adapters. If it sees NVIDIA but the active Python has CPU-only torch, it force-reinstalls the CUDA PyTorch wheel once so the app does not silently stay on CPU.

If you need a different CUDA wheel, set `TORCH_COMMAND` manually from the command shown on <https://pytorch.org/get-started/locally/>.

## Useful official links

- AMD ROCm Radeon/Ryzen Windows compatibility: <https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html>
- Microsoft PyTorch with DirectML on Windows: <https://learn.microsoft.com/windows/ai/directml/pytorch-windows>
- PyTorch install selector: <https://pytorch.org/get-started/locally/>

## What “using the GPU” means in the app

Most detector/OCR/inpainter modules expose a **device** setting. Set that to the detected GPU device (`cuda`, `cuda:0`, `privateuseone:0`, `mps`, etc.) where available. Some modules are CPU-only or have their own runtime limitations, so a GPU runtime does not guarantee every module can use every backend.

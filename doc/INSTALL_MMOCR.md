# How to install MMOCR for BallonsTranslator (text detection)

MMOCR is **optional**. It powers the **mmocr_det** text detector. Use the **same Python** that runs BallonsTranslator.

On **Windows**, the MMOCR stack (mmengine, mmcv, mmdet, mmocr) can fail with:

- `ModuleNotFoundError: No module named 'pkg_resources'` when building mmcv
- Build errors when mmcv is compiled from source
- **`DLL load failed while importing _ext: The specified procedure could not be found`** — the pre-built mmcv wheel was built for a **different PyTorch version** (e.g. 2.0). If you use **PyTorch 2.5+ or 2.7+**, that wheel is incompatible. Use **CTD**, **Surya**, or **Paddle** detection instead, or build mmcv from source (see step 4).

Follow the steps below in order.

---

## 1. Fix `pkg_resources` (do this first)

`pkg_resources` comes from **setuptools**. Install or upgrade it:

```cmd
python -m pip install --upgrade pip setuptools wheel
```

Then retry the MMOCR install (step 2). If mmcv still tries to **build from source** and fails, use a **pre-built wheel** (step 3).

---

## 2. Install MMOCR stack with MIM

Open **Command Prompt** or **PowerShell** and run:

```cmd
pip install -U openmim
mim install mmengine
mim install mmcv mmdet
mim install mmocr
```

- If **mmcv** installs successfully, you are done; **mmocr_det** will appear in the app’s text detection list.
- If **mmcv** fails (e.g. “error: subprocess-exited-with-error” or long build errors), go to step 3.

---

## 3. Install mmcv from a pre-built wheel (Windows)

MIM may try to build mmcv from source on Windows, which often fails. Use a **pre-built wheel** that matches your **PyTorch** and **CUDA** versions.

### 3.1 Check PyTorch and CUDA

```cmd
python -c "import torch; print('PyTorch', torch.__version__); print('CUDA', torch.version.cuda or 'cpu')"
```

Example output: `PyTorch 2.2.0+cu118` and `CUDA 11.8`. Note your **torch** version (e.g. 2.2, 2.5) and **CUDA** (e.g. 11.8, 12.1) or “cpu”.

### 3.2 Download the right mmcv wheel

Open in a browser:

**https://download.openmmlab.com/mmcv/dist/**

Then open the folder that matches your setup, for example:

- **CUDA 11.8 + PyTorch 2.2:** `cu118/torch2.2/`
- **CUDA 11.8 + PyTorch 2.5:** `cu118/torch2.5/` (if present)
- **CPU only:** `cpu/torch2.2/` (or the torch version you have)

Inside that folder, pick the **.whl** that matches:

- **Python 3.10:** filename contains `cp310`
- **Windows 64-bit:** filename contains `win_amd64`

Example: `mmcv-2.2.0-cp310-cp310-win_amd64.whl` for Python 3.10 on Windows with that build.

If there is **no** folder for your exact PyTorch version (e.g. no torch2.7), try the **nearest** torch version (e.g. torch2.2 or torch2.5). **Important:** the wheel must match your **installed** PyTorch version. If you have PyTorch 2.7 and only a torch2.0 wheel is available, loading mmcv will fail at runtime with "DLL load failed while importing _ext". In that case use **CTD**, **Surya**, or **Paddle** detection (step 4).

### 3.3 Install the wheel and the rest

Use a **compatible mmcv** so that mmocr 1.0.1 and mmdet accept it (mmcv >= 2.0.0rc4, **< 2.1.0**; mmdet **< 3.2.0**). Example with pip find-links:

```cmd
pip install "mmcv>=2.0.0rc4,<2.1.0" -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html
pip install "mmdet>=3.0.0rc5,<3.2.0"
mim install mmocr
```

Or install a downloaded wheel:

```cmd
pip install <path-or-url-to-downloaded-mmcv.whl>
mim install mmdet
mim install mmocr
```

Example if the wheel is in your Downloads folder:

```cmd
pip install "%USERPROFILE%\Downloads\mmcv-2.2.0-cp310-cp310-win_amd64.whl"
mim install mmdet
mim install mmocr
```

After this, **mmocr_det** should appear in the app.

---

## 4. If MMOCR still won’t install on Windows

**"DLL load failed while importing _ext"** at runtime means the mmcv wheel was built for a different PyTorch (e.g. 2.0) than yours (e.g. 2.7). Use another detector below.

You can keep using the other text detectors:

- **CTD** – best for manga/comic bubbles (already in the project).
- **surya_det** – line-level, 90+ languages (`pip install surya-ocr`).
- **paddle_det** – PaddleOCR detection (needs `paddleocr` and `paddlepaddle`).
- **paddle_det_v5** – PP-OCRv5 detection-only (needs **paddleocr 3.x**); see PaddleOCR docs for install.

So MMOCR is optional; the app works without it.

---

## 5. Quick reference

| Step | Command |
|------|--------|
| Fix setuptools | `python -m pip install --upgrade pip setuptools wheel` |
| Install with MIM | `pip install -U openmim` then `mim install mmengine` then `mim install mmcv mmdet` then `mim install mmocr` |
| If mmcv fails | Install a matching mmcv **.whl** from https://download.openmmlab.com/mmcv/dist/ then `mim install mmdet` and `mim install mmocr` |

---

## Summary

1. Run **`python -m pip install --upgrade pip setuptools wheel`** to fix `pkg_resources`.
2. Retry **`mim install mmengine`** and **`mim install mmcv mmdet`**.
3. If mmcv still fails, install mmcv from a **pre-built Windows wheel** for your PyTorch/CUDA, then install mmdet and mmocr.
4. If you can’t get MMOCR working, use **CTD** or **paddle_det** / **paddle_det_v5** instead.

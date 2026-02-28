# How to install PaddleOCR for BallonsTranslator

PaddleOCR is **optional**. If it’s not installed, the **paddle_ocr** OCR module and **paddle_det** text detector won’t appear. Use the **same Python** that runs BallonsTranslator (the one shown in the app’s “About” or in the log).

**paddle_det is CPU-only.** The **paddle_det** text detector in BallonsTranslator runs on CPU only (no CUDA/GPU option). This avoids conflicts when using Ocean OCR or other PyTorch-based modules in the same session (e.g. “_gpuDeviceProperties is already registered”). Install **paddlepaddle** (CPU) for paddle_det. If you use **paddle_ocr** and want GPU, you can install **paddlepaddle-gpu** separately; use a different detector (e.g. Surya, CTD) when using Ocean OCR.

Reference: [GitHub issue #835](https://github.com/dmMaze/BallonsTranslator/issues/835#issuecomment-2772940806).

---

## Where to install

- **Use the same Python executable that runs BallonsTranslator.**
- If you start the app with `python launch.py`, that’s your system Python (e.g. `C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe`).
- If you use a bundled env (e.g. `ballontrans_pylibs_win\python.exe`), use that path instead.

Check which Python the app uses: start BallonsTranslator and check the log line that says `Python executable: ...`.

---

## Steps (Windows, Python 3.10) – CPU-only (recommended for paddle_det + Ocean OCR)

Open **Command Prompt** or **PowerShell** and run the following **in order**.

### 1. Go to the project folder (optional but clearer)

```cmd
cd c:\Users\Administrator\BallonsTranslator
```

### 2. Install PaddlePaddle (CPU)

Use **paddlepaddle** (CPU) for **paddle_det**. This avoids “_gpuDeviceProperties is already registered” when using Ocean OCR or other PyTorch modules.

```cmd
"C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe" -m pip install -U paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. Install PaddleOCR

```cmd
"C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe" -m pip install -U paddleocr
```

### 4. Restart BallonsTranslator

After a successful install, restart the app. **paddle_det** and **paddle_ocr** should appear in Config. **paddle_det** runs on CPU only.

---

## If you use a virtual environment or another Python

Replace the Python path in the commands above with your actual executable, for example:

```cmd
path\to\your\python.exe -m pip install -U paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple
path\to\your\python.exe -m pip install -U paddleocr
```

---

## Verify

**PaddleOCR import:**

```cmd
"C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe" -c "from paddleocr import PaddleOCR; print('PaddleOCR OK')"
```

If you see `PaddleOCR OK`, the same Python will see PaddleOCR when you run BallonsTranslator.

**CPU install:** With paddlepaddle (CPU), the verify command above is enough. paddle_det has no GPU option.

---

## Troubleshooting

- **“generic_type: type _gpuDeviceProperties is already registered!”**  
  You have **paddlepaddle-gpu** installed while using PyTorch (e.g. Ocean OCR). **Fix:** uninstall GPU Paddle and use CPU-only:  
  `python -m pip uninstall paddlepaddle-gpu -y`  
  then  
  `python -m pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple`  
  Restart BallonsTranslator. paddle_det is CPU-only and does not offer a GPU option.

- **“ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute…]” (paddle_det_v5)**  
  Known Paddle/oneDNN bug; **both PP-OCRv5_mobile_det and PP-OCRv5_server_det** can hit it on Paddle 3.3+. **Reliable fix:** install a matching pair: `pip install paddlepaddle==3.2.0 paddleocr==3.3.0` (or `paddleocr==3.3.3`), then restart the app. See [PaddleOCR discussion #17350](https://github.com/PaddlePaddle/PaddleOCR/discussions/17350).

- **“OSError: [WinError 127] … phi.dll or one of its dependencies”**  
  Typically from **paddlepaddle-gpu** (CUDA/cuDNN missing or PATH). For paddle_det, use **paddlepaddle** (CPU) instead: uninstall paddlepaddle-gpu, then install paddlepaddle (see Steps above).

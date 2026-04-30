# Windows .exe build

Use this on a **Windows** machine (PowerShell):

```powershell
cd BallonsTranslator-Pro
powershell -ExecutionPolicy Bypass -File scripts/build_windows_exe.ps1 -Clean
```

## Optional: quantize optional ONNX models

For smaller/faster CPU-oriented optional ONNX inpainters:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_windows_exe.ps1 -Clean -QuantizeOptionalOnnx -OnnxQuantMode dynamic
```

Modes:
- `dynamic` (recommended): weights-only int8 quantization.
- `static`: calibrated quantization path.

The quantizer writes side-by-side models:
- `data/models/inpainting_lama_2025jan.dynamic.int8.onnx`
- `data/models/lama_manga.dynamic.int8.onnx`

You can also run quantization manually:

```bash
python scripts/quantize_optional_onnx.py --mode dynamic
```

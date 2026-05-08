# GPU / CUDA troubleshooting

If CUDA is installed on the machine but the app only shows `cpu` in module device selectors, the Python environment is usually using a CPU-only PyTorch wheel.

## Automatic launcher repair

When started through `launch.py`, BallonsTranslator now checks for an NVIDIA GPU before importing the main app. If an NVIDIA GPU is visible and the installed PyTorch build is CPU-only or cannot use CUDA, the launcher reinstalls the CUDA-enabled PyTorch wheel so `cuda` appears in device selectors after restart.

Set `BT_SKIP_CUDA_TORCH_REINSTALL=1` if you intentionally want to keep a CPU-only PyTorch install.

## Manual checks

Run:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
```

Expected for NVIDIA GPU use:

- `torch.version.cuda` is not `None`.
- `torch.cuda.is_available()` is `True`.
- `torch.cuda.device_count()` is at least `1`.

If those checks fail, reinstall PyTorch with the command printed by the launcher or run:

```bash
python -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then restart the app and check **Config → General → Device diagnostics**.

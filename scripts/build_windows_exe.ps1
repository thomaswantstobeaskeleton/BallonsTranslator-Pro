param(
    [string]$PythonExe = "python",
    [switch]$Clean,
    [switch]$QuantizeOptionalOnnx,
    [ValidateSet("dynamic", "static")]
    [string]$OnnxQuantMode = "dynamic"
)

$ErrorActionPreference = "Stop"

Write-Host "[1/6] Verifying platform..."
if (-not $IsWindows) {
    throw "This script must run on Windows to produce a native .exe."
}

Write-Host "[2/6] Verifying Python..."
& $PythonExe --version

Write-Host "[3/6] Installing build deps..."
& $PythonExe -m pip install -r requirements.txt
& $PythonExe -m pip install pyinstaller

if ($QuantizeOptionalOnnx) {
    Write-Host "[4/6] Quantizing optional ONNX models ($OnnxQuantMode)..."
    $env:BT_QUANTIZE_OPTIONAL_ONNX = "1"
    $env:BT_ONNX_QUANT_MODE = $OnnxQuantMode
    & $PythonExe scripts/quantize_optional_onnx.py --mode $OnnxQuantMode
}
else {
    Write-Host "[4/6] Skipping optional ONNX quantization."
}

if ($Clean) {
    Write-Host "[5/6] Cleaning prior build artifacts..."
    if (Test-Path build) { Remove-Item -Recurse -Force build }
    if (Test-Path dist) { Remove-Item -Recurse -Force dist }
}
else {
    Write-Host "[5/6] Keeping prior build artifacts (no --Clean)."
}

Write-Host "[6/6] Building exe via launch.spec..."
& $PythonExe -m PyInstaller --noconfirm launch.spec

Write-Host "Done. Expected output folder: dist\\launch"

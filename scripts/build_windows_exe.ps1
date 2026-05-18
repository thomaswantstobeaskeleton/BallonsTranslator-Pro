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
$BadPython310 = & $PythonExe -c "import sys; print(int((3, 10, 0) <= sys.version_info[:3] < (3, 10, 2)))"
if ($BadPython310 -eq "1") {
    throw "Python 3.10.0/3.10.1 can crash PyInstaller with IndexError: tuple index out of range. Install Python 3.10.2 or newer and rerun this script."
}

Write-Host "[3/6] Installing build deps..."
& $PythonExe -m pip install -r requirements.txt
& $PythonExe -m pip install --upgrade "pyinstaller>=6.11,<7"

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
& $PythonExe -m PyInstaller --clean --noconfirm launch.spec

Write-Host "Done. Expected output folder: dist\\launch"

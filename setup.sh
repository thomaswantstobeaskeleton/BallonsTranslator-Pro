#!/usr/bin/env bash
# Section 11: Portable one-click setup (Linux / macOS)
# Installs dependencies. PyTorch is auto-installed on first "python launch.py" run.

set -e
cd "$(dirname "$0")"

echo "BallonsTranslator-Pro setup (Linux/macOS)"
echo ""

# Prefer venv in project root if it exists
if [ -f "venv/bin/activate" ]; then
    echo "Using existing venv."
    source venv/bin/activate
else
    echo "Using system Python."
fi

echo "Installing requirements..."
python3 -m pip install --upgrade pip -q
python3 -m pip install -r requirements.txt --disable-pip-version-check

echo ""
echo "Setup done. Next:"
echo "  1. Run:  python3 launch.py"
echo "  2. First run will install PyTorch (auto-detects NVIDIA/AMD GPU or CPU)."
echo "  3. Fonts:  fonts/   (add .ttf/.otf here)"
echo "  4. Models:  data/   (downloaded on first use; copy this folder when moving)"
echo "  5. Output:  saved in each project folder you open"
echo "  6. Problems?  See docs/TROUBLESHOOTING.md"
echo ""

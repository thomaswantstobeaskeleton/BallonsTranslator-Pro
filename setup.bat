@echo off
REM Section 11: Portable one-click setup (Windows)
REM Installs dependencies. PyTorch is auto-installed on first "python launch.py" run.

setlocal
cd /d "%~dp0"

echo BallonsTranslator setup (Windows)
echo.

if not defined PYTHON set PYTHON=python
REM Prefer venv in project root if it exists
if exist "venv\Scripts\python.exe" (
    echo Using existing venv.
    set PYTHON=venv\Scripts\python.exe
) else (
    echo Using: %PYTHON%
)

echo Installing requirements...
"%PYTHON%" -m pip install --upgrade pip --quiet
"%PYTHON%" -m pip install -r requirements.txt --disable-pip-version-check
if errorlevel 1 (
    echo Failed to install requirements.
    exit /b 1
)

echo.
echo Setup done. Next:
echo   1. Run:  python launch.py
echo   2. First run will install PyTorch (auto-detects NVIDIA/AMD GPU or CPU).
echo   3. Fonts:  fonts\   (add .ttf/.otf here)
echo   4. Models:  data\   (downloaded on first use; copy this folder when moving)
echo   5. Output:  saved in each project folder you open
echo   6. Problems?  See docs\TROUBLESHOOTING.md
echo.
endlocal

@echo off
REM Windows source-clone setup helper (supports global Python or project venv).
REM Installs dependencies. PyTorch is auto-installed on first "python launch.py" run.

setlocal
cd /d "%~dp0"

echo BallonsTranslator setup (Windows)
echo.
echo Choose Python environment:
echo   [1] Use regular/system Python (requires python or py launcher in PATH)
echo   [2] Use project venv (create it if missing)
echo.
set /p BT_PY_MODE=Enter 1 or 2 [2]: 
if "%BT_PY_MODE%"=="" set "BT_PY_MODE=2"

set "PYTHON="

if "%BT_PY_MODE%"=="1" goto :setup_system
if "%BT_PY_MODE%"=="2" goto :setup_venv

echo Invalid selection: %BT_PY_MODE%
exit /b 1

:setup_system
py -3 -c "" >NUL 2>NUL
if %ERRORLEVEL% == 0 (
    set "PYTHON=py -3"
) else (
    python -c "" >NUL 2>NUL
    if %ERRORLEVEL% == 0 (
        set "PYTHON=python"
    )
)
if not defined PYTHON (
    echo Python was not found in PATH.
    echo Install Python 3.10+ and re-run setup, or choose option 2 to create a venv.
    pause
    exit /b 1
)
echo Using system interpreter: %PYTHON%
goto :install

:setup_venv
if not exist "venv\Scripts\python.exe" (
    echo Creating venv in .\venv ...
    py -3 -m venv venv >NUL 2>NUL
    if %ERRORLEVEL% neq 0 (
        python -m venv venv >NUL 2>NUL
    )
)
if not exist "venv\Scripts\python.exe" (
    echo Failed to create venv. Ensure Python 3.10+ is installed and available.
    pause
    exit /b 1
)
set "PYTHON=venv\Scripts\python.exe"
echo Using venv interpreter: %PYTHON%
goto :install

:install
echo.
echo Installing requirements...
%PYTHON% -m pip install --upgrade pip --quiet
%PYTHON% -m pip install -r requirements.txt --disable-pip-version-check
if errorlevel 1 (
    echo Failed to install requirements.
    pause
    exit /b 1
)

echo.
echo Setup done. Next:
echo   1. Run:  launch_win.bat  (or: %PYTHON% launch.py)
echo   2. First run will install PyTorch (auto-detects NVIDIA/AMD GPU or CPU).
echo   3. Fonts:  fonts\   (add .ttf/.otf here)
echo   4. Models: data\    (downloaded on first use; copy this folder when moving)
echo   5. Output: saved in each project folder you open
echo   6. Problems? See docs\TROUBLESHOOTING.md
echo.
pause
endlocal

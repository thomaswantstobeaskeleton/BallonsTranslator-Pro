@echo off
REM Double-click this file to run the app with the project venv (not global Python).
REM For setup: py -3.10 -m venv venv  then  venv\Scripts\python.exe -m pip install -r requirements.txt
setlocal
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo.
    echo [BallonsTranslator] No venv found at: %~dp0venv
    echo Create it, then install deps:
    echo   py -3.10 -m venv venv
    echo   venv\Scripts\python.exe -m pip install -r requirements.txt
    echo   Optional: GPU PyTorch — see README or run launch.py --reinstall-torch
    echo.
    pause
    exit /b 1
)
"venv\Scripts\python.exe" launch.py %*
set ERR=%ERRORLEVEL%
if not "%ERR%"=="0" (
    echo.
    echo launch.py exited with code %ERR%
    pause
)
endlocal
exit /b %ERR%

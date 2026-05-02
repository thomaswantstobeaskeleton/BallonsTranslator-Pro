@echo off
setlocal
cd /d "%~dp0"
python scripts\build_windows_installer.py
pause

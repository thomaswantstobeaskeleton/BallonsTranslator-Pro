@echo off
setlocal
cd /d "%~dp0"
python scripts\package_release.py
pause

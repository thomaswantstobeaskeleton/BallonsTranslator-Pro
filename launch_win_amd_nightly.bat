@echo off
REM Compatibility wrapper. Prefer launcher.bat or launch_win.bat --gpu-profile amd-rocm-preview.
cd /d "%~dp0"
echo [Info] launch_win_amd_nightly.bat is now a compatibility wrapper.
echo [Info] Use launcher.bat for the unified menu, or launch_win.bat --gpu-profile amd-rocm-preview.
call "%~dp0launch_win.bat" --gpu-profile amd-rocm-preview %*
exit /b %ERRORLEVEL%

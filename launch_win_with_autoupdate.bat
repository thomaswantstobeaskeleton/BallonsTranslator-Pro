@echo off
REM Compatibility wrapper. Prefer launcher.bat or launch_win.bat --update.
cd /d "%~dp0"
echo [Info] launch_win_with_autoupdate.bat is now a compatibility wrapper.
echo [Info] Use launcher.bat for the unified menu, or launch_win.bat --update.
call "%~dp0launch_win.bat" --update %*
exit /b %ERRORLEVEL%

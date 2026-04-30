@echo off
REM BallonsTranslator Windows entrypoint.
REM Simplified to avoid PowerShell/script parsing issues.

setlocal
cd /d "%~dp0"

if not "%~1"=="" goto :direct_launch

echo.
echo ==========================================
echo        BallonsTranslator Launcher
echo ==========================================
echo.
echo [1] Start (auto-detect Python / env)
echo [2] Setup dependencies
echo [3] Start with auto-update
echo [4] Start AMD nightly mode
echo [Q] Quit
echo.
choice /C 1234Q /N /M "Select [1-4, Q]: "
set "K=%ERRORLEVEL%"

if "%K%"=="1" goto :menu_launch
if "%K%"=="2" goto :menu_setup
if "%K%"=="3" goto :menu_update
if "%K%"=="4" goto :menu_nightly
goto :end

:menu_launch
call "%~dp0launch_win.bat"
goto :finish

:menu_setup
call "%~dp0setup.bat"
goto :finish

:menu_update
call "%~dp0launch_win_with_autoupdate.bat"
goto :finish

:menu_nightly
call "%~dp0launch_win_amd_nightly.bat"
goto :finish

:direct_launch
call "%~dp0launch_win.bat" %*

:finish
set "ERR=%ERRORLEVEL%"
endlocal & exit /b %ERR%

:end
endlocal
exit /b 0

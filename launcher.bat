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
echo [1] Start (auto GPU: NVIDIA CUDA / AMD ROCm-DirectML / CPU)
echo [2] Setup dependencies
echo [3] Start with auto-update
echo [4] Force AMD ROCm preview (RX 7000/9000, Python 3.12)
echo [5] Force AMD DirectML (broad Windows AMD fallback)
echo [6] CPU-only safe mode
echo [Q] Quit
echo.
choice /C 123456Q /N /M "Select [1-6, Q]: "
set "K=%ERRORLEVEL%"

if "%K%"=="1" goto :menu_launch
if "%K%"=="2" goto :menu_setup
if "%K%"=="3" goto :menu_update
if "%K%"=="4" goto :menu_rocm
if "%K%"=="5" goto :menu_directml
if "%K%"=="6" goto :menu_cpu
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

:menu_rocm
call "%~dp0launch_win.bat" --gpu-profile amd-rocm-preview
goto :finish

:menu_directml
call "%~dp0launch_win.bat" --gpu-profile amd-directml
goto :finish

:menu_cpu
call "%~dp0launch_win.bat" --gpu-profile cpu
goto :finish

:direct_launch
call "%~dp0launch_win.bat" %*

:finish
set "ERR=%ERRORLEVEL%"
endlocal & exit /b %ERR%

:end
endlocal
exit /b 0

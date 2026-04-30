@echo off
REM BallonsTranslator Windows entrypoint (source clone + portable bundle).
REM - No args: interactive menu (Arrow keys/W/S + Enter when PowerShell is available).
REM - With args: direct launch (same behavior as previous source-clone launcher).

setlocal
cd /d "%~dp0"

if not "%~1"=="" goto :direct_launch

goto :interactive_menu

:interactive_menu
set "OPT="
where pwsh >NUL 2>NUL
if %ERRORLEVEL%==0 goto :powershell_menu_pwsh
where powershell >NUL 2>NUL
if %ERRORLEVEL%==0 goto :powershell_menu_windows

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
choice /C 1234QWS /N /M "Select [1-4, W/S, Q]: "
set "K=%ERRORLEVEL%"
if "%K%"=="1" set "OPT=1"
if "%K%"=="2" set "OPT=2"
if "%K%"=="3" set "OPT=3"
if "%K%"=="4" set "OPT=4"
if "%K%"=="5" set "OPT=Q"
if "%K%"=="6" set "OPT=1"
if "%K%"=="7" set "OPT=2"
goto :dispatch

:powershell_menu_pwsh
call :run_ps_menu "pwsh"
goto :dispatch

:powershell_menu_windows
call :run_ps_menu "powershell"
goto :dispatch

:run_ps_menu
set "PSBIN=%~1"
set "PSTMP=%TEMP%\bt_menu_%RANDOM%%RANDOM%.ps1"
(
  echo $ErrorActionPreference = 'Stop'
  echo $items = @(
  echo   @{ Key = '1'; Label = 'Start (auto-detect Python / env)' },
  echo   @{ Key = '2'; Label = 'Setup dependencies' },
  echo   @{ Key = '3'; Label = 'Start with auto-update' },
  echo   @{ Key = '4'; Label = 'Start AMD nightly mode' },
  echo   @{ Key = 'Q'; Label = 'Quit' }
  echo ^)
  echo $sel = 0
  echo function DrawMenu { param([int]$Index)
  echo   Clear-Host
  echo   Write-Host '=========================================='
  echo   Write-Host '       BallonsTranslator Launcher'
  echo   Write-Host '=========================================='
  echo   Write-Host ''
  echo   for ^($i=0; $i -lt $items.Count; $i++^) {
  echo     $prefix = if ^($i -eq $Index^) { '>' } else { ' ' }
  echo     Write-Host ^("$prefix [$($items[$i].Key)] $($items[$i].Label)"^)
  echo   }
  echo   Write-Host ''
  echo   Write-Host 'Use Arrow Up/Down or W/S. Press Enter/Space to select.'
  echo }
  echo DrawMenu $sel
  echo while ^($true^) {
  echo   $k = [Console]::ReadKey^($true^)
  echo   switch ^($k.Key^) {
  echo     'UpArrow' { if ^($sel -gt 0^) { $sel-- }; DrawMenu $sel; continue }
  echo     'DownArrow' { if ^($sel -lt ($items.Count - 1)^) { $sel++ }; DrawMenu $sel; continue }
  echo     'W' { if ^($sel -gt 0^) { $sel-- }; DrawMenu $sel; continue }
  echo     'S' { if ^($sel -lt ($items.Count - 1)^) { $sel++ }; DrawMenu $sel; continue }
  echo     'Enter' { break }
  echo     'Spacebar' { break }
  echo     default {
  echo       $c = $k.KeyChar.ToString^(^).ToUpperInvariant^(^)
  echo       $idx = [Array]::FindIndex^($items, [Predicate[object]]{ param^($it^) $it.Key -eq $c }^)
  echo       if ^($idx -ge 0^) { $sel = $idx; break }
  echo     }
  echo   }
  echo }
  echo [Console]::Out.Write^($items[$sel].Key^)
) > "%PSTMP%"

for /f "usebackq delims=" %%I in (`%PSBIN% -NoProfile -ExecutionPolicy Bypass -File "%PSTMP%"`) do set "OPT=%%I"
del /f /q "%PSTMP%" >NUL 2>NUL
exit /b 0

:dispatch
if /I "%OPT%"=="1" goto :menu_launch
if /I "%OPT%"=="2" goto :menu_setup
if /I "%OPT%"=="3" goto :menu_update
if /I "%OPT%"=="4" goto :menu_nightly
goto :end

:menu_launch
call "%~dp0launch_win.bat"
goto :end

:menu_setup
call "%~dp0setup.bat"
goto :end

:menu_update
call "%~dp0launch_win_with_autoupdate.bat"
goto :end

:menu_nightly
call "%~dp0launch_win_amd_nightly.bat"
goto :end

:direct_launch
call "%~dp0launch_win.bat" %*
set "ERR=%ERRORLEVEL%"
goto :finish

:finish
endlocal & exit /b %ERR%

:end
endlocal
exit /b 0

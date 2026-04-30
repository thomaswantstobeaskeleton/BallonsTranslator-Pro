@echo off
setlocal enabledelayedexpansion

set "ROOT_DIR=%~dp0.."
set "OUT_DIR=%ROOT_DIR%\dist"
set "OUT_ZIP=%OUT_DIR%\qm_files_bundle.zip"

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

echo [1/3] Compiling TS to QM...
for %%F in ("%ROOT_DIR%\translate\*.ts") do (
  pyside6-lrelease "%%~fF" -qm "%%~dpnF.qm" >nul
  if errorlevel 1 (
    echo Failed compiling %%~nxF
    exit /b 1
  )
)

echo [2/3] Creating zip...
powershell -NoProfile -Command ^
  "if (Test-Path '%OUT_ZIP%') { Remove-Item -Force '%OUT_ZIP%' }; " ^
  "Compress-Archive -Path '%ROOT_DIR%\translate\*.qm' -DestinationPath '%OUT_ZIP%' -CompressionLevel Optimal"
if errorlevel 1 (
  echo Failed creating zip archive.
  exit /b 1
)

echo [3/3] Done.
echo Created: %OUT_ZIP%
powershell -NoProfile -Command "Get-FileHash -Algorithm SHA256 '%OUT_ZIP%' | Format-List"

endlocal

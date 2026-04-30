@REM Launch BallonsTranslator for Windows portable bundle layout.
@echo off
setlocal
cd /d "%~dp0"

:: Set the path for PaddleOCR and PyTorch libraries (when bundled runtime is present)
set "PADDLE_PATH=%~dp0ballontrans_pylibs_win\Lib\site-packages\torch\lib"
if exist "%PADDLE_PATH%" set "PATH=%PADDLE_PATH%;%PATH%"

:: Prefer existing project venv, then bundled portable python, then system Python launcher/python.
set "PYTHON="
set "PY_MODE_FILE=.bt_python_mode"
set "PY_MODE="
if exist "%PY_MODE_FILE%" (
    set /p PY_MODE=<"%PY_MODE_FILE%"
)
if /I "%PY_MODE%"=="venv" (
    if exist "venv\Scripts\python.exe" set "PYTHON=venv\Scripts\python.exe"
)
if /I "%PY_MODE%"=="system" (
    py -3 -c "" >NUL 2>NUL
    if %ERRORLEVEL% == 0 set "PYTHON=py -3"
    if not defined PYTHON (
        python -c "" >NUL 2>NUL
        if %ERRORLEVEL% == 0 set "PYTHON=python"
    )
)

if not defined PYTHON if exist "venv\Scripts\python.exe" set "PYTHON=venv\Scripts\python.exe"
if not defined PYTHON if exist "ballontrans_pylibs_win\python.exe" set "PYTHON=ballontrans_pylibs_win\python.exe"
if not defined PYTHON (
    py -3 -c "" >NUL 2>NUL
    if %ERRORLEVEL% == 0 set "PYTHON=py -3"
)
if not defined PYTHON (
    python -c "" >NUL 2>NUL
    if %ERRORLEVEL% == 0 set "PYTHON=python"
)

set ERROR_REPORTING=FALSE
mkdir tmp 2>NUL

if not defined PYTHON (
    >tmp\stdout.txt echo.
    >tmp\stderr.txt echo Python was not found. Install Python 3.10+ or run setup.bat to create/use a venv.
    echo Couldn't launch python
    goto :show_stdout_stderr
)

%PYTHON% -c "" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :check_pip
echo Couldn't launch python
goto :show_stdout_stderr

:check_pip
%PYTHON% -m pip --help >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :launch
if "%PIP_INSTALLER_LOCATION%" == "" goto :show_stdout_stderr
%PYTHON% "%PIP_INSTALLER_LOCATION%" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :launch
echo Couldn't install pip
goto :show_stdout_stderr

:launch
%PYTHON% launch.py %*
pause
exit /b

:show_stdout_stderr
echo.
echo exit code: %errorlevel%

for /f %%i in ("tmp\stdout.txt") do set size=%%~zi
if %size% equ 0 goto :show_stderr
echo.
echo stdout:
type tmp\stdout.txt

:show_stderr
for /f %%i in ("tmp\stderr.txt") do set size=%%~zi
if %size% equ 0 goto :endofscript
echo.
echo stderr:
type tmp\stderr.txt

:endofscript
echo.
echo Launch unsuccessful. Exiting.
pause
exit /b

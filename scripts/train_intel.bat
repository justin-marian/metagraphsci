@echo off
REM Double-click launcher: starts MetaGraphSci training on Intel GPU (XPU).
REM Run scripts\setup_intel_windows.bat first.

setlocal
cd /d "%~dp0\.."

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0train_intel.ps1"
set ERR=%ERRORLEVEL%

echo.
if %ERR% NEQ 0 (
    echo Training exited with code %ERR%.
)
pause
endlocal

@echo off
REM Double-click launcher: sets up the Intel XPU training environment.
REM Delegates to the PowerShell script so the logic stays in one place.

setlocal
cd /d "%~dp0\.."

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_intel_windows.ps1"
set ERR=%ERRORLEVEL%

echo.
if %ERR% NEQ 0 (
    echo Setup failed with exit code %ERR%.
) else (
    echo Setup complete. You can now double-click train_intel.bat
)
pause
endlocal

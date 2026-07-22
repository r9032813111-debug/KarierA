@echo off
chcp 65001 >nul
cd /d "%~dp0"
title KarierA WEB3 Stereo - Camera Calibration
set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"

rem Thread counts are controlled per phase inside Python. Global 16-thread
rem environment variables made the live chessboard preview slower on Windows.

echo Stopping WEB3 server so both cameras are free...
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/api/commands/stop' -TimeoutSec 2 | Out-Null } catch {}; $listener = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue; if ($listener) { Stop-Process -Id $listener.OwningProcess -Force }; Start-Sleep -Milliseconds 700"

echo Backing up the active stereo calibration...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'; $backup = Join-Path (Get-Location) 'calibration_backups'; New-Item -ItemType Directory -Path $backup -Force | Out-Null; if (Test-Path -LiteralPath 'stereo_calibration.yaml') { Copy-Item -LiteralPath 'stereo_calibration.yaml' -Destination (Join-Path $backup ('stereo_calibration_' + $stamp + '.yaml')) }; if (Test-Path -LiteralPath 'stereo_calibration.json') { Copy-Item -LiteralPath 'stereo_calibration.json' -Destination (Join-Path $backup ('stereo_calibration_' + $stamp + '.json')) }"

echo.
echo Board: 7 x 6 inner corners, 8 x 7 squares, square size 22.8125 mm.
echo C = capture pair, SPACE = calculate and save, R = reset, Q = exit.
echo Capture 20 varied views before pressing SPACE.
echo CPU calculation: 2 x 8 threads for mono cameras, then 16 threads for stereo.
echo.

"%PYTHON%" calibration_program\full_calibrate_v2.py --left-camera 0 --right-camera 1
set "calibration_exit=%errorlevel%"

echo.
if not "%calibration_exit%"=="0" (
  echo Calibration finished with error %calibration_exit%. The previous YAML remains available in calibration_backups.
) else (
  echo Calibration window closed. If SPACE completed successfully, stereo_calibration.yaml is already active for WEB3.
)
echo Start the project again with START_KARIERA.bat.
pause
exit /b %calibration_exit%

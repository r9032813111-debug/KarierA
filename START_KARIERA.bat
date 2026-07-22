@echo off
chcp 65001 >nul
cd /d "%~dp0"
title KarierA
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
) else (
  python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
)
pause

@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title KarierA - Install
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv || goto :error
)
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :error
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error
echo.
echo Готово. Запустите START_KARIERA.bat
pause
exit /b 0

:error
echo.
echo Установка завершилась с ошибкой. Проверьте Python 3.12 и повторите запуск.
pause
exit /b 1

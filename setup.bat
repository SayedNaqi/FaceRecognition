@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ======================================================
echo   SETUP - Smart Face Recognition
echo ======================================================
where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python not found. Install Python 3.12 first.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Creating virtual environment ...
  python -m venv .venv
)
echo [2/3] Installing dependencies ...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
echo [3/3] Downloading YuNet + SFace models ...
".venv\Scripts\python.exe" src\download_models.py
echo.
echo Setup complete. Run run.bat to start the app.
pause

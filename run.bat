@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ======================================================
echo   CHAHRAHYAB - Smart Face Recognition (Web)
echo   Ghorjistan University - Faculty of Computer Science
echo ======================================================
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Virtual environment not found.
  echo Run setup.bat first.
  pause
  exit /b 1
)
if not exist "models\face_detection_yunet_2023mar.onnx" (
  echo [ERROR] Models not downloaded.
  echo Run setup.bat first.
  pause
  exit /b 1
)
echo Starting server at http://127.0.0.1:5000 ...
start "" /b ".venv\Scripts\python.exe" src\web_app.py
timeout /t 4 >nul
start "" "http://127.0.0.1:5000"
echo.
echo Server is running in the background.
echo To stop it: taskkill /F /IM python.exe  (or close the CMD window)
echo.
pause

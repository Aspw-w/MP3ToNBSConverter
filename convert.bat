@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    call setup.bat
    if errorlevel 1 exit /b 1
)

".venv\Scripts\python.exe" -c "import demucs, librosa, onnxruntime; from basic_pitch.inference import Model" >nul 2>nul
if errorlevel 1 (
    call setup.bat
    if errorlevel 1 exit /b 1
)

".venv\Scripts\python.exe" mp3_to_nbs.py %*

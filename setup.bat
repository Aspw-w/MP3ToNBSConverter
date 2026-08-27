@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo Setting up the high-quality MP3 to NBS environment.
echo The CUDA PyTorch download is about 2 GB. Please wait.
echo.

if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 goto :error
)

set "PYTHON_EXE=.venv\Scripts\python.exe"
"%PYTHON_EXE%" -m pip install --upgrade pip
if errorlevel 1 goto :error

where nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo NVIDIA GPU was not found. Installing CPU PyTorch.
    "%PYTHON_EXE%" -m pip install --upgrade torch
) else (
    echo Installing CUDA 13.0 PyTorch for the NVIDIA GPU.
    "%PYTHON_EXE%" -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu130 "torch==2.13.0+cu130"
)
if errorlevel 1 goto :error

"%PYTHON_EXE%" -m pip install -r requirements.txt
if errorlevel 1 goto :error

rem Basic Pitch declares a TensorFlow dependency that is unavailable on
rem Python 3.14. requirements.txt supplies the ONNX dependencies instead.
"%PYTHON_EXE%" -m pip install --no-deps "basic-pitch==0.4.0"
if errorlevel 1 goto :error

echo Downloading the compact singing-voice detector.
"%PYTHON_EXE%" -c "from mp3_to_nbs import _ensure_yamnet_model; print('Vocal detector:', _ensure_yamnet_model(print))"
if errorlevel 1 echo Warning: vocal detector download failed; conversion will retry later.

"%PYTHON_EXE%" -c "import torch; from mp3_to_nbs import _load_basic_pitch_model; _load_basic_pitch_model(); print('PyTorch:', torch.__version__); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
if errorlevel 1 goto :error

echo.
echo Setup complete. Run convert.bat to start converting.
pause
exit /b 0

:error
echo.
echo Setup failed. Review the error shown above.
pause
exit /b 1

@echo off
REM Windows installation script for RDN_segmentation with uv
REM This script sets up the complete environment for Windows users

echo ========================================
echo RDN Segmentation Environment Setup
echo ========================================
echo.

REM Check if uv is installed
where uv >nul 2>nul
if errorlevel 1 (
    echo ERROR: uv is not installed or not in PATH
    echo Please install uv first:
    echo   PowerShell: irm https://astral.sh/uv/install.ps1 ^| iex
    echo   Or use winget: winget install --id=astral-sh.uv -e
    pause
    exit /b 1
)

echo Found uv package manager
echo.

REM Check for NVIDIA GPU (required for CUDA support)
echo Checking for NVIDIA GPU...
wmic path win32_VideoController get name | findstr /i "NVIDIA" >nul
if errorlevel 1 (
    echo.
    echo ========================================
    echo ERROR: NVIDIA GPU NOT DETECTED
    echo ========================================
    echo.
    echo This software requires an NVIDIA graphics card for CUDA acceleration.
    echo Without CUDA support, the deep learning models will not function properly.
    echo.
    echo Please ensure you have:
    echo   1. An NVIDIA graphics card installed
    echo   2. NVIDIA drivers installed
    echo.
    echo If you have an NVIDIA GPU but it's not detected, try:
    echo   1. Update your NVIDIA drivers
    echo   2. Restart your computer
    echo   3. Run this installer again
    echo.
    pause
    exit /b 1
)
echo Found NVIDIA GPU - proceeding with CUDA installation
echo.

REM Check if Python 3.11+ is available
uv python list >nul 2>nul
if errorlevel 1 (
    echo Installing Python 3.11...
    uv python install 3.11
) else (
    echo Python is available
    echo Ensuring Python 3.11+ compatibility...
    uv python install 3.11
)

REM Create and sync the project
echo Creating uv project...
uv sync

REM Install PyTorch with CUDA support
echo.
echo Installing PyTorch with CUDA support...
uv sync

REM Install special dependencies
echo.
echo Installing pytorch_wavelets from source...
uv run python setup_pytorch_wavelets.py

REM Verify installation
echo.
echo Verifying installation...
uv run python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
uv run python -c "import pytorch_wavelets; print('pytorch_wavelets imported successfully')"

echo.
echo ========================================
echo Installation completed successfully!
echo ========================================
echo.
echo To activate the environment, use:
echo   uv shell
echo.
echo To run applications:
echo   uv run streamlit run MARS/streamlit_apps/streamlit_RDN_segmentation.py
echo   uv run python Scripts/3_class.py
echo.
echo For CPU-only installation (if CUDA failed), run:
echo   uv sync --extra cpu
echo.
pause
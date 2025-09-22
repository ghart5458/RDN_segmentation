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

REM Check if Python is available
uv python list >nul 2>nul
if errorlevel 1 (
    echo Installing Python 3.11...
    uv python install 3.11
) else (
    echo Python is available
)

REM Create and sync the project
echo Creating uv project...
uv sync

REM Install PyTorch with CUDA support (default for Windows)
echo.
echo Installing PyTorch with CUDA support...
uv sync --extra cuda

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
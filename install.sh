#!/bin/bash
# Linux/macOS installation script for RDN_segmentation with uv
# This script sets up the complete environment for Unix-like systems

set -e  # Exit on any error

echo "========================================"
echo "RDN Segmentation Environment Setup"
echo "========================================"
echo

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv is not installed"
    echo "Please install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  Then restart your shell or run: source ~/.bashrc"
    exit 1
fi

echo "Found uv package manager"
echo

# Function to check for NVIDIA GPU
check_nvidia_gpu() {
    echo "Checking for NVIDIA GPU..."
    
    # Check for nvidia-smi command (most reliable)
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            return 0  # GPU found
        fi
    fi
    
    # Fallback: check lspci for NVIDIA devices (Linux)
    if command -v lspci &> /dev/null; then
        if lspci | grep -i nvidia &> /dev/null; then
            return 0  # GPU found
        fi
    fi
    
    # Fallback: check system_profiler for NVIDIA devices (macOS)
    if [[ "$OSTYPE" == "darwin"* ]] && command -v system_profiler &> /dev/null; then
        if system_profiler SPDisplaysDataType | grep -i nvidia &> /dev/null; then
            return 0  # GPU found
        fi
    fi
    
    return 1  # No GPU found
}

# Platform-specific GPU checking
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "macOS detected - this software requires NVIDIA GPU acceleration"
    echo "Unfortunately, modern macOS systems do not support NVIDIA CUDA."
    echo "This software requires CUDA for deep learning acceleration."
    echo ""
    echo "Please use a Linux or Windows system with NVIDIA GPU."
    exit 1
else
    # Linux - check for NVIDIA GPU
    if ! check_nvidia_gpu; then
        echo
        echo "========================================"
        echo "ERROR: NVIDIA GPU NOT DETECTED"
        echo "========================================"
        echo
        echo "This software requires an NVIDIA graphics card for CUDA acceleration."
        echo "Without CUDA support, the deep learning models will not function properly."
        echo
        echo "Please ensure you have:"
        echo "  1. An NVIDIA graphics card installed"
        echo "  2. NVIDIA drivers installed"
        echo "  3. nvidia-smi command available"
        echo
        echo "To install NVIDIA drivers on Ubuntu/Debian:"
        echo "  sudo apt update"
        echo "  sudo apt install nvidia-driver-XXX"
        echo "  (replace XXX with appropriate driver version)"
        echo
        echo "After installing drivers, restart your computer and run this installer again."
        echo
        exit 1
    fi
    echo "Found NVIDIA GPU - proceeding with CUDA installation"
    echo
fi

# Check if Python 3.11+ is available
if ! uv python list &> /dev/null; then
    echo "Installing Python 3.11..."
    uv python install 3.11
else
    echo "Python is available"
    echo "Ensuring Python 3.11+ compatibility..."
    uv python install 3.11
fi

# Create and sync the project
echo "Creating uv project..."
uv sync

# Install PyTorch with CUDA support (Linux only, macOS exits above)
echo
echo "Installing PyTorch with CUDA support..."
uv sync

# Install special dependencies
echo
echo "Installing pytorch_wavelets from source..."
uv run python setup_pytorch_wavelets.py

# Verify installation
echo
echo "Verifying installation..."
uv run python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
uv run python -c "import pytorch_wavelets; print('pytorch_wavelets imported successfully')"

echo
echo "========================================"
echo "Installation completed successfully!"
echo "========================================"
echo
echo "To activate the environment, use:"
echo "  uv shell"
echo
echo "To run applications:"
echo "  uv run streamlit run MARS/streamlit_apps/streamlit_RDN_segmentation.py"
echo "  uv run python Scripts/3_class.py"
echo
echo "For CPU-only installation (if CUDA failed), run:"
echo "  uv sync --extra cpu"
echo
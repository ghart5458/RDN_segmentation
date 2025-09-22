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

# Check if Python is available
if ! uv python list &> /dev/null; then
    echo "Installing Python 3.11..."
    uv python install 3.11
else
    echo "Python is available"
fi

# Create and sync the project
echo "Creating uv project..."
uv sync

# Install PyTorch - detect platform and install appropriate version
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo
    echo "Installing PyTorch for macOS (CPU-only)..."
    uv sync --extra cpu
else
    echo
    echo "Installing PyTorch with CUDA support..."
    uv sync --extra cuda
fi

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
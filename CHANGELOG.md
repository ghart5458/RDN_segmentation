# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2024-09-22

### Major Changes - Environment Modernization

**BREAKING CHANGES:**
- **NVIDIA GPU now mandatory**: Software will not install or function without NVIDIA GPU
- **macOS support dropped**: Due to lack of CUDA support on modern macOS systems
- **Python 3.11+ required**: Dropped support for Python 3.8-3.10
- **CPU-only execution removed**: Deep learning models require CUDA acceleration

### Added
- **Modern uv package manager support**: Faster, more reliable dependency management
- **Automated installation scripts**: 
  - `install.bat` for Windows
  - `install.sh` for Linux
- **NVIDIA GPU detection**: Installation scripts automatically detect and validate GPU presence
- **PyTorch 2.5.0 support**: Latest stable PyTorch with CUDA 12.4
- **Comprehensive error handling**: Clear messages when requirements not met

### Changed
- **Environment setup**: Migrated from conda to uv-based installation
- **Python version**: Updated from >=3.8 to >=3.11,<3.14
- **PyTorch version**: Upgraded from 1.7.0 to 2.5.0
- **CUDA version**: Updated from CUDA 11.8 to CUDA 12.4
- **TensorBoard version**: Updated from 2.4.0 to 2.15.0
- **Project structure**: Added proper setuptools configuration for multiple packages

### Improved
- **Installation speed**: uv provides 10-100x faster package resolution
- **Dependency management**: More reliable and consistent across platforms
- **Error messages**: Better guidance when installation fails
- **Documentation**: Comprehensive setup instructions and system requirements

### Fixed
- **Package discovery**: Resolved setuptools warnings for multiple top-level packages
- **License format**: Updated to modern SPDX string format
- **Build system**: Eliminated deprecation warnings

### Deprecated
- **conda environment**: Setup/pytorch.yaml marked as legacy, will be removed in future version

### Technical Details
- **pyproject.toml**: Complete rewrite with modern Python packaging standards
- **uv sources**: Custom PyTorch CUDA index configuration
- **Cross-platform**: Windows and Linux support with platform-specific optimizations
- **Special dependencies**: pytorch_wavelets still requires GitHub source installation

### Migration Guide

**For new users:**
Use the automated installation scripts - no manual environment setup needed.

**For existing users:**
1. Ensure you have an NVIDIA GPU with updated drivers
2. Remove old conda environment: `conda env remove -n pytorch`
3. Run the new installation script for your platform
4. Verify installation with: `uv run python -c "import torch; print(torch.cuda.is_available())"`

**Note for macOS users:**
This version drops macOS support due to CUDA requirements. Please use Linux or Windows systems with NVIDIA GPUs.

## [1.0.0] - Previous Releases

Historical releases used conda-based installation with broader platform support but less reliable dependency management. See git history for details on earlier versions.
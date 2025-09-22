# RDN_segmentation
Home of the [FEMR Lab](https://femr.la.psu.edu/) RDN segmentation software at PSU

[![DOI](https://zenodo.org/badge/425930676.svg)](https://zenodo.org/badge/latestdoi/425930676)

## System Requirements

**NVIDIA GPU REQUIRED**

This software requires an NVIDIA graphics card with CUDA support for deep learning acceleration. It will **NOT** work on systems without NVIDIA GPUs.

### Supported Platforms
- **Windows** with NVIDIA GPU
- **Linux** with NVIDIA GPU
- **macOS** - Not supported (no CUDA support)

### Prerequisites
- NVIDIA graphics card with CUDA support
- NVIDIA drivers installed
- Python 3.11 or later

## Quick Install

### Automated Installation (Recommended)

**Windows:**
```cmd
git clone https://github.com/femr-lab/RDN_segmentation.git
cd RDN_segmentation
install.bat
```

**Linux:**
```bash
git clone https://github.com/femr-lab/RDN_segmentation.git
cd RDN_segmentation
chmod +x install.sh
./install.sh
```

The installation script will:
1. Check for NVIDIA GPU (required)
2. Install uv package manager if needed
3. Set up Python 3.11+ environment
4. Install PyTorch with CUDA support
5. Install all dependencies
6. Verify installation

### Manual Installation

If you prefer manual installation:

1. **Install uv package manager:**
   ```bash
   # Windows (PowerShell)
   irm https://astral.sh/uv/install.ps1 | iex

   # Linux/macOS
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone and install:**
   ```bash
   git clone https://github.com/femr-lab/RDN_segmentation.git
   cd RDN_segmentation
   uv sync
   uv run python setup_pytorch_wavelets.py
   ```

## Legacy Installation

For systems still using conda:
```bash
conda env create -f Setup/pytorch.yaml
conda activate pytorch
```

**Note:** The conda environment is deprecated and will be removed in future versions. Please migrate to uv.

## Standalone Version

For Windows users who prefer a pre-built container, download RDN_segmentation_container.zip from [Google Drive](https://drive.google.com/drive/folders/1iPtUwCdCiEAd8kMslV4znbZyJqAlmCKl?usp=sharing) and follow RDN_Install_Use_Instructions.txt.


## Authors

* **Nicholas B. Stephens** - *Initial work* nbstephens@proton.me
* **Amiraseed Yazdani** - *Network design*
* **Yung-Chen Sun** - *Network design*
* **Sharon Kuo** -  kuo@d.umn.edu
* **Lily J. Demars** - lvd5263@psu.edu
* **Tim M. Ryan** - *PI* tmr21@psu.edu
* **Vishal Monga** - *PI* vum4@psu.edu

### Funded by
[NSF BCS 1719187](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1719187)

## Please cite
```
@INPROCEEDINGS{9443322,
  author={Yazdani, Amirsaeed and Sun, Yung-Chen and Stephens, Nicholas B. and Ryan, Timothy and Monga, Vishal},
  booktitle={2020 54th Asilomar Conference on Signals, Systems, and Computers},
  title={Multi-Class Micro-CT Image Segmentation Using Sparse Regularized Deep Networks},
  year={2020},
  pages={1553-1557},
  doi={10.1109/IEEECONF51394.2020.9443322}}
```
```
@INPROCEEDINGS{9048654,
  author={Yazdani, Amirsaeed and Stephens, Nicholas B. and Cherukuri, Venkateswararao and Ryan, Timothy and Monga, Vishal},
  booktitle={2019 53rd Asilomar Conference on Signals, Systems, and Computers},
  title={Domain-Enriched Deep Network for Micro-CT Image Segmentation},
  year={2019},
  pages={1867-1871},
  doi={10.1109/IEEECONF44664.2019.9048654}}
```


## License

This project is open-source software distributed under the terms and conditions of the Free Software Foundation's GNU General Public License. https://www.gnu.org/licenses/gpl-3.0.txt

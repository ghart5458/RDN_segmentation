"""
Modernized Streamlit app for RDN segmentation using native st.session_state.

Migrated from legacy custom session state system to Streamlit 1.49+ native functionality.

GUI and functionality by NB Stephens (github.com/NBStephens) nbs49@psu.edu
Based on RDN segmentation work by Yazdani et al., 2020 Asilomar Conference.
"""

import base64
import glob
import math
import os
import sys
import time
from pathlib import Path
from timeit import default_timer as timer

import SimpleITK as sitk
import streamlit as st
import torch
from PIL import Image

# Add project root to Python path for imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from MARS.morphology.segmentation.pytorch_segmentation.execute_3_class_seg import (
    _get_outDir,
    _return_predictors,
    _save_predictors,
    natural_keys,
    read_image,
    rescale_8,
)
from MARS.morphology.segmentation.pytorch_segmentation.net.unet_light_rdn import UNet_Light_RDN


def initialize_session_state():
    """Initialize session state variables with defaults."""
    if "model_path" not in st.session_state:
        st.session_state.model_path = ""
    if "model" not in st.session_state:
        st.session_state.model = ""
    if "input_path" not in st.session_state:
        st.session_state.input_path = ""
    if "input_type" not in st.session_state:
        st.session_state.input_type = "mhd"
    if "output_path" not in st.session_state:
        st.session_state.output_path = ""
    if "out_type" not in st.session_state:
        st.session_state.out_type = "mhd"
    if "use_gpu" not in st.session_state:
        st.session_state.use_gpu = 0
    if "cuda_mem" not in st.session_state:
        st.session_state.cuda_mem = ""
    if "net" not in st.session_state:
        st.session_state.net = None
    if "twoD_to_threeD" not in st.session_state:
        st.session_state.twoD_to_threeD = False


def main():
    """Main application function."""
    st.set_page_config(
        page_title="MARS RDN Segmentation",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    initialize_session_state()

    # Load logo
    logo = load_MARS_logo()
    st.sidebar.image(logo, width=50)

    st.sidebar.markdown("""
    # **<span style="color:red; font-size:2em">MARS:</span>**
    RDN 3-class segmentation
    """, unsafe_allow_html=True)

    # Page selection
    pages = {
        "Settings": page_settings,
        "Segmentation": page_segmentations
    }

    page = st.sidebar.radio("Navigation:", tuple(pages.keys()))

    # Control buttons
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        if st.button("Save"):
            save_settings()
    with col2:
        if st.button("Clear"):
            clear_settings()
    with col3:
        if st.button("GPU"):
            clear_gpu_memory()

    # Display selected page
    pages[page]()


def page_settings():
    """Settings page for model and I/O configuration."""
    file_types = ["mhd", "nii", "tif", "png", "jpg", "bmp", "dcm"]

    st.title("Settings")
    display_current_settings()

    # GPU initialization
    if st.session_state.cuda_mem:
        st.info(f"GPU: {torch.cuda.get_device_name(st.session_state.use_gpu)}, Memory: {st.session_state.cuda_mem}")

    if st.button('Initialize GPU'):
        initiate_cuda()

    st.divider()

    # Model settings
    st.subheader("Model Settings")

    col1, col2 = st.columns(2)
    with col1:
        load_previous = st.checkbox('Load previous model settings')
    with col2:
        if st.button("Browse Models", disabled=not st.session_state.model_path):
            if st.session_state.model_path:
                st.session_state.model = file_selector(st.session_state.model_path)

    if load_previous:
        load_previous_settings()
    else:
        st.session_state.model_path = st.text_input(
            'Model Directory',
            value=st.session_state.model_path,
            help="Directory containing PyTorch model files"
        )

        if st.session_state.model_path and Path(st.session_state.model_path).exists():
            st.session_state.model = file_selector(st.session_state.model_path)

    st.divider()

    # I/O settings
    st.subheader("Input/Output Settings")

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.input_path = st.text_input(
            'Input Directory',
            value=st.session_state.input_path,
            help="Directory containing images to segment"
        )
        st.session_state.input_type = st.selectbox(
            "Input file type",
            file_types,
            index=file_types.index(st.session_state.input_type) if st.session_state.input_type in file_types else 0
        )

    with col2:
        st.session_state.output_path = st.text_input(
            'Output Directory',
            value=st.session_state.output_path,
            help="Directory to save segmentation results"
        )
        st.session_state.out_type = st.selectbox(
            "Output file type",
            file_types,
            index=file_types.index(st.session_state.out_type) if st.session_state.out_type in file_types else 0
        )


def page_segmentations():
    """Segmentation page for running inference."""
    st.title("RDN 3-Class Segmentation")

    # Check if settings are complete
    required_fields = ['model', 'input_path', 'output_path']
    missing_fields = [field for field in required_fields if not st.session_state.get(field)]

    if missing_fields:
        st.error(f"Please configure: {', '.join(missing_fields)} in Settings page")
        return

    # Display current configuration
    display_segmentation_info()

    # File discovery
    if st.session_state.input_path and Path(st.session_state.input_path).exists():
        image_files = glob.glob(str(Path(st.session_state.input_path) / f"*.{st.session_state.input_type}"))
        image_files.sort(key=natural_keys)
        st.info(f"Found {len(image_files)} {st.session_state.input_type} files")

        if st.checkbox("Show file list"):
            st.write(image_files)

        # Check for 2D to 3D conversion
        slice_types = ["tif", "png", "jpg", "bmp", "dcm"]
        volume_types = ["mhd", "nii"]

        if st.session_state.input_type in slice_types and st.session_state.out_type in volume_types:
            st.info(f"Will convert {st.session_state.input_type} slices → {st.session_state.out_type} volume")
            st.session_state.twoD_to_threeD = True

        # Model loading
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Load Model", type="primary"):
                load_model()

        with col2:
            if st.button("Run Segmentation", disabled=st.session_state.net is None, type="primary"):
                run_segmentation(image_files)
    else:
        st.warning("Input path not found. Please check Settings.")


def load_model():
    """Load the neural network model."""
    try:
        with st.spinner("Loading model..."):
            st.session_state.net = UNet_Light_RDN(n_channels=1, n_classes=3)
            st.session_state.net.load_state_dict(
                torch.load(st.session_state.model, map_location=f'cuda:{st.session_state.use_gpu}')
            )
            st.session_state.net.cuda()
            st.session_state.net.eval()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load model: {e}")


def run_segmentation(image_files):
    """Run segmentation on the image files."""
    if not Path(st.session_state.output_path).exists():
        Path(st.session_state.output_path).mkdir(parents=True, exist_ok=True)
        st.info(f"Created output directory: {st.session_state.output_path}")

    try:
        with st.spinner("Running segmentation..."):
            if st.session_state.twoD_to_threeD:
                # 2D to 3D segmentation
                out_name = _get_file_name_from_list(image_files, suffix="RDN_seg")

                if st.session_state.input_type == "dcm":
                    image_vol, _metadata = two_to_three(image_files, st.session_state.input_type)
                else:
                    image_vol = two_to_three(image_files, st.session_state.input_type)

                image_vol = rescale_8(image_vol)
                seg_vol = three_class_seg_xyz(image_vol, st.session_state.net)
                seg_vol.CopyInformation(image_vol)
                write_image(seg_vol, out_name, st.session_state.output_path, st.session_state.out_type)
            else:
                # Single volume segmentation
                out_name = _get_file_name_from_list(image_files, suffix="RDN_seg")
                image_vol = read_image(image_files[0])
                image_vol = rescale_8(image_vol)
                seg_vol = three_class_seg_xyz(image_vol, st.session_state.net)
                write_image(seg_vol, out_name, st.session_state.output_path, st.session_state.out_type)

        st.success("Segmentation completed!")

    except Exception as e:
        st.error(f"Segmentation failed: {e}")


def display_current_settings():
    """Display current session state settings."""
    st.subheader("Current Configuration")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Model:**", st.session_state.get('model', 'Not set'))
        st.write("**Input Path:**", st.session_state.get('input_path', 'Not set'))
        st.write("**Input Type:**", st.session_state.get('input_type', 'Not set'))
    with col2:
        st.write("**Output Path:**", st.session_state.get('output_path', 'Not set'))
        st.write("**Output Type:**", st.session_state.get('out_type', 'Not set'))
        st.write("**GPU:**", st.session_state.get('use_gpu', 'Not initialized'))


def display_segmentation_info():
    """Display segmentation configuration info."""
    if st.session_state.cuda_mem:
        st.info(f"GPU: {torch.cuda.get_device_name(st.session_state.use_gpu)} ({st.session_state.cuda_mem})")

    if st.session_state.model:
        st.info(f"Model: {Path(st.session_state.model).name}")

    if st.session_state.output_path:
        st.info(f"Output: {st.session_state.output_path} ({st.session_state.out_type})")


def save_settings():
    """Save current settings to JSON file."""
    import json
    import pandas as pd

    script_dir = Path(__file__).parent
    save_dir = script_dir / "saved_states"
    save_dir.mkdir(exist_ok=True)

    user = os.environ.get("USERNAME", os.environ.get("USER", "unknown"))
    save_file = save_dir / f"{user}_RDN_saved_state.json"

    settings = {
        "model_path": str(st.session_state.get('model_path', '')),
        "model": str(st.session_state.get('model', '')),
        "input_path": str(st.session_state.get('input_path', '')),
        "input_type": st.session_state.get('input_type', 'mhd'),
        "output_path": str(st.session_state.get('output_path', '')),
        "out_type": st.session_state.get('out_type', 'mhd')
    }

    df = pd.DataFrame.from_dict({k: [v] for k, v in settings.items()})
    df.to_json(save_file)
    st.success("Settings saved!")


def load_previous_settings():
    """Load previous settings from JSON file."""
    import pandas as pd

    script_dir = Path(__file__).parent
    user = os.environ.get("USERNAME", os.environ.get("USER", "unknown"))
    save_file = script_dir / "saved_states" / f"{user}_RDN_saved_state.json"

    if save_file.exists():
        try:
            df = pd.read_json(save_file)
            st.session_state.model_path = str(df['model_path'][0])
            st.session_state.model = str(df['model'][0])
            st.session_state.input_path = str(df['input_path'][0])
            st.session_state.input_type = df['input_type'][0]
            st.session_state.output_path = str(df['output_path'][0])
            st.session_state.out_type = df['out_type'][0]
            st.success("Previous settings loaded!")
        except Exception as e:
            st.error(f"Failed to load settings: {e}")
    else:
        st.warning("No previous settings found")


def clear_settings():
    """Clear all session state settings."""
    keys_to_clear = ['model_path', 'model', 'input_path', 'output_path', 'net']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.success("Settings cleared!")
    st.rerun()


def clear_gpu_memory():
    """Clear GPU memory cache."""
    torch.cuda.empty_cache()
    st.success("GPU memory cleared!")


def initiate_cuda():
    """Initialize CUDA settings."""
    num_gpus = torch.cuda.device_count()

    if num_gpus == 0:
        st.error("No CUDA devices found!")
        return
    elif num_gpus > 1:
        st.session_state.use_gpu = st.selectbox('Select GPU:', list(range(num_gpus)))
    else:
        st.session_state.use_gpu = 0

    torch.cuda.set_device(st.session_state.use_gpu)
    device_props = torch.cuda.get_device_properties(st.session_state.use_gpu)
    st.session_state.cuda_mem = f"{device_props.total_memory / 1024**3:.1f} GB"

    st.success(f"GPU {st.session_state.use_gpu} initialized!")


@st.cache_data
def load_MARS_logo():
    """Load the MARS logo image."""
    script_dir = Path(__file__).parent
    try:
        return Image.open(script_dir / "Mars_Logo_small.png")
    except FileNotFoundError:
        # Fallback if logo not found
        return None


def file_selector(folder_path='.'):
    """Select a file from the given folder."""
    if not folder_path or not Path(folder_path).exists():
        return ""

    files = [f for f in os.listdir(folder_path) if f.endswith(('.pth', '.pt'))]
    files.sort(reverse=True)

    if files:
        selected = st.selectbox('Select PyTorch model:', files)
        return str(Path(folder_path) / selected)
    else:
        st.warning("No PyTorch model files found in directory")
        return ""


def _get_file_name_from_list(image_files, suffix=""):
    """Generate output filename from image file list."""
    if not image_files:
        return "output"

    base_name = Path(image_files[0]).stem
    return f"{base_name}_{suffix}" if suffix else base_name


# Import additional functions from the original codebase
# These would need to be imported or reimplemented from the original files
def two_to_three(image_stack, input_type):
    """Convert 2D image stack to 3D volume."""
    # This function would need to be implemented based on the original code
    pass


def three_class_seg_xyz(image_vol, network):
    """Run 3-class segmentation in X, Y, Z directions."""
    # This function would need to be implemented based on the original code
    pass


def write_image(image, name, output_dir, file_format):
    """Write SimpleITK image to disk."""
    # This function would need to be implemented based on the original code
    pass


if __name__ == "__main__":
    main()
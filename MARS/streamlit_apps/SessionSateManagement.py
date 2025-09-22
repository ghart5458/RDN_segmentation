import base64
import glob
import math
import os
import pathlib
import sys
import time
from timeit import default_timer as timer

import SimpleITK as sitk
import streamlit as st
import torch
from PIL import Image
from streamlit.hashing import _CodeHasher
from streamlit.ReportThread import get_report_ctx
from streamlit.server.Server import Server

#script_dir = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(r"D:\Desktop\git_repo")
from MARS.morphology.segmentation.pytorch_segmentation.execute_3_class_seg import (
    _get_outDir,
    _return_predictors,
    _save_predictors,
    natural_keys,
    read_image,
    rescale_8,
)
from MARS.morphology.segmentation.pytorch_segmentation.net.unet_light_rdn import UNet_Light_RDN


def main():
    logo = load_MARS_loggo(0)['MARS']
    st.sidebar.image(logo, width=50)

    pages = {
        "Settings": page_settings,
        "Segmentation": page_segmentations
    }
    state = _get_state()

    st.sidebar.markdown(
    """

    # ** <span style="color:red; font-size:2em"> MARS:</span> ** #
    RDN 3-class segmentation
    """, unsafe_allow_html=True)

    page = st.sidebar.radio("Navigation:", tuple(pages.keys()))

    if st.sidebar.button("Save settings"):
        save_state_values(state=state)

    if st.sidebar.button("Clear settings"):
        state.clear()
        st.write("Settings cleared.")

    if st.sidebar.button("Clear GPU memory"):
        torch.cuda.empty_cache()
        st.write("GPU memory cleared.")

    # Display the selected page with the session state
    pages[page](state)

    # Mandatory to avoid rollbacks with widgets, must be called at the end of your app
    state.sync()

def page_settings(state):
    file_types = ["mhd", "nii", "tif", "png", "jpg", "bmp", "dcm"]

    st.title(":wrench: Settings")
    display_state_values(state)

    st.write(f"Segmenting with {torch.cuda.get_device_name(state.use_gpu)}, {state.cuda_mem} of memory.")
    if st.button('Initiate GPU'):
        initiate_cuda(state)

    st.write("---")
    st.subheader("Model settings")

    if st.checkbox('Load previous model settings'):
        restored_state = load_state_values(state)
        state.model_path = restored_state.model_path
        st.write(state.model_path)
        state.model = pathlib.Path(restored_state.model)
        st.write(pathlib.Path(state.model).parts[-1])
    elif st.checkbox("Just the model path"):
        restored_state = load_state_values(state)
        state.model_path = restored_state.model_path
        state.model = file_selector(state.model_path)
    else:
        state.model_path = pathlib.Path(st.text_input('Model location', ""))
        state.model = file_selector(state.model_path)

    st.write("---")
    st.subheader("Input/Output settings")
    if st.checkbox('Load previous input/output settings'):
        restored_state = load_state_values(state)
        if st.checkbox("Previous input path"):
            state.input_path = pathlib.Path(restored_state.input_path)
            st.write(state.input_path)
        else:
            state.input_path = pathlib.Path(st.text_input('Input location', ""))

        if st.checkbox("Previous input type"):
            state.input_type = restored_state.input_type
            st.write(state.input_type)
        else:
            state.input_type = st.selectbox("Input file types", file_types,
                                            file_types.index(state.input_type) if state.input_type else 0)

        if st.checkbox("Previous output path"):
            state.output_path = pathlib.Path(restored_state.output_path)
            st.write(state.output_path)
        else:
            state.output_path = pathlib.Path(st.text_input('Output location', ""))

        if st.checkbox("Previous output type"):
            state.out_type = restored_state.out_type
            st.write(state.out_type)
        else:
            state.out_type = st.selectbox("Select value.", file_types,
                                          file_types.index(state.out_type) if state.out_type else 0)
    else:
        state.input_path = pathlib.Path(st.text_input('Input location', ""))
        state.input_type = st.selectbox("Input file type", file_types,
                                        file_types.index(state.input_type) if state.input_type else 0)
        state.output_path = pathlib.Path(st.text_input('Output location', ""))
        state.out_type = st.selectbox("Output file type", file_types,
                                      file_types.index(state.out_type) if state.out_type else 0)

def page_segmentations(state):
    slice_types = ["tif", "png", "jpg", "bmp", "dcm"]
    volume_types = ["mhd", "nii"]
    segmentation_state_values(state)
    #st.write(state._state)
    if len(state._state["data"]) == 0:
        st.error("Must fill out settings!")
    else:
        input_path = pathlib.Path(state.input_path)
        input_type = state.input_type

        output_path = pathlib.Path(state.output_path)
        out_type = state.out_type

        if not output_path.exists():
            st.warning(f"{output_path} doesn't exist and will be created!")

        if st.button("Load model!"):
            state.net = UNet_Light_RDN(n_channels=1, n_classes=3)
            # Load in the trained model
            state.net.load_state_dict(torch.load(state.model, map_location=f'cuda:{state.use_gpu}'))
            state.net.cuda()
            state.net.eval()
            st.info("Model loaded!")

        image_files = glob.glob(str(input_path.joinpath(f"*.{input_type}")))
        image_files.sort(key=natural_keys)
        st.write(f"Found {len(image_files)} image files for segmentation in {input_path}...")

        if st.checkbox("Show image list"):
            st.write("Images to segment:")
            st.write(image_files)

        if input_type in slice_types and out_type in volume_types:
            st.info(f"{input_type} is assumed to be slices and will be converted to 3d output type: {out_type}")
            state.twoD_to_threeD = True

        if st.button('Segment!'):
            if not output_path.exists():
                pathlib.Path.mkdir(output_path)
            net = state.net
            if state.twoD_to_threeD:
                out_name = _get_file_name_from_list(image_files, suffix="RDN_seg")
                if input_type == "dcm":
                    image_vol, _metadata = two_to_three(image_stack=image_files, input_type=input_type)
                else:
                    image_vol = two_to_three(image_stack=image_files, input_type=input_type)
                image_vol = rescale_8(image_vol)
                st.write("Segmenting....")
                seg_vol = three_class_seg_xyz(inputImage=image_vol, network=net)
                #seg_vol = three_class_segmentation_volume(inputImage=image_vol, direction="z", network=net)
                seg_vol.CopyInformation(image_vol)
                write_image(inputImage=seg_vol, outName=out_name, outDir=output_path, fileFormat=out_type)


            elif True:
                out_name = _get_file_name_from_list(image_files, suffix="RDN_seg")
                image_vol = read_image(inputImage=image_files[0])
                image_vol = rescale_8(image_vol)

                st.write("Segmenting....")
                seg_vol = three_class_seg_xyz(inputImage=image_vol, network=net)
                write_image(inputImage=seg_vol, outName=out_name, outDir=output_path, fileFormat=out_type)
            #else:
            #    three_class_segmentation(input_image=image_files, outDir=output_path, outType=out_type, network=net)

def display_state_values(state):
    st.write("Model:", state.model)
    st.write("Input path:", state.input_path)
    st.write("Input file type:", state.input_type)
    st.write("Output path:", state.output_path)
    st.write("Output file type:", state.out_type)

def segmentation_state_values(state):
    st.title("RDN 3-class segmentation:")
    st.info(f"Segmenting with {torch.cuda.get_device_name(state.use_gpu)}, {state.cuda_mem} of memory.")
    st.info(f"Model path: {state.model}")
    st.info(f"Segmentations will be written to {state.output_path!s} in {state.out_type} file format")

def save_state_values(state):
    script_dir = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
    saved_dir = script_dir.joinpath("saved_state.json")
    session_dict = {"model_path": [str(state.model_path)],
                    "model": [str(state.model)],
                    "input_path": [str(state.input_path)],
                    "input_type": [state.input_type],
                    "output_path": [str(state.output_path)],
                    "out_type": [str(state.out_type)]}
    session_dict = pd.DataFrame.from_dict(session_dict)
    session_dict.to_json(saved_dir)
    st.write("Saved settings!")

def load_state_values(state):
    script_dir = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
    saved_dir = script_dir.joinpath("saved_state.json")
    restored_session = pd.read_json(str(saved_dir))
    st.write("Restored previous settings")
    state.model_path = str(restored_session['model_path'][0])
    state.model = restored_session['model'][0]
    state.input_path = restored_session['input_path'][0]
    state.input_type = restored_session['input_type'][0]
    state.output_path = pathlib.Path(restored_session['output_path'][0])
    state.out_type = restored_session['out_type'][0]
    return state

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    filenames.sort(reverse=True)
    selected_filename = st.selectbox('Pytorch model', filenames)
    return os.path.join(folder_path, selected_filename)

def _get_file_name_from_list(image_files, suffix=""):
    outName = pathlib.Path(image_files[0]).parts[-1]
    outName = outName.split(".")[0]
    if suffix != "":
        outName = f"{outName}_{suffix}"
    return outName

def render_svg(svg_file):

    with open(svg_file) as f:
        lines = f.readlines()
        svg = "".join(lines)

        """Renders the given svg string."""
        b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
        html = rf'<img src="data:image/svg+xml;base64,{b64}"/>'
        return html

def initiate_cuda(state):
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        st.write("No cuda device found!")
    elif num_gpus > 1:
        st.write("Multiple gpu's found...")
        state.use_gpu = gpu_selector(num_gpus=num_gpus)
    else:
        state.use_gpu = 0
    torch.cuda.set_device(state.use_gpu)
    state.cuda_mem = str(torch.cuda.get_device_properties(state.use_gpu)).split(",")[-2].split("=")[-1]

def gpu_selector(num_gpus):
    gpu_list = list(range(num_gpus))
    selected_gpu = st.selectbox('Select the gpu', gpu_list)
    return selected_gpu

def model_initiation(model_path, cuda_index):
    net = UNet_Light_RDN(n_channels=1, n_classes=3)
    # Load in the trained model
    net.load_state_dict(torch.load(model_path, map_location=f'cuda:{int(cuda_index)}'))
    net.cuda()
    net.eval()
    return(net)

def _convert_size(sizeBytes):
    """
    Function to return file size in a human readable manner.
    :param sizeBytes: bytes calculated with file_size
    :return:
    """
    if sizeBytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = math.floor(math.log(sizeBytes, 1024))
    p = math.pow(1024, i)
    s = round(sizeBytes / p, 2)
    return f"{s} {size_name[i]}", s

def _file_size(dim1, dim2, dim3, bits):
    """
    Get the file size of an image volume from the x, y, and z dimensions.
    :param dim1: The x dimension of an image file.
    :param dim2: The y dimension of an image file.
    :param dim3: The z dimension of an image file.
    :param bits: The type of bytes being used (e.g. unsigned 8 bit, float 32, etc.)
    :return: Returns the size in bytes.
    """
    if bits == 8:
        bit = 1
    elif bits == 16:
        bit = 2
    elif bits == 32:
        bit = 4
    else:
        bit = 8
    file_size = dim1 * dim2 * dim3 * bit
    file_s = _convert_size(sizeBytes=file_size)
    st.text(f"file size: {file_s[0]}")
    size = int(np.ceil(file_s[1]))
    return size

def _print_info(inputImage):
    """
    Function to return the basic information of an image volume.
    :param inputImage: A SimpleITK formated image volume.
    """
    image_type = inputImage.GetPixelIDTypeAsString()
    size = inputImage.GetSize()
    xdim, ydim, zdim = size[0], size[1], size[2]
    xres, yres, zres = inputImage.GetSpacing()
    if image_type == "8-bit unsigned integer":
        bits = 8
    elif True:
        bits = 16
    elif True:
        bits = 32
    else:
        bits = 64
    _file_size(xdim, ydim, zdim, bits)
    st.text(f"{image_type}")
    st.text(f"Dimensions, x:{xdim:5}, y:{ydim:5}, z:{zdim:5}")
    st.text(f"Resolution, x:{xres:5}, {yres:5}, {zres:5}")

def _setup_sitk_image(image_slice, direction="z"):
    """
    Internal function to read in an image and setup for classification by pytorch.
    """
    # Open the image using pillow and ensure it is grey scale ('L'), then turn it into a numpy array

    direction = str(direction).lower()

    #Convert the image slice into a numpy array
    image = sitk.GetArrayFromImage(image_slice)

    # Deal with the variation in the 3d versus 2d array.
    if len(image.shape) == 2:
        if direction == "z":
            #Expand the z axis
            image = np.expand_dims(image, axis=2)
            # Check the dimensionality of the image, expand, transpose, for pytorch.
            image = image.transpose((2, 0, 1))
        elif direction == "y":
            image = np.expand_dims(image, axis=1)
            image = image.transpose((1, 0, 2))
        else:
            image = np.expand_dims(image, axis=0)
    return image


def read_image(inputImage):
    """
    Reads in various image file formats (mha, mhd, nia, nii, vtk, etc.) and places them into a SimpleITK volume format.
    :param inputImage: Either a volume (mhd, nii, vtk, etc.).
    :return: Returns a SimpleITK formatted image object.
    """
    st.write(f"Reading in {inputImage}.")
    start = timer()
    inputImage = sitk.ReadImage(str(inputImage))
    _end_timer(start, message="Reading in the image")
    _print_info(inputImage)
    st.write("\n")
    return inputImage

def rescale_8(inputImage):
    """
    Takes in a SimpleITK image and rescales it to 8 bit.
    :param inputImage: A SimpleITK formatted volume.
    :return: Returns an unsigned 8-bit SimpleITK formatted volume with gray values scaled between 0-255.
    """

    # Check to see if it is already unisgned 8 bit.
    imageType = inputImage.GetPixelID()
    if imageType == 1:
        st.write("Image is already unsigned 8...")
        scaled_8 = inputImage

    #If it isn't, go ahead and rescale.
    else:
        st.write("Rescaling to unsigned 8...")
        start = timer()
        scaled_8 = sitk.Cast(sitk.RescaleIntensity(inputImage), sitk.sitkUInt8)
        _print_info(scaled_8)
        _end_timer(start, message="Rescaling to unsigned 8")
    return scaled_8

def write_image(inputImage, outName, outDir="", fileFormat="mhd"):
    """
    Writes out a SimpleITK image in any supported file format (mha, mhd, nii, dcm, tif, vtk, etc.).
    :param inputImage: SimpleITK formated image volume
    :param outName: The file name
    :param outDir: The directory where the file should be written to. If not path is provided the current directory will
    be used.
    :param fileFormat: The desired file format. If no file format is provided, mhd will be used.
    :return: Returns an image file written to the hard disk.
    """
    start = timer()
    outName = str(outName)
    outDir == _get_outDir(outDir)
    fileFormat = str(fileFormat)

    fileFormat = fileFormat.replace(".", "")
    outputImage = pathlib.Path(outDir).joinpath(str(outName) + "." + str(fileFormat))

    _print_info(inputImage)
    st.write(f"Writing {outName} to {outDir} as {fileFormat}.")
    sitk.WriteImage(inputImage, str(outputImage))
    _end_timer(start, message="Writing the image")

def read_stack(inputStack):
    """
    Reads in a series of images and then places them into a SimpleITK volume format.

    :param inputStack: A stack of images (e.g. tif, png, etc).
    :return: Returns a SimpleITK formatted image object.
    """
    # Read in the other image and recast to float 32
    start = timer()
    st.write("Reading in files...")
    inputStack.sort(key=natural_keys)
    inputStack = sitk.ReadImage(inputStack)
    _end_timer(start, message="Reading in the stack")
    _print_info(inputStack)
    print("\n")
    return inputStack


def read_dicom(inputStack):
    """
    Specialized DICOM reader that preserves the metadata tags in the dicom files.
    :param inputStack: Dicom stack.
    :return: Returns a SimpleITK image and a list containing the metadata.
    note that tag 0020|000e is a unique identifier that may be modified in the returned metadata. Otherwise the data
    and time are used.
    """

    start = timer()
    st.write(f"Reading in {len(inputStack)} DICOM images...")

    inputStack.sort(key=natural_keys)
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(inputStack)
    series_reader.MetaDataDictionaryArrayUpdateOn()
    series_reader.LoadPrivateTagsOn()
    sitk_image = series_reader.Execute()

    _print_info(sitk_image)

    #Grab the metadata, Name, ID, DOB, etc.
    direction = sitk_image.GetDirection()
    tags_to_copy = ["0010|0010", "0010|0020", "0010|0030", "0020|000D", "0020|0010",
                    "0008|0020", "0008|0030", "0008|0050", "0008|0060", "0028|0030"]
    process_tag = ["0008|103e"]

    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")

    series_tag_values = [(k, series_reader.GetMetaData(0, k)) for k in tags_to_copy if series_reader.HasMetaDataKey(0, k)]


    modified_tags = [("0008|0031", modification_time), ("0008|0021", modification_date), ("0008|0008", "DERIVED\\SECONDARY"),
                     ("0020|000e", "" + modification_date + ".1" + modification_time),
                     ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],
                                                       direction[1], direction[4], direction[7]))))]

    series_tag_values = series_tag_values + modified_tags

    #Inset the new processing data
    if series_reader.HasMetaDataKey(0, process_tag[0]):
        series_tag_values = [*series_tag_values, ("0008|103e", series_reader.GetMetaData(0, "0008|103e") + " Processed-SimpleITK")]
    else:
        series_tag_values = [*series_tag_values, ("0008|103e", "Processed-SimpleITK")]

    #To prevent the stacking of the same processing information
    if series_tag_values[-1] == ('0008|103e', 'Processed-SimpleITK  Processed-SimpleITK'):
        series_tag_values[-1] = ("0008|103e", "Processed-SimpleITK")
    _end_timer(start_timer=start, message="Reading DICOM stack")
    return sitk_image, series_tag_values

def write_dicom(inputImage, metadata, outName, outDir=""):
    """
    """
    #Modified from: https://simpleitk.readthedocs.io/en/master/link_DicomSeriesReadModifyWrite_docs.html

    start = timer()
    series_tag_values = metadata
    outDir = pathlib.Path.cwd() if outDir == "" else pathlib.Path(outDir)

    #Make is so the file name generator deal with these parts of the name
    if outName[-4] == ".dcm":
        outName = outName[:-4]

    if outName[-1] == "_":
        outName = outName[:-1]

    outName = pathlib.Path(outDir).joinpath(outName)
    slice_num = inputImage.GetDepth()

    # Use the study/series/frame of reference information given in the meta-data
    # dictionary and not the automatically generated information from the file IO
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    digits_offset = len(str(slice_num))

    for i in range(slice_num):
        image_slice = inputImage[:, :, i]

        # Tags shared by the series.
        for tag, value in series_tag_values:
            image_slice.SetMetaData(tag, value)
        # Slice specific tags.
        #   Instance Creation Date
        image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))
        #   Instance Creation Time
        image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))
        #   Image Position (Patient)
        image_slice.SetMetaData("0020|0032", '\\'.join(map(str, inputImage.TransformIndexToPhysicalPoint((0, 0, i)))))
        #   Instance Number
        image_slice.SetMetaData("0020|0013", str(i))

        # Write to the output directory and add the extension dcm, to force writing
        # in DICOM format.
        writer.SetFileName(f'{outName!s}_{i:0{int(digits_offset)}}.dcm')
        writer.Execute(image_slice)
    print("\n")
    _end_timer(start, message="Writing DICOM slices")

def two_to_three(image_stack, input_type):
    image_stack.sort(key=natural_keys)
    st.write("Converting image stack...")
    if input_type == "dcm":
        image_vol, metadata = read_dicom(image_stack)
        resolution = image_vol.GetSpacing()
        if resolution[2] == 1.0:
            st.write(f"Setting z resolution to {resolution[1]} to match x and y resolution....")
            image_vol.SetSpacing((resolution[0], resolution[1], resolution[1]))
        return image_vol, metadata
    elif input_type in ["tif", "png", "jpg", "bmp"]:
        image_vol = read_stack(image_stack)
        resolution = image_vol.GetSpacing()
        image_vol.SetSpacing((resolution[0], resolution[1], resolution[1]))
        return image_vol
    else:
        st.markdown(f"Input type {input_type} not supported :frowning:")

def three_class_segmentation(input_image, outDir, outType, network=""):
    """
    Function to segment a directory of 2d images using a pytorch model
    Images must be in a pillow readable format (e.g. "tif", "png", "jpg", "bmp")
    :param inDir: The input directory where the images are located
    :param outDir: The output directory. If this doesn't exist it will be created.
    :param outType: The output file type. Supported type are tif, png, jpg, and bmp.
    :return: Returns a segmented 2d image with grey values representing air, dirt, and bone.
    """
    start = timer()
    save_folder = outDir

    net = network

    # The file types that can be output along with the corresponding dictionary
    if not pathlib.Path(outDir).exists():
        pathlib.Path.mkdir(save_folder)

    # Get a list of files from the input folder using a list comprehension approach, then sort them numerically.
    image_names = input_image
    image_names.sort()

    st.write(f"Processing {len(image_names)} images...")

    progress_bar = st.progress(0)
    # Loop through the images in the folder and use the image name for the output name
    for i in range(len(image_names)):
        image_name = image_names[i]

        out_name = str(pathlib.Path(image_name).parts[-1]) if True else image_name

        if "." in out_name:
            out_name = out_name.rsplit(".", 1)[0]


        #Read the image in with pillow and set it as a numpy array for pytorch
        image = sitk.ReadImage(str(image_name))
        image = _setup_sitk_image(image, direction="z")

        #Pass the numpy array to pytorch, convert to a float between 0-1,then copy into cuda memory for classifcation.
        image = torch.from_numpy(image)
        image = image.unsqueeze(0).float() / 255.0
        image = image.cuda()

        #Turn all the gradients to false and get the maximum predictors from the network
        with torch.no_grad():
            pred = net(image)
        pred = pred.argmax(1)
        pred = pred.cpu().squeeze().data.numpy()

        #Pass the predictions to be saved using pillow
        _save_predictors(pred=pred, save_folder=outDir, image_name=out_name, file_type=outType)
        iteration = np.floor(100 * ((i + 1)/len(image_names)))
        progress_bar.progress(int(iteration))

    st.write('\n\nSegmentations are done!\n\n')
    _end_timer(start_timer=start, message="Segmentations")

@st.cache(allow_output_mutation=True)
def load_MARS_loggo(self):
    logo = {}
    logo['MARS'] = Image.open(r'D:/Desktop/git_repo/Setup/mars_icon.bmp')
    return logo

def three_class_segmentation_volume(inputImage, direction="z", network=""):
    """
    Function to segment a directory of 2d images using a pytorch model
    Images must be in a pillow readable format (e.g. "tif", "png", "jpg", "bmp")
    :param inDir: The input directory where the images are located
    :param outDir: The output directory. If this doesn't exist it will be created.
    :param outType: The output file type. Supported type are tif, png, jpg, and bmp.
    :return: Returns a segmented 2d image with grey values representing air, dirt, and bone.
    """
    net = network
    start = timer()
    # Set up the count so there's something to watch as it processes
    direction = str(direction).lower()

    if direction == "z":
        seg_count = inputImage.GetSize()[2]
    elif direction == "y":
        seg_count = inputImage.GetSize()[1]
    else:
        seg_count = inputImage.GetSize()[0]
    print(f"Processing {seg_count} {direction} slices...")

    # Create an empty volume to stuff the results into. A numpy approach was tested but proved to be slower
    vol_image = sitk.Image(inputImage.GetSize(), sitk.sitkUInt8)

    progress_bar = st.progress(0)
    # Loop through the images in the folder and use the image name for the output name
    for i in range(seg_count):
        image = feed_slice(inputImage, slice=i, direction=str(direction))

        #Read the image in with pillow and set it as a numpy array for pytorch
        image = _setup_sitk_image(image_slice=image, direction=direction)

        #Pass the numpy array to pytorch, convert to a float between 0-1,then copy into cuda memory for classifcation.
        image = torch.from_numpy(image)
        image = image.unsqueeze(0).float() / 255.0
        image = image.cuda()

        #Turn all the gradients to false and get the maximum predictors from the network
        with torch.no_grad():
            pred = net(image)
        pred = pred.argmax(1)
        pred = pred.cpu().squeeze().data.numpy()

        #Pass the predictions to be saved using pillow.
        pred = _return_predictors(pred=pred, direction=direction)
        slice_vol = sitk.GetImageFromArray(pred)
        #slice_vol = sitk.JoinSeries(slice)
        if direction == "z":
            vol_image = sitk.Paste(vol_image, slice_vol, slice_vol.GetSize(), destinationIndex=[0, 0, i])
        elif direction == "y":
            vol_image = sitk.Paste(vol_image, slice_vol, slice_vol.GetSize(), destinationIndex=[0, i, 0])
        else:
            vol_image = sitk.Paste(vol_image, slice_vol, slice_vol.GetSize(), destinationIndex=[i, 0, 0])

        iteration = np.floor(100 * ((i + 1)/seg_count))
        progress_bar.progress(int(iteration))

    _end_timer(start_timer=start, message=f"{direction}-plane segmentations")
    return vol_image


def three_class_seg_xyz(inputImage, network=""):
    progress_bar = st.progress(0)
    steps = 6
    #Segment the volume from all three directions
    seg_z = three_class_segmentation_volume(inputImage=inputImage, direction="z", network=network)

    iteration = np.floor(100 * (1 / steps))
    progress_bar.progress(int(iteration))

    seg_y = three_class_segmentation_volume(inputImage=inputImage, direction="y", network=network)

    iteration = np.floor(100 * (2 / steps))
    progress_bar.progress(int(iteration))

    seg_x = three_class_segmentation_volume(inputImage=inputImage, direction="x", network=network)

    iteration = np.floor(100 * (3 / steps))
    progress_bar.progress(int(iteration))


    #Rescale them to prevent overflow when we combine
    seg_z = rescale_16(seg_z)
    seg_y = rescale_16(seg_y)
    seg_z = combine_images(seg_z, seg_y)

    iteration = np.floor(100 * (4 / steps))
    progress_bar.progress(int(iteration))


    # Free up memory
    seg_y = 0
    seg_x = rescale_16(seg_x)

    seg_z = combine_images(seg_z, seg_x)

    iteration = np.floor(100 * (5 / steps))
    progress_bar.progress(int(iteration))

    seg_x = 0
    #Get the final product
    seg = rescale_8(seg_z)

    iteration = np.floor(100 * (6 / steps))
    progress_bar.progress(int(iteration))

    return seg

def _end_timer(start_timer, message=""):
    """
    Simple function to print the end of a timer in a single line instead of being repeated.
    :param start_timer: timer start called using timer() after importing: from time import time as timer.
    :param message: String that makes the timing of what event more clear (e.g. "segmenting", "meshing").
    :return: Returns a sring mesuraing the end of a timed event in seconds.
    """
    start = start_timer
    message = str(message)
    end = timer()
    elapsed = abs(start - end)
    if message == "":
        st.text(f"Operation took: {float(elapsed):10.4f} seconds")
    else:
        st.text(f"{message} took: {float(elapsed):10.4f} seconds")

class _SessionState:

    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()

    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False

        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)


def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")

    return session_info.session


def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state


if __name__ == "__main__":
    main()

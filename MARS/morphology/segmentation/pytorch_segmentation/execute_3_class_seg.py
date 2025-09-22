"""Script to run the 3_class model in the command line. This should be executed using the pytorch_seg conda environment

Author: Sun, Yung-Chen yzs5463@psu.edu
Author: Yazdani, Amirsaeed auy200@psu.edu
Author: Nick Stephens nbs49@psu.edu

#If you get a win 95 error reinstall prompt-toolkit
python -m pip install -U prompt-toolkit~=2.0

"""
import math
import multiprocessing
import os
import platform
import re
import socket
import sys
import time
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image
from tqdm import tqdm

# Image dimension constants
IMAGE_2D_DIMS = 2

if platform.system() == "Windows":
    if socket.gethostname() == 'L2ANTH-WT0023':
        sys.path.append(r"Z:\RyanLab\Projects\NStephens\git_repo")
    else:
        sys.path.append(r"D:\Desktop\git_repo")
if platform.system().lower() == 'linux':
    if 'redhat' in platform.platform():
        sys.path.append(r"/gpfs/group/LiberalArts/default/tmr21_collab/RyanLab/Projects/NStephens/git_repo")
    else:
        sys.path.append(r"/mnt/ics/RyanLab/Projects/NStephens/git_repo")

# Provide the location of the net folder. This will work until packaged.
script_dir = Path(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(str(script_dir))
#sys.path.append(r"Z:\RyanLab\Projects\NStephens\git_repo\MARS\morphology\segmentation\pytorch_segmentation")
#sys.path.append(r"D:\Desktop\git_repo\MARS\morphology\segmentation\pytorch_segmentation")

class ReportPosition(sitk.Command):
    """class object to report progress in a consistent way.
    #Taken from https://simpleitk.readthedocs.io/en/master/link_FilterProgressReporting_docs.html
    """

    def __init__(self, po):
        # required
        super().__init__()
        self.processObject = po

    def Execute(self):
        """Print progress information to the console.

        Returns:
            None: Prints progress percentage to console

        """
        print(f"\r           Progress:    {100 * self.processObject.GetProgress():03.1f}%", end='')

    def filterPosition(self, startorstop=""):
        """Print filter start or stop message.

        Args:
            startorstop (str): Either "start" or "stop" to indicate filter state

        Returns:
            None: Prints status message to console

        """
        if startorstop == 'start':
            print(f"\n{self.processObject.GetName()} executing....\nPlease stand by...")
        elif startorstop == 'stop':
            print("\nFiltering done!\n")
        else:
            print("Unknown command...")


def _setup_image(data_folder, image_name):
    """Internal function to read in an image and setup for classification by pytorch.
    """
    # Open the image using pillow and ensure it is grey scale ('L'), then turn it into a numpy array
    image = Image.open(os.path.join(data_folder, image_name)).convert('L')
    image = np.array(image)

    # Check the dimensionality of the image, expand, transpose, for pytorch.
    if len(image.shape) == IMAGE_2D_DIMS:
        image = np.expand_dims(image, axis=2)
    image = image.transpose((2, 0, 1))
    return image

def _end_timer(start_timer, message=""):
    """Print elapsed time for a timed operation.

    Args:
        start_timer (float): Timer start value from timer() function
        message (str): Optional description of the timed operation

    Returns:
        None: Prints elapsed time to console

    """
    start = start_timer
    message = str(message)
    end = timer()
    elapsed = abs(start - end)
    if message == "":
        print(f"Operation took: {float(elapsed):10.4f} seconds")
    else:
        print(f"{message} took: {float(elapsed):10.4f} seconds")

def _convert_size(sizeBytes):
    """Convert file size from bytes to human-readable format.

    Args:
        sizeBytes (int): File size in bytes

    Returns:
        tuple: (formatted_size_string, numeric_value) in appropriate units

    """
    if sizeBytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = math.floor(math.log(sizeBytes, 1024))
    p = math.pow(1024, i)
    s = round(sizeBytes / p, 2)
    return f"{s} {size_name[i]}", s

def _file_size(dim1, dim2, dim3, bits):
    """Calculate file size of an image volume from its dimensions.

    Args:
        dim1 (int): X dimension of the image
        dim2 (int): Y dimension of the image
        dim3 (int): Z dimension of the image
        bits (int): Bit depth (8, 16, 32, or 64)

    Returns:
        int: Estimated file size in appropriate units

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
    print(f"file size: {file_s[0]}")
    size = int(np.ceil(file_s[1]))
    return size

def _get_threads(threads):
    """Get thread count for parallel processing.

    Args:
        threads (str or int): Either "threads" for auto-detection or specific count

    Returns:
        int: Number of threads to use (CPU count - 1 for auto, otherwise specified value)

    """
    threads = int(multiprocessing.cpu_count()) - 1 if threads == "threads" else int(threads)
    return threads

def _print_info(inputImage):
    """Print basic information about an image volume.

    Args:
        inputImage (sitk.Image): SimpleITK formatted image volume

    Returns:
        None: Prints image information to console

    """
    image_type = inputImage.GetPixelIDTypeAsString()
    size = inputImage.GetSize()
    xdim, ydim, zdim = size[0], size[1], size[2]
    res = inputImage.GetSpacing()[0]
    if image_type == "8-bit unsigned integer":
        bits = 8
    elif True:
        bits = 16
    elif True:
        bits = 32
    else:
        bits = 64
    _file_size(xdim, ydim, zdim, bits)
    print(f"{image_type}\nx:{xdim} y:{ydim} z:{zdim}\nResolution:{res}\n")

def _setup_image(data_folder, image_name):
    """Internal function to read in an image and setup for classification by pytorch.
    """
    # Open the image using pillow and ensure it is grey scale ('L'), then turn it into a numpy array
    image = Image.open(os.path.join(data_folder, image_name)).convert('L')
    image = np.array(image)

    # Check the dimensionality of the image, expand, transpose, for pytorch.
    if len(image.shape) == IMAGE_2D_DIMS:
        image = np.expand_dims(image, axis=2)
    image = image.transpose((2, 0, 1))
    return image

def _save_predictors(pred, save_folder, image_name, file_type):
    """Internal function to convert predictions to an image and save in an output folder.
    """
    # The dictionary for the grey value means for each class.
    # This will results in 0 for air, 128 for dirt, and 255 for bone.
    color_dict = [[0.0], [128.0], [255.0]]

    #File type dictionary for pillow
    type_dict = {"tif": "TIFF", "png": "PNG", "jpg": "JPEG", "bmp": "BMP"}
    f_type = type_dict[str(file_type)]

    # Set up a blank numpy array to put the results into according to the values in the color_dict
    pred_img = np.zeros(pred.shape)
    for i in range(len(color_dict)):
        for _j in range(len(color_dict[i])):
            pred_img[pred == i] = color_dict[i][0]

    # Cast the data as unsigned 8 bit and reconstruct the image for writing.
    pred_img = pred_img.astype(np.uint8)
    pred_img = Image.fromarray(pred_img, 'L')
    pred_img.save(os.path.join(save_folder, f"{image_name[:-3]}.{file_type!s}"), str(f_type))

def _setup_sitk_image(image_slice, direction="z"):
    """Internal function to read in an image and setup for classification by pytorch.
    """
    # Open the image using pillow and ensure it is grey scale ('L'), then turn it into a numpy array

    direction = str(direction).lower()

    #Convert the image slice into a numpy array
    image = sitk.GetArrayFromImage(image_slice)

    # Deal with the variation in the 3d versus 2d array.
    if len(image.shape) == IMAGE_2D_DIMS:
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
            #image = image.transpose((0, 1, 2))
    return image

def _return_predictors(pred, direction="z"):
    """Internal function to convert predictions to an image and save in an output folder.
    """
    direction = str(direction).lower()

    # The dictionary for the grey value means for each class.
    # This will results in 0 for air, 128 for dirt, and 255 for bone.
    color_dict = [[0.0], [128.0], [255.0]]

    # Set up a blank numpy array to put the results into according to the values in the color_dict
    pred_img = np.zeros(pred.shape)
    for i in range(len(color_dict)):
        for _j in range(len(color_dict[i])):
            pred_img[pred == i] = color_dict[i][0]

    # Cast the data as unsigned 8 bit and reconstruct the image for writing.
    pred_array = pred_img.astype(np.uint8)

    if direction == "z":
        pred_array = np.expand_dims(pred_array, axis=0)
    elif direction == "y":
        pred_array = np.expand_dims(pred_array, axis=1)
    else:
        pred_array = np.expand_dims(pred_array, axis=2)
    return pred_array


def _get_outDir(outDir):
    """Validate and format output directory path using pathlib.

    Args:
        outDir (str): Output directory path (empty string uses current directory)

    Returns:
        Path: Pathlib Path object for the output directory

    """
    outDir = Path.cwd() if outDir == "" else Path(str(outDir))
    return outDir

def _get_inDir(inDir):
    """Validate and format input directory path using pathlib.

    Args:
        inDir (str): Input directory path (empty string uses current directory)

    Returns:
        Path: Pathlib Path object for the input directory

    """
    inDir = Path.cwd() if inDir == "" else Path(str(inDir))
    return inDir

def alpha_to_int(text):
    """Convert text to integer if it represents a digit, otherwise return as-is.

    Args:
        text (str): Input text that may represent an integer

    Returns:
        int or str: Integer value if text is numeric, otherwise original text

    """
    clean_text = int(text) if text.isdigit() else text
    return clean_text

def alpha_to_float(text):
    """Convert text to float if possible, otherwise return as-is.

    Args:
        text (str): Input text that may represent a float

    Returns:
        float or str: Float value if text is numeric, otherwise original text

    """
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    """Generate keys for natural sorting of strings with embedded numbers.

    Enables human-friendly sorting (e.g., ['file1', 'file2', 'file10'] instead
    of ['file1', 'file10', 'file2']).

    Args:
        text (str): Input string to generate sorting key for

    Returns:
        list: List of mixed integers and strings for natural sorting

    Reference:
        http://nedbatchelder.com/blog/200712/human_sorting.html

    """
    return [alpha_to_int(c) for c in re.split(r'(\d+)', text)]

def natural_keys_float(text):
    """Generate keys for natural sorting of strings with embedded floating point numbers.

    Similar to natural_keys but handles floating point numbers in addition to integers.

    Args:
        text (str): Input string to generate sorting key for

    Returns:
        list: List of mixed floats and strings for natural sorting

    References:
        http://nedbatchelder.com/blog/200712/human_sorting.html
        Float regex from https://stackoverflow.com/a/12643073/190597

    """
    return [alpha_to_float(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]

def read_image(inputImage):
    """Read various image file formats into SimpleITK format.

    Supports medical imaging formats including mha, mhd, nii, vtk, and others.
    Prints timing and basic image information during processing.

    Args:
        inputImage (str or Path): Path to image file to read

    Returns:
        sitk.Image: SimpleITK formatted image object

    """
    print(f"Reading in {inputImage}.")
    start = timer()
    inputImage = sitk.ReadImage(str(inputImage))
    _end_timer(start, message="Reading in the image")
    _print_info(inputImage)
    print("\n")
    return inputImage

def write_image(inputImage, outName, outDir="", fileFormat="mhd"):
    """Write SimpleITK image to disk in specified format.

    Supports medical imaging formats including mha, mhd, nii, dcm, tif, vtk, and others.
    Prints timing and file information during writing.

    Args:
        inputImage (sitk.Image): SimpleITK formatted image volume
        outName (str): Output filename without extension
        outDir (str): Output directory (defaults to current directory if empty)
        fileFormat (str): File format extension (defaults to "mhd")

    Returns:
        None: Writes image file to disk

    """
    start = timer()
    outName = str(outName)
    outDir == _get_outDir(outDir)
    fileFormat = str(fileFormat)

    fileFormat = fileFormat.replace(".", "")
    outputImage = Path(outDir).joinpath(str(outName) + "." + str(fileFormat))

    _print_info(inputImage)
    print(f"Writing {outName} to {outDir} as {fileFormat}.")
    sitk.WriteImage(inputImage, str(outputImage))
    _end_timer(start, message="Writing the image")

def feed_slice(inputImage, slice_num, direction="Z"):
    """Extract a single slice from a SimpleITK volume.

    Args:
        inputImage (sitk.Image): SimpleITK formatted image volume
        slice_num (int): Slice number to extract
        direction (str): Slice direction - "Z", "Y", or "X" (default: "Z")

    Returns:
        sitk.Image: 2D image slice in memory

    """
    direction = str(direction).lower()

    if direction == "z":
        image_slice = inputImage[:, :, slice_num]
    elif direction == "y":
        image_slice = inputImage[:, slice_num,:]
    else:
        image_slice = inputImage[slice_num, :, :]
    return image_slice

def three_class_segmentation(inDir, outDir, outType, network=""):
    """Segment a directory of 2D images using PyTorch model.

    Processes all images in input directory and generates 3-class segmentation
    with gray values representing air, dirt/tissue, and bone classes.

    Args:
        inDir (str): Input directory containing images to segment
        outDir (str): Output directory for segmented images (created if needed)
        outType (str): Output file format ("tif", "png", "jpg", "bmp")
        network: PyTorch neural network model for segmentation

    Returns:
        None: Saves segmented images to output directory

    """
    # Check to make sure the output folder exists, and if it doesn't make it
    data_folder = inDir
    save_folder = outDir

    net = network

    # The file types that can be output along with the corresponding dictionary
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # Get a list of files from the input folder using a list comprehension approach, then sort them numerically.
    image_names = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]
    image_names.sort()

    # Set up the count so there's something to watch as it processes
    seg_count = len(image_names)
    seg_now = seg_count
    print(f"Processing images in {data_folder}")

    # Loop through the images in the folder and use the image name for the output name
    for i in tqdm(range(len(image_names), unit=" slices", desc=" Segmenting...")):
        image_name = image_names[i]

        #Read the image in with pillow and set it as a numpy array for pytorch
        image = _setup_image(data_folder=data_folder, image_name=image_name)

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
        _save_predictors(pred=pred, save_folder=outDir, image_name=image_name, file_type=outType)

        #Write out the information for the command line.
        seg_now -= 1
        per_complete = abs((1 - (seg_now / seg_count)) * 100)
        print(f'{seg_now} of {seg_count} remaining, {per_complete:3.2f}% complete...\r', end="")

    print('\n\nSegmentations are done!\n\n')

def three_class_segmentation_volume(inputImage, direction="z", network=""):
    """Segment a 3D volume slice-by-slice using PyTorch model.

    Processes volume along specified direction and generates 3-class segmentation
    with gray values representing air, dirt/tissue, and bone classes.

    Args:
        inputImage (sitk.Image): SimpleITK formatted 3D image volume
        direction (str): Slicing direction - "z", "y", or "x" (default: "z")
        network: PyTorch neural network model for segmentation

    Returns:
        sitk.Image: Segmented 3D volume with 3-class labels

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
    print(f"Processing {seg_count} slices...")

    # Create an empty volume to stuff the results into. A numpy approach was tested but proved to be slower
    vol_image = sitk.Image(inputImage.GetSize(), sitk.sitkUInt8)

    # Loop through the images in the folder and use the image name for the output name
    for i in tqdm(range(seg_count), unit=" slices", desc=f" Segmenting {direction}"):
        image = feed_slice(inputImage, slice_num=i, direction=str(direction))

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


    #vol_image = empty_slice[1:]
    print('\n\nSegmentations are done!\n\n')
    _end_timer(start_timer=start, message="Segmentations")
    return vol_image

def rescale_8(inputImage):
    """Rescale SimpleITK image to 8-bit unsigned integer format.

    Args:
        inputImage (sitk.Image): SimpleITK formatted image volume

    Returns:
        sitk.Image: Unsigned 8-bit image with values scaled to 0-255 range

    """
    imageType = inputImage.GetPixelID()

    #Check to see if it is already unisgned 8 bit.
    if imageType == 1:
        print("Image is already unsigned 8...")
        scaled_8 = inputImage

    #If it isn't, go ahead and rescale.
    else:
        print("Rescaling to unsigned 8...")
        start = timer()
        scaled_8 = sitk.Cast(sitk.RescaleIntensity(inputImage), sitk.sitkUInt8)
        _print_info(scaled_8)
        _end_timer(start, message="Rescaling to unsigned 8")
    return scaled_8

def rescale_16(inputImage):
    """Rescale SimpleITK image to 16-bit unsigned integer format.

    Args:
        inputImage (sitk.Image): SimpleITK formatted image volume

    Returns:
        sitk.Image: Unsigned 16-bit image with values scaled to 0-65535 range

    """
    imageType = inputImage.GetPixelID()
    if imageType == 3:
        print("Image is already unsigned 16...")
        scaled_16 = inputImage
    else:
        # Read in the other image and recast to float 32
        print("Rescaling to unsigned 16...")
        start = timer()
        scaled_16 = sitk.Cast(sitk.RescaleIntensity(inputImage), sitk.sitkUInt16)
        _print_info(scaled_16)
        _end_timer(start, message="Rescaling to unsigned 16")
    return scaled_16

def rescale_32(inputImage):
    """Rescale SimpleITK image to 32-bit float format.

    Args:
        inputImage (sitk.Image): SimpleITK formatted image volume

    Returns:
        sitk.Image: 32-bit float image with full precision values

    """
    imageType = inputImage.GetPixelID()
    if imageType == 8:
        print("Image is already float 32...")
        scaled_32 = inputImage
    else:
        # Read in the other image and recast to float 32
        print('Rescaling to float 32...')
        start = timer()
        scaled_32 = sitk.Cast(sitk.RescaleIntensity(inputImage), sitk.sitkFloat32)
        _print_info(scaled_32)
        _end_timer(start, message="Rescaling to 32-bit float")
    return scaled_32

def combine_images(inputImage1, inputImage2):
    """Combine two SimpleITK images using addition.

    Args:
        inputImage1 (sitk.Image): First SimpleITK image
        inputImage2 (sitk.Image): Second SimpleITK image

    Returns:
        sitk.Image: Combined image result of addition

    """
    start = timer()

    # Add the two images together
    print("Combining...")
    combined = sitk.Add(inputImage1, inputImage2)
    _end_timer(start, message="Combing the two images")
    return combined

def three_class_seg_xyz(inputImage, network=""):
    """Segment 3D volume from all three orthogonal directions and combine results.

    Performs segmentation along X, Y, and Z axes then combines the results
    for improved segmentation accuracy through multi-directional consensus.

    Args:
        inputImage (sitk.Image): SimpleITK formatted 3D image volume
        network: PyTorch neural network model for segmentation

    Returns:
        sitk.Image: Combined 3-class segmentation result

    """
    # Segment the volume from all three directions
    seg_z = three_class_segmentation_volume(inputImage=inputImage, direction="z", network=network)
    seg_y = three_class_segmentation_volume(inputImage=inputImage, direction="y", network=network)
    seg_x = three_class_segmentation_volume(inputImage=inputImage, direction="x", network=network)

    #Rescale them to prevent overflow when we combine
    seg_z = rescale_16(seg_z)
    seg_y = rescale_16(seg_y)
    seg_z = combine_images(seg_z, seg_y)

    # Free up memory
    seg_y = 0
    seg_x = rescale_16(seg_x)

    seg_z = combine_images(seg_z, seg_x)

    seg_x = 0
    #Get the final product
    seg = rescale_8(seg_z)
    return seg

def _set_filter_events(sitkfilter):
    """Configure filter events for progress reporting.

    Args:
        sitkfilter: SimpleITK filter object

    Returns:
        tuple: (configured_filter, filter_events) with progress reporting enabled

    """
    filter_events = ReportPosition(sitkfilter)
    sitkfilter.AddCommand(sitk.sitkStartEvent, filter_events)
    sitkfilter.AddCommand(sitk.sitkProgressEvent, filter_events)
    sitkfilter.AddCommand(sitk.sitkProgressEvent, lambda: sys.stdout.flush())
    return sitkfilter, filter_events


def thresh_simple(inputImage, background=0, foreground=1, outside=0, threads="threads"):
    """Apply simple threshold filter to image.

    Args:
        inputImage (sitk.Image): SimpleITK image to threshold
        background (int): Lower threshold value (default: 0)
        foreground (int): Upper threshold value (default: 1)
        outside (int): Value for pixels outside threshold range (default: 0)
        threads (str or int): Thread count for processing (default: "threads")

    Returns:
        sitk.Image: Thresholded image

    """
    start = timer()
    thresh = sitk.ThresholdImageFilter()
    thresh.SetNumberOfThreads(_get_threads(threads))
    thresh.SetLower(background)
    thresh.SetUpper(foreground)
    thresh.SetOutsideValue(outside)
    thresh, _filter_events = _set_filter_events(thresh)
    threshold = thresh.Execute(inputImage)
    print("\n")
    _end_timer(start, message="Simple threshold")
    return threshold

def subtract_images(inputImage1, inputImage2):
    """Subtract two SimpleITK images.

    Args:
        inputImage1 (sitk.Image): First SimpleITK image (minuend)
        inputImage2 (sitk.Image): Second SimpleITK image (subtrahend)

    Returns:
        sitk.Image: Result of image subtraction

    """
    start = timer()

    # Subtract the two images together
    print("Subtracting...")
    subtracted = sitk.Subtract(inputImage1, inputImage2)
    _end_timer(start, message="Subtracting the two images")
    return subtracted

def read_stack(inputStack):
    """Read a series of images into a SimpleITK volume.

    Args:
        inputStack (list): List of image file paths to combine into volume

    Returns:
        sitk.Image: 3D SimpleITK volume assembled from image stack

    """
    # Read in the other image and recast to float 32
    start = timer()
    print("Reading in files...")
    inputStack.sort()
    inputStack = sitk.ReadImage(inputStack)
    _end_timer(start, message="Reading in the stack")
    _print_info(inputStack)
    print("\n")
    return inputStack

def read_dicom(inputStack):
    """Read DICOM image series while preserving metadata.

    Specialized reader that maintains DICOM metadata tags for medical imaging
    workflows. Tag 0020|000e may be modified with timestamp information.

    Args:
        inputStack (list): List of DICOM file paths to read

    Returns:
        tuple: (sitk_image, series_tag_values) containing the volume and metadata

    """
    start = timer()
    print(f"Reading in {len(inputStack)} DICOM images...")

    inputStack.sort()
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(inputStack)
    series_reader.MetaDataDictionaryArrayUpdateOn()
    series_reader.LoadPrivateTagsOn()
    sitk_image = series_reader.Execute()

    _print_info(sitk_image)

    #Grab the metadata, Name, ID, DOB, etc.
    direction = sitk_image.GetDirection()
    tags_to_copy = ["0010|0010", "0010|0020", "0010|0030", "0020|000D", "0020|0010",
                    "0008|0020", "0008|0030", "0008|0050", "0008|0060"]
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
    """Write SimpleITK image as DICOM series with preserved metadata.

    Args:
        inputImage (sitk.Image): SimpleITK formatted image volume
        metadata (list): Series metadata tags to preserve
        outName (str): Base output filename (without .dcm extension)
        outDir (str): Output directory (defaults to current directory if empty)

    Returns:
        None: Writes DICOM series to disk

    """
    #Modified from: https://simpleitk.readthedocs.io/en/master/link_DicomSeriesReadModifyWrite_docs.html

    start = timer()
    series_tag_values = metadata
    outDir = Path.cwd() if outDir == "" else Path(outDir)

    #Make is so the file name generator deal with these parts of the name
    if outName[-4] == ".dcm":
        outName = outName[:-4]

    if outName[-1] == "_":
        outName = outName[:-1]

    outName = Path(outDir).joinpath(outName)
    slice_num = inputImage.GetDepth()

    # Use the study/series/frame of reference information given in the meta-data
    # dictionary and not the automatically generated information from the file IO
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    digits_offset = len(str(slice_num))

    for i in tqdm(range(slice_num), unit=" slices", desc=f" Writing out {slice_num} DICOM slices to {outDir}..."):
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


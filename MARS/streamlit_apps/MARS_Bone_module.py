import os
import pathlib
import subprocess
import sys

import SimpleITK as sitk
import streamlit as st

script_dir = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(str(script_dir.parent.parent))
from MARS.morphology.segmentation.pytorch_segmentation.execute_3_class_seg import (
    read_image,
    rescale_8,
    write_image,
)


def file_selector(folder_path='.', extension="", selectbox_text=""):
    if folder_path == '.' or '':
        folder_path = pathlib.Path.cwd()
    filenames = os.listdir(folder_path)
    filenames.sort(reverse=True)
    if extension != "":
        filenames = [num for num in filenames if extension in num]
    selected_filename = st.selectbox(f'{selectbox_text}', filenames)
    try:
        return os.path.join(folder_path, selected_filename)
    except:
        pass


#Define where ImageJ is installed. The .as_posix() makes certain they have forward slashes "/"

ImageJDIR = st.sidebar.text_input("ImageJ location", pathlib.Path(r"D:\Desktop\Fiji.app\ImageJ-win64.exe").as_posix())

#Give the input analysis script
scriptFile = st.sidebar.text_input("Script location", pathlib.Path(r"Z:\RyanLab\Projects\NStephens\git_repo\MARS\analysis\ImageJ_analysis.py").as_posix())

#Give the directory where the RDN segmented image is
in_directory = st.sidebar.text_input("Segmented volume location", pathlib.Path(r"D:\Desktop\From_Lily").as_posix())
directory = file_selector(in_directory, extension="mhd", selectbox_text="Select volume")



#Change to that directory because it makes life easier
os.chdir(in_directory)

#Give the input file name
#file = "NF_27_819941_Humerus_Prox_R_UnSeg_reoriented_sphereVOI_RDN_seg.mhd"
file = pathlib.Path(directory).parts[-1]

#Define the output namer
seg_name = file.replace("_RDN_seg.mhd", "_seg")

#Since we used the bounds of the VOI and shrank it by 15% we do the same here to make the composite image
shrink = 0.15

#Get the full directory
input_file = pathlib.Path(in_directory).joinpath(file)

if st.button("Analyze!"):
    #Read in the image
    sitk_image = read_image(input_file)

    #Not used right now
    #tempfile.gettempdir()

    # Get the bounds from the xyz limits and copy over the relevant metadata
    bounds = sitk_image.GetSize()
    cube_origin = sitk_image.GetOrigin()
    cube_resolution = sitk_image.GetSpacing()

    # Threshold the image so the most "certain" values from the RDN are taken as bone
    sitk_image = thresh_simple(inputImage=sitk_image, background=210, foreground=255, outside=0)

    # Rescale them so everything is up to 255 and BoneJ recognizes it is a binary image
    sitk_image = rescale_intensity(inputImage=sitk_image, old_min=0, old_max=210, new_min=0, new_max=255)

    # Write that image out at the segmentation
    write_image(sitk_image, outName=seg_name, outDir=pathlib.Path.cwd(), fileFormat='mhd')

    # Get the sphere from the bounds of the ROI, then shrink it by how much we had to shrink it by
    sphere = get_sphere(bounds=bounds, r="", center="", shrink=float(shrink))

    # Take the Image and place it into a numpy array, copy the metadata, then rescale it to 8bit unsigned
    sphere = sitk.GetImageFromArray(sphere)
    sphere.CopyInformation(sitk_image)
    sphere = rescale_8(sphere)

    # Mask the ROI and invert it So ImageJ can calculate the trabecular spacing withput keeling over
    masked_ROI = ROI_mask(sitk_image, sphere, background=255, foreground=0, threads="threads")
    black_inside = sitk.InvertIntensity(masked_ROI, maximum=255)

    # Rescale and then combine the sphere and ROI to make a 3 level image.
    sitk_image = rescale_intensity(inputImage=sitk_image, old_min=0, old_max=1, new_min=0, new_max=1)
    sphere = rescale_intensity(inputImage=sphere, old_min=0, old_max=1, new_min=0, new_max=1)

    #This will add together whatever images you place within the Add
    composite_image = sitk.NaryAdd(sphere, sitk_image)

    #Calculate the BVTV using the composite image so we can ignore the "outside" values.
    BVTV = calcualte_BVTV_from_composite(inputImage=composite_image)

    #Write these out as generic composite iamge and spcaing voi for imagej
    write_image(inputImage=composite_image, outName="Composite_Image", outDir=pathlib.Path.cwd(), fileFormat='mhd')
    write_image(inputImage=black_inside, outName="Spacing_VOI", outDir=pathlib.Path.cwd(), fileFormat='mhd')

    #Build the command to pass to the imageJ javascript input

    #Directory and file, just as above.
    command1 = "DIRECTORY="
    command2 = ",FILE="
    DIRECTORY = f"{directory!s}/"
    FILE = f"'{seg_name}.mhd'"

    #These get joined together with doubel quotes " to deal with any spaces people may place into their file strcuture....
    command = f'"{command1}' + f"'{DIRECTORY}'" + f'{command2}' + f'{FILE}' + '"'
    command = str(command)

    #Prints out the command so you can make certain it looks fine on the console.
    setup = f"{ImageJDIR!s} --ij2 --console --run {scriptFile!s} {command!s}"
    print(setup)

    # Use subprocess to send it to an external terminal and then print the output (e.g. if there are java errors or imgaej
    # can't read it
    task = subprocess.run(str(setup), check=False, shell=True)
    print(task)

    # Replace the BoneJ results with those from the composite image with the correct background values,
    # and then multiple by the input resoultion. ImageJ has terrible support for this sort of metadata.
    image_res = cube_resolution[0]
    results_name = f"{seg_name}_BoneJ_results.csv"
    df = pd.read_csv(f"{results_name}")
    df["Bone_volume"] = BVTV[1]
    df["Total_volume"] = BVTV[0]
    df["Volume_fraction"] = BVTV[2]
    df["TbTh"] = df["TbTh"] * image_res
    df["TbTh_std"] = df["TbTh"] * image_res
    df["TbTh_max"] = df["TbTh_max"] * image_res
    df["Tb_Surface_area"] = df["Tb_Surface_area"] * image_res
    df["TbSp"] = df["TbSp"] * image_res
    df["TbSp_std"] = df["TbSp"] * image_res
    df["TbSp_max"] = df["TbSp_max"] * image_res
    df["Sp_Surface_area"] = df["Sp_Surface_area"] * image_res
    df.to_csv(results_name)



#This reproduces this example
#D:/Desktop/Fiji.app/ImageJ-win64.exe --ij2 --console --run D:/Desktop/Apply_Threshold.py "DIRECTORY='D:/Desktop/Lily/imagej/',FILE='DM_254_534_Humerus_Overview_resampled.mhd'"

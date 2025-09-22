import glob
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
import streamlit as st
import tetgen
from pymeshfix import _meshfix
from pymeshfix import meshfix as mf
from streamlit.hashing import _CodeHasher
from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server

script_dir = Path(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(str(script_dir.parent.parent))
from MARS.registration.Meshlab_tools import *

from MARS.streamlit_apps.streamlit_utils import _get_user, _load_MARS_logo, file_selector

supported_mesh = ["off", "ply", "stl", "obj", "wrl"]
supported_volume_mesh = ["vtk", "inp", "msh"]


# Get the small logo for the tab
script_dir = Path(os.path.dirname(os.path.realpath(__file__)))
tab_logo = Image.open(str(script_dir.joinpath('mars_square.jpg')))

# This is a beta feature to control the default elements of the app
st.beta_set_page_config(page_title="Meshlab batch",
                        page_icon=tab_logo,
                        layout='wide',
                        initial_sidebar_state='auto')

# Check the version
streamlit_version = st.__version__.split(".")
streamlit_major = int(streamlit_version[0])
streamlit_minor = int(streamlit_version[1])
streamlit_patch = int(streamlit_version[2])
if streamlit_minor < 65:
    st.error(f"The version of streamlit needs to be 0.65.2+ and this is {st.__version__}")
    st.error("Please install the newest version using:")
    with st.echo():
        'pip install streamlit --upgrade'


#Defines the body of the gui
def main():
    """Streamlit GUI to perform segmentation of grey value images using a trained RDN 3-class segmentation network
    implemented in pytorch. Reading and writing of values is predominantly handled with SimpleITK.
    """
    current_user = _get_user()

    #Load in the logo
    logo = _load_MARS_logo()
    st.sidebar.image(logo, width=185, caption="Meshlab Batch         ", output_format="PNG")

    #Defines the pages on the side bar and what they show at the top
    pages = {
        "Mesh batch": meshlab_settings,
    }

    #Saves settings between pages.
    state = _get_state()

    st.sidebar.radio("Navigation:", tuple(pages.keys()))

    # Button to save state values
    if st.sidebar.button("Save settings"):
        save_state_values(state=state, user=current_user, app_name="meshlab")

    if st.sidebar.button("Load previous settings"):
        load_state_values(state=state, user=current_user, app_name="meshlab")

    if st.sidebar.button("Clear settings"):
        state.clear()
        save_state_values(state=state, user=current_user, app_name="meshlab")

    if st.sidebar.button("Open meshlab"):
        subprocess.run(f'{state.meshlab_path!s}', check=False, shell=True)

    if st.sidebar.button("Open mesh input folder"):
        subprocess.run(f'start {state.mesh_location!s}', check=False, shell=True)

    # Button to clear state values
    if st.sidebar.button("Open mesh output folder"):
        subprocess.run(f'start {state.mesh_output_location!s}', check=False, shell=True)

    ###
    st.sidebar.subheader("Common operations")

    # Help file
    if st.sidebar.button("Help!"):
        st.sidebar.info("Note: Settings may be saved between runs by clicking 'Save settings', and restored by hitting"
                        "'Load previous settings'.")
        st.sidebar.markdown("**Steps:**")
        st.sidebar.markdown("1) Provide where meshlab is located")
        st.sidebar.markdown("2) Paste the location of the mesh files to process, along with the input file format.")
        st.sidebar.markdown("*If you want to subset the meshes by a unique string you may*")
        st.sidebar.markdown("2) Paste the output location for the meshes, along with the desired output file type")
        st.sidebar.markdown("3) If you wnat to apply a meshlab filter script you can paste the location of the script "
                            "and then select the appropriate .mlx file.")
        st.sidebar.markdown("4) Generate the mesh file list, and view it prior to selecting the desired operation")
        save_state_values(state=state, user=current_user)

    ## The select boxes and inputs are here
    state.meshlab_path = Path(st.text_input("Meshlab location", r"C:\Program Files\VCG\MeshLab\meshlab.exe"))
    state.meshlab_server_path = Path(str(state.meshlab_path)).parent.joinpath("meshlabserver.exe")


    state.mesh_location = Path(st.text_input("Mesh location"), key=999073)
    state.mesh_input_type = st.selectbox("Mesh input file format", supported_mesh, key=9990110)


    if st.checkbox("Subset with glob?"):
        st.info("Glob uses \\* for wildcards. Use a string that will match only the files you want to process"
                "For example: StJ_*_Femur_* will match any file that starts with StJ_ and has _Femur_ in the middle")
        glob_string = str(st.text_input("Glob string match for glob", "*"))
        state.glob_string = escape_markdown(glob_string)
        save_state_values(user=current_user, state=state)

    else:
        state.glob_string = ""
        state.glob_string = escape_markdown(state.glob_string)


    state.mesh_output_location = Path(st.text_input("Mesh output location"), key=9990125)
    save_state_values(user=current_user, state=state)
    state.mesh_output_type = st.selectbox("Mesh output file format", supported_mesh, key=9990126)
    save_state_values(user=current_user, state=state)


    if st.checkbox("Apply meshlab script?"):
        st.info("You can create meshlab script by running a single mesh through the filters you'd like and saving it. " \
        "This can then be applied to a group of meshes by providing the .mlx script here")
        try:
            state.filter_script_location = Path(st.text_input("Meshlab script location", state.filter_script_location))
        except FileNotFoundError:
            st.error("Couldn't find the directory, please check that everything is spelled correctly. "
                     "Note: This is case sensistive.")
        if str(state.filter_script_location) == "." or str(state.filter_script_location) == "None":
            st.info("Please paste in the filter script location")
        else:
            state.filter_script = Path(file_selector(folder_path=str(state.filter_script_location),
                                                             extension="mlx",
                                                             unique_key=9990138))
            save_state_values(user=current_user, state=state)

    if st.button("Get mesh list!"):
        load_state_values(state=state, user=current_user)
        state.mesh_list = []
        glob_string = state.glob_string.replace("\\", "")
        glob_string = f"{glob_string!s}*.{state.mesh_input_type!s}"
        glob_string = glob_string.replace("**", "*")
        st.write(state.mesh_location)
        glob_path = f"{Path(state.mesh_location).joinpath(glob_string)!s}"
        st.write(f"Globbing {glob_path}...")
        mesh_list = glob.glob(glob_path)
        st.write(f"Found {len(mesh_list)} mesh files!")
        mesh_list.sort()
        state.mesh_list = mesh_list
        save_state_values(user=current_user, state=state)

    if st.checkbox("View mesh list"):
        load_state_values(state=state, user=current_user)
        st.write(state.mesh_list)

    function_list = ["Convert mesh",
                     "Apply PyMeshfix",
                     "Apply Meshlab Script",
                     "Chain Meshlab Scripts",
                     "Remove internal geometry",
                     "Make solid"]
    function_radio_list = function_list
    mesh_settings_activity = st.experimental_get_query_params()
    # Query parameters are returned as a list to support multiselect.
    # Get the first item in the list if the query parameter exists.
    mesh_default = int(
        mesh_settings_activity["activity"][0]) if "activity" in mesh_settings_activity else 0
    mesh_settings_activity = st.radio(
        "Select meshlab function",
        function_radio_list,
        index=mesh_default)

    st.write("---")
    if mesh_settings_activity == "Convert mesh":
        load_state_values(state=state, user=current_user)
        mesh_list = state.mesh_list
        out_type = str(state.mesh_output_type)
        if st.button("Convert", key=9990175):
            num_meshes = len(mesh_list)
            with st.spinner(f"Converting meshes {num_meshes} to {out_type} file format..."):
                progress_bar = st.progress(0)
                current_total = 0
                for mesh_file in mesh_list:
                    input_directory = Path(mesh_file).as_posix().rsplit("/", 1)[0]
                    mesh_file = str(Path(mesh_file).parts[-1])
                    st.write(f"Processing {mesh_file} in {input_directory}")
                    convert_mesh(in_dir=input_directory,
                                 in_file=mesh_file,
                                 out_type=out_type,
                                 output_directory=str(state.mesh_output_location),
                                 meshlabserverpath=str(state.meshlab_server_path)
                                 )
                    st.write(f"{mesh_file} writen to {state.mesh_output_location!s}")

                    current_total += 1
                    iteration = np.floor(100 * ((current_total) / num_meshes))
                    progress_bar.progress(int(iteration))

    if mesh_settings_activity == "Apply Meshlab Script":
        if st.checkbox("Help", key=9990197):
            st.info("Meshlab scripts may be saved in mlx format and loaded to be used here.")
            st.warning("Due to the verbosity of Meshlab server, please be aware that the standard messages and errors"
                       " will be written to the command prompt/shell you launched this app from. If something doesn't "
                       " look right, check this output for answers to your confusion.")

        load_state_values(state=state, user=current_user)
        mesh_list = state.mesh_list
        out_type = str(state.mesh_output_type)
        meshlab_script = str(state.filter_script)
        st.write(meshlab_script)

        if st.button("Apply script", key=9990203):
            num_meshes = len(mesh_list)
            with st.spinner(f"Applying {meshlab_script} to {num_meshes}..."):
                progress_bar = st.progress(0)
                current_total = 0
                for mesh_file in mesh_list:
                    input_directory = Path(mesh_file).as_posix().rsplit("/", 1)[0]
                    mesh_file = str(Path(mesh_file).parts[-1])

                    st.write(f"Processing {mesh_file} in {input_directory}....")

                    highres_to_lowres(in_dir=input_directory,
                                      in_file=mesh_file,
                                      out_type=out_type,
                                      output_directory=str(state.mesh_output_location),
                                      meshlabserverpath=str(state.meshlab_server_path),
                                      filter_script_path=str(meshlab_script),
                                      shell=True)

                    current_total += 1
                    iteration = np.floor(100 * ((current_total) / num_meshes))
                    progress_bar.progress(int(iteration))
                    st.write(f"{mesh_file} writen to {state.mesh_output_location!s}")

    if mesh_settings_activity == "Chain Meshlab Scripts":
        load_state_values(state=state, user=current_user)
        if st.checkbox("Help", key=9990229):
            st.info("Because Meshlab has some strange bugs when combining filters, it is sometimes necessary"
                    "to chain together smaller scripts. This allows you to select scripts in the order of execution")
            st.warning("Please note that the first script will write to your output folder, then the subsequent scripts"
                       "will use the output folder to read in the processed meshes.")

        mesh_list = state.mesh_list
        out_type = str(state.mesh_output_type)

        script_list = glob.glob(str(Path(state.filter_script_location).joinpath("*.mlx")))
        meshlab_script_list = st.multiselect("Select scripts in execution order", script_list)
        st.write(meshlab_script_list)
        state.meshlab_script_list = meshlab_script_list
        if len(meshlab_script_list) <= 1:
            st.error("Must select more than 1 meshlab script!")

        else:
            st.write(f"Will run {state.meshlab_script_list[0]} first.")
            if st.button("Apply scripts", key=9990236):
                meshlab_script_list = state.meshlab_script_list
                meshlab_script = meshlab_script_list[0]
                num_meshes = len(mesh_list)
                large_current_total = 0
                scripts_total = len(meshlab_script_list)
                with st.spinner(f"Applying {scripts_total} scripts to {num_meshes} meshes..."):
                    large_progress_bar = st.progress(0)
                    with st.spinner(f"Applying {meshlab_script} to {num_meshes}..."):
                        progress_bar = st.progress(0)
                        current_total = 0
                        for mesh_file in mesh_list:
                            input_directory = Path(mesh_file).as_posix().rsplit("/", 1)[0]
                            mesh_file = str(Path(mesh_file).parts[-1])

                            st.write(f"Processing {mesh_file} in {input_directory}....")

                            highres_to_lowres(in_dir=input_directory,
                                              in_file=mesh_file,
                                              out_type=out_type,
                                              output_directory=str(state.mesh_output_location),
                                              meshlabserverpath=str(state.meshlab_server_path),
                                              filter_script_path=str(meshlab_script),
                                              shell=True)

                            current_total += 1
                            iteration = np.floor(100 * ((current_total) / num_meshes))
                            progress_bar.progress(int(iteration))
                            st.write(f"{mesh_file} writen to {state.mesh_output_location!s}")

                        large_current_total += 1
                        large_iteration = np.floor(100 * ((large_current_total) / scripts_total))
                        large_progress_bar.progress(int(large_iteration))

                    #Should probably make this into a function for readability
                    for next_script in meshlab_script_list[1:]:
                        meshlab_script = next_script
                        with st.spinner(f"Applying {meshlab_script} to {num_meshes}..."):
                            progress_bar = st.progress(0)
                            current_total = 0
                            for mesh_file in mesh_list:
                                input_directory = str(state.mesh_output_location)
                                mesh_file = str(Path(mesh_file).parts[-1])
                                st.write(f"Processing {mesh_file} in {input_directory}....")
                                highres_to_lowres(in_dir=input_directory,
                                                  in_file=mesh_file,
                                                  out_type=out_type,
                                                  output_directory=str(state.mesh_output_location),
                                                  meshlabserverpath=str(state.meshlab_server_path),
                                                  filter_script_path=str(meshlab_script),
                                                  shell=True)

                                current_total += 1
                                iteration = np.floor(100 * ((current_total) / num_meshes))
                                progress_bar.progress(int(iteration))
                                st.write(f"{mesh_file} writen to {state.mesh_output_location!s}")
                            large_current_total += 1
                            large_iteration = np.floor(100 * ((large_current_total) / scripts_total))
                            large_progress_bar.progress(int(large_iteration))

    if mesh_settings_activity == "Apply PyMeshfix":
        load_state_values(state=state, user=current_user)
        mesh_list = state.mesh_list
        out_type = str(state.mesh_output_type)
        if st.button("Fix meshes", key=9990334):
            num_meshes = len(mesh_list)
            if num_meshes == 1:
                spinner_message = f"Apply PyMeshfix to {num_meshes} mesh..."
            else:
                spinner_message = f"Apply PyMeshfix to {num_meshes} meshes..."
            with st.spinner(f"{spinner_message}"):
                progress_bar = st.progress(0)
                current_total = 0
                for mesh_file in mesh_list:
                    input_directory = Path(mesh_file).as_posix().rsplit("/", 1)[0]
                    mesh_file = str(Path(mesh_file).parts[-1])
                    st.write(f"Processing {mesh_file} in {input_directory}")
                    pymeshfix(in_dir=input_directory,
                              in_file=mesh_file,
                              out_type=out_type,
                              output_directory=str(state.mesh_output_location),
                              write=True)
                    st.write(f"{mesh_file} writen to {state.mesh_output_location!s}")

                    current_total += 1
                    iteration = np.floor(100 * ((current_total) / num_meshes))
                    progress_bar.progress(int(iteration))
    if mesh_settings_activity == "Remove internal geometry":
        load_state_values(state=state, user=current_user)
        mesh_list = state.mesh_list
        out_type = str(state.mesh_output_type)
        st.warning("This is something of a brute force method for removing internal geometry. "
                   " This page may tell you that the connection has timed out, but that simply means that the "
                   " functions in the background are still working.")
        st.warning("You can verify this by looking at your task manager or the output from the console."
                   " It can take a very long time (~60 min) with complex meshes."
                   " It's still better than doing it manually... :smile:")
        if st.checkbox("Use TetWild"):
            state.tet_wild = True
        else:
            state.tet_wild = False
        if st.button("Isolate surface", key=9990356):
            num_meshes = len(mesh_list)
            with st.spinner(f"Removing the internal geometry of {num_meshes} meshes..."):
                progress_bar = st.progress(0)
                current_total = 0
                for mesh_file in mesh_list:
                    input_directory = Path(mesh_file).as_posix().rsplit("/", 1)[0]
                    mesh_file = str(Path(mesh_file).parts[-1])
                    st.write(f"Processing {mesh_file} in {input_directory}")
                    if state.tet_wild:
                        #This is all pretty ad-hoc and will need to be cleaned up.
                        temp_dir = Path(tempfile.gettempdir())
                        cleaned_mesh = pymeshfix(in_dir=str(input_directory),
                                                 in_file=mesh_file,
                                                 out_type="ply",
                                                 output_directory=str(temp_dir),
                                                 write=False)
                        print(cleaned_mesh)
                        temp_save = temp_dir.joinpath("temp_mesh.ply")
                        cleaned_mesh.save(str(temp_save), binary=True)
                        TetWild_tetrahedralize(input_path=str(temp_dir),
                                               in_file="temp_mesh.ply",
                                               output_path=str(temp_dir),
                                               out_name="temp_mesh",
                                               edge_length="",
                                               target_verts="",
                                               laplacian=True)
                        temp_mesh = pv.read(str(temp_dir.joinpath("temp_mesh.vtk")))
                        temp_mesh = temp_mesh.extract_geometry()
                        largest_surface = temp_mesh.connectivity(largest=True)
                        output = Path(str(state.mesh_output_location)).joinpath(f"{mesh_file}.{out_type}")
                        if out_type in ['ply', 'vtp', 'stl', 'vtk']:
                            largest_surface.save(str(output))
                        else:
                            pv.save_meshio(filename=str(output), mesh=largest_surface)
                    else:
                        isolate_external_surface(in_dir=input_directory,
                                                 in_file=mesh_file,
                                                 out_type=out_type,
                                                 output_directory=str(state.mesh_output_location),
                                                 write_mesh=True)
                    st.write(f"{mesh_file} writen to {state.mesh_output_location!s}")

                    current_total += 1
                    iteration = np.floor(100 * ((current_total) / num_meshes))
                    progress_bar.progress(int(iteration))

    if mesh_settings_activity == "Make solid":
        load_state_values(state=state, user=current_user)
        mesh_list = state.mesh_list
        if state.mesh_input_type not in ["off", "obj", "stl", "ply"]:
            st.warning("Tetwild only supports off, obj, stl, and ply file formats. Input meshes will be "
                       "converted and written to a temporary folder.")
        st.warning("TetWild is very robust, but this comes at the cost of time. It is recommended that you run "
                   "pymeshfix prior to this step.")
        st.info("All solid mesh files will be written as vtk, which has good support in many open source libraries.")
        if st.checkbox("Citation", key=9990397):
            st.info("Please make sure to cite Hu, Y. et al., 2018 DOI: 10.1145/3197517.3201353.")


        if st.button("Make solid!", key=9990386):
            num_meshes = len(mesh_list)
            with st.spinner(f"Solidizinginginging {num_meshes} meshes..."):
                progress_bar = st.progress(0)
                current_total = 0
                for mesh_file in mesh_list:
                    input_directory = Path(mesh_file).as_posix().rsplit("/", 1)[0]
                    mesh_file = str(Path(mesh_file).parts[-1])
                    st.write(f"Processing {mesh_file} in {input_directory}")
                    TetWild_tetrahedralize(input_path=str(input_directory),
                                           in_file=mesh_file,
                                           output_path=str(state.mesh_output_location),
                                           out_name="",
                                           edge_length="",
                                           target_verts="",
                                           laplacian=True)

                    st.write(f"{mesh_file} writen to {state.mesh_output_location!s}")

                    current_total += 1
                    iteration = np.floor(100 * ((current_total) / num_meshes))
                    progress_bar.progress(int(iteration))

    mesh_state_values(state)



def meshlab_settings(state):
    """Empty page for now

    """
    _get_user()


def mesh_state_values(state):
    if state.mesh_location != ".":
        st.info(f"Mesh input path: {state.mesh_location}")
    else:
        st.error("Mesh input path not set!")

    if not state.mesh_list:
        st.error("There are no meshes in the mesh list!")
    else:
        st.info(f"{len(state.mesh_list)} meshes ready to process")

    if state.mesh_output_location != ".":
        st.info(f"Mesh output location: {state.mesh_output_location}")
    else:
        st.warning("Parameter file not set")

    if str(state.mesh_output_type) != ".":
        st.info(f"Mesh output type: {state.mesh_output_type}")
    else:
        st.warning("No output type set!")
    if str(state.filter_script) != ".":
        st.info(f"Loaded filter script: {state.filter_script}")
    else:
        st.warning("No filter script set")


###
#
#  Streamlit functions
#
###

def _get_citation(bibtext_key):
    #THis is simply a place holder for an idea.
    return bibtext_key

def direct_pymeshfix(in_dir, in_file, out_type, output_directory):
    """Cite: M. Attene. A lightweight approach to repairing digitized polygon meshes.
    The Visual Computer, 2010. (c) Springer. DOI: 10.1007/s00371-010-0416-3
    :param:
    :return:
    """
    in_file = str(in_file)
    in_dir = Path(in_dir)

    # Define the output file by replacing the last 3 characters with the new type
    out_file = str(in_file[:-3]) + str(out_type)

    # Use pathlib to join the input file and output file
    in_file = Path(in_dir).joinpath(in_file)
    output = Path(output_directory).joinpath(out_file)

    # Print the input name to the console
    print(f"\nInput {in_file}.\n")
    print(f"Output {output}.\n")
    _meshfix.clean_from_file(infile=str(in_file), outfile=str(output), verbose=True)
    print("\n\nDone:")

def TetWild_tetrahedralize(input_path, in_file, output_path, out_name="", edge_length="", target_verts="",
                         laplacian=False):
    """Tetrahedralize from a 2d .off, .obj, .stl, or .ply file. Output is in .msh/.mesh format. The .msh is then converted to an inp using gmsh.
    It is assumed that gmsh is installed and on the system path.
    See Hu Yixin, et al. 2018 (http://doi.acm.org/10.1145/3197517.3201353) and https://github.com/Yixin-Hu/TetWild for further information.
    :param input_path:
    :param in_file:
    :param output_path:
    :param tetwildpath:
    :param out_name:
    :param edge_length:
    :param target_verts:
    :param laplacian:
    :return:
    """
    script_dir = _get_script_dir()
    tetwildpath = Path(script_dir.parent.joinpath("TetWild"))

    # Reads in the input file name as a string.
    in_file = str(in_file)

    if out_name == "":
        out_file = str(in_file[:-3]) + "vtk"
    else:
        out_file = str(out_name) + ".vtk"
        out_file = out_file.replace("..", ".")

    # Removes the last 3 characters and appends the new output type
    msh_file = str(in_file[:-4]) + "_.msh"

    # Gives the long file path and name for reading in
    in_file = Path(input_path).joinpath(in_file)

    # in_file = Path(input_path).joinpath(in_file)
    print(f"Input: {in_file}")

    # output = Path(input_path).joinpath("lowres")
    output = Path(output_path).joinpath(out_file)
    print(f"Output: {output}")

    # This is where meshlabserver lives
    tetwildpath = Path(tetwildpath).joinpath("TetWild.exe")

    # So this is just putting it into a format that is readable by tetwild
    command = '"' + str(tetwildpath) + '"' + " " + str(in_file)
    if edge_length != "":
        edge_length = float(edge_length)
        command += " --ideal-edge-length " + str(edge_length)
    if target_verts != "":
        target_verts = int(target_verts)
        command += " --targeted-num-v " + str(target_verts)
    if laplacian:
        command += " --is-laplacian "

    # Tells us the command we are sending to the console
    print(f"\nGoing to execute:\n                            {command!s}")

    # Actually sends this to the console
    output = subprocess.call(command)

    # This SHOULD show us the error messages or whatever....
    last_line = output
    print(last_line)
    print("\n\nDone!\n")

    temp = Path(tempfile.gettempdir())
    msh_file = Path(temp).joinpath(msh_file)
    out_file = Path(output_path).joinpath(out_file)

    # Use meshio to read in the msh file and output the inp. Easier than hoping gmsh installed properly.
    new_mesh = pv.read(str(msh_file))
    new_mesh.save(out_file)


def pymeshfix(in_dir, in_file, out_type, output_directory, write=False):
    """Cite: M. Attene. A lightweight approach to repairing digitized polygon meshes.
    The Visual Computer, 2010. (c) Springer. DOI: 10.1007/s00371-010-0416-3
    :param:
    :return:
    """
    in_file = str(in_file)
    in_dir = Path(in_dir)

    # Define the output file by replacing the last 3 characters with the new type
    out_file = str(in_file[:-3]) + str(out_type)

    # Use pathlib to join the input file and output file
    in_file = Path(in_dir).joinpath(in_file)
    output = Path(output_directory).joinpath(out_file)

    rough_mesh = pv.read(str(in_file))
    st.text(f"Loaded mesh with {rough_mesh.n_cells} cells and { rough_mesh.n_points} vertices")
    if isinstance(rough_mesh, pv.core.pointset.UnstructuredGrid):
        st.write("Extracting geometry from unstructured grid")
        rough_mesh = rough_mesh.extract_geometry()

    st.write("Repairing mesh...")
    meshfix = mf.MeshFix(rough_mesh)

    meshfix.repair(verbose=True,
                   joincomp=True,
                   remove_smallest_components=True)
    new_mesh = meshfix.mesh
    st.write("Mesh repaired")
    if write:
        if out_type in ['ply', 'vtp', 'stl', 'vtk']:
            new_mesh.save(str(output))
        else:
            pv.save_meshio(filename=str(output), mesh=new_mesh)
    else:
        return new_mesh
    print("\n\nDone")


def isolate_external_surface(in_dir, in_file, out_type, output_directory, write_mesh=True):
    """Function that uses brute force to get the outside surface of a mesh by filling holes, tetrahedralizing, then
    extracting the resulting surface of the mesh. Depening on the complexity of the initial mesh, this may fail or take
    a long time.
    Cite: M. Attene. A lightweight approach to repairing digitized polygon meshes.
    The Visual Computer, 2010. (c) Springer. DOI: 10.1007/s00371-010-0416-3
    :param:
    :return:
    """
    in_file = str(in_file)
    in_dir = Path(in_dir)

    # Define the output file by replacing the last 3 characters with the new type
    out_file = str(in_file[:-3]) + str(out_type)

    # Use pathlib to join the input file and output file
    in_file = Path(in_dir).joinpath(in_file)
    output = Path(output_directory).joinpath(out_file)

    rough_mesh = pv.read(str(in_file))
    st.text(f"Loaded mesh with {rough_mesh.n_cells} cells and { rough_mesh.n_points} vertices")
    if isinstance(rough_mesh, pv.core.pointset.UnstructuredGrid):
        st.write("Extracting geometry from unstructured grid")
        rough_mesh = rough_mesh.extract_geometry()

    st.write("Repairing mesh...")
    meshfix = mf.MeshFix(rough_mesh)

    meshfix.repair(verbose=True,
                   joincomp=True,
                   remove_smallest_components=True)
    rough_mesh = meshfix.mesh
    st.write("Mesh repaired")
    st.write("Filling internal geometry")
    tet = tetgen.TetGen(rough_mesh)
    tet.make_manifold(verbose=True)
    tet.tetrahedralize(facet_overlap_ang_tol=0.00001,
                       nobisect=True,
                       quality=True,
                       minratio=2.1,
                       mindihedral=5,
                       steinerleft=-1,
                       verbose=1)
    filled_mesh = tet.grid
    new_mesh = filled_mesh.extract_geometry()
    print("\n\nDone")
    if not write_mesh:
        return new_mesh
    elif out_type in ['ply', 'vtp', 'stl', 'vtk']:
        new_mesh.save(str(output))
    else:
        pv.save_meshio(filename=str(output), mesh=rough_mesh)

def escape_markdown(text):
    MD_SPECIAL_CHARS = r"\`*_{}[]()#+-.!"
    for char in MD_SPECIAL_CHARS:
        text = text.replace(char, "\\" + char)
    return text

def _get_script_dir():
    script_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    return script_dir

def save_state_values(state, user, app_name="meshlab"):
    script_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    saved_dir = script_dir.joinpath("saved_states").joinpath(f"{user}_{app_name}_saved_state.json")
    session_dict = {"meshlab_path":  [str(state.meshlab_path)],
                    "meshlab_server_path": [str(state.meshlab_server_path)],
                    "mesh_location": [str(state.mesh_location)],
                    "mesh_input_type": [str(state.mesh_input_type)],
                    "glob_string": [str(state.glob_string)],
                    "mesh_output_location": [str(state.mesh_output_location)],
                    "mesh_output_type": [str(state.mesh_output_type)],
                    "filter_script_location": [str(state.filter_script_location)],
                    "filter_script": [str(state.filter_script)]
                    }
    session_dict = pd.DataFrame.from_dict(session_dict)
    session_dict.to_json(saved_dir)
    st.write(f"Saved settings for {user}!")


def load_state_values(state, user, app_name="meshlab", verbose=False):
    script_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    saved_dir = script_dir.joinpath("saved_states").joinpath(f"{user}_{app_name}_saved_state.json")
    restored_session = pd.read_json(str(saved_dir))
    state.meshlab_path = Path(restored_session['meshlab_path'][0])
    state.meshlab_server_path = Path(restored_session['meshlab_server_path'][0])
    state.mesh_location = Path(restored_session['mesh_location'][0])
    state.mesh_input_type = str(restored_session['mesh_input_type'][0])
    state.glob_string = str(restored_session['glob_string'][0])
    state.mesh_output_location = Path(restored_session['mesh_output_location'][0])
    state.mesh_output_type = str(restored_session['mesh_output_type'][0])
    state.filter_script_location = Path(restored_session['filter_script_location'][0])
    state.filter_script = Path(restored_session['filter_script'][0])
    if verbose:
        st.write("Restored previous settings")
    return state


###
#  Classes for hacking the streamlit internals
#
#  !!!!If you touch this stuff everything will break!!!!
#
#
# Modifed from https://gist.github.com/Ghasel/0aba4869ba6fdc8d49132e6974e2e662
####



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

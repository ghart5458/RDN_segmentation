"""This application experiments with the (grid) layout and some styling
Can we make a compact dashboard across several columns and with a dark theme?"""
#import gmsh
import os
import subprocess
from pathlib import Path
from timeit import default_timer as timer

import pandas as pd
import streamlit as st
import trimesh


def mesh_info(mesh):
    '''
    A function to return basic information about a loaded 2d mesh
    :param mesh: A 2d mesh loaded through trimesh.load
    '''
    mesh_2d = mesh
    triangles = len(mesh_2d.triangles)
    points = len(mesh_2d.vertices)
    edgemin = mesh_2d.edges_unique_length.min()
    edgemax = mesh_2d.edges_unique_length.max()
    edgemean = mesh_2d.edges_unique_length.mean()
    if mesh_2d.is_volume:
        st.write("Mesh can be represented as a 3d volume.")
    elif not mesh_2d.is_watertight:
        st.write("Mesh has holes.")
    else:
        st.write("Mesh doesn't have holes, please check for other issues in Meshlab.")
    st.write(f"Mesh has {triangles} faces and {points} vertices.")
    st.write(f"Mesh edge length: \n           mean {edgemean:06.4f}, max {edgemax:06.4f}, min {edgemin:06.4f}")

#Function to read in an inp and output a case
def inp_to_case(in_name, outname, out_dir):
    '''
    Function to read in an ascii inp file and output a point cloud, for cloud compare, and a case file,
    readable by paraview.

    :param in_name: Abaqus Ascii inp file name.
    :param outname: Output file name
    :return:
    '''

    start = timer()
    file_name = str(outname)
    outname = Path(out_dir).joinpath(outname)
    # Open the inp and find the diagnostic lines
    with open(in_name) as f:
        # Get the number of lines in the file
        size = len([0 for _ in f])
        st.write("Lines in inp file:", size)
    with open(in_name) as f:
        # Get the number of lines in the file
        start_xyz = "*NODE"
        start_elements = "******* E L E M E N T S *************"
        start_element_sets = "*ELSET"
        for num, line in enumerate(f, 1):
            if start_xyz in line:
                st.write('Node set starts at line:', num)
                st.write(line)
                xyz = num
            if start_elements in line:
                st.write('Elements start at line:', num)
                st.write(line)
                elements = num
            if start_element_sets in line:
                st.write('Element sets start at line:', num)
                st.write(line)
                sets = num
        f.close()

    # Use the line values to define the skip rows list for pandas

    # Start of file to nodes line
    range_1 = list(range(0, int(xyz)))

    # Start of element line to end of file.
    range_2 = list(range(int(elements - 1), int(size)))

    # Start of file until the end of the elements.
    range_3 = list(range(0, int(elements + 1)))

    # Start of the element set until the end of the file.
    range_4 = list(range(int(sets - 1), int(size)))

    # Create a list with the individual range lists
    xyz_range = range_1 + range_2
    element_range = range_3 + range_4
    list(range(0, int(sets)))

    # Read in individual dataframes for the portions. It isn't efficient, but it is readable.
    xyz_df = pd.read_csv(in_name, header=None, sep=",", skiprows=xyz_range)

    # Use ravel to stack the rows untop of one another
    stacked_xyz = pd.DataFrame(xyz_df.iloc[:, 1:4].values.ravel('F'))

    # Define the elements dataframe
    elements_df = pd.read_csv(in_name, header=None, sep=",", index_col=False, skiprows=element_range)


    # Define the sets dataframe, which isn't really used but who knows what will happen
    #try:
    #    set_df = pd.read_csv(in_name, header=None, sep=",", index_col=False, skiprows=set_range)
    #    set_df.to_csv(str(in_name) + "elementsets.csv", header=False, sep = " ")


    with open(str(outname) + ".case", 'w', newline="") as fout:
        # tells where to find the geometry (i.e. .geo file) and what format
        fout.write('FORMAT\ntype: ensight gold\n\nGEOMETRY\nmodel:                           ')
        # Scalars per node for 3d geometry
        fout.write(str(file_name) + ".geo\n")
        fout.close()
    with open(str(outname) + '.geo', 'w', newline="") as fout:
        # Case header defining the name of the scalar array
        Title = str(file_name) + ".inp"
        # fout to write the header
        fout.write("Title:  " + str(Title) + "\n")
        # fout to write the shape of the tetrahedral geometry (i.e. tetra4)
        fout.write('Description 2\nnode id given\nelement id given\npart\n         1\n')
        fout.write('Description PART\ncoordinates \n     ' + str(len(xyz_df)) + "\n")
        # use pandas in conjunction with fout to append the scalars
        xyz_df[0].to_csv(fout, header=False, index=False, sep=" ")
        # Write out the case format header with fout
        stacked_xyz.to_csv(fout, header=False, index=False, sep=" ")
        # Write out the elments
        fout.write("\ntetra4\n" + str(len(elements_df)) + "\n")
        elements_df[0].to_csv(fout, header=False, index=False, sep=" ")
        elements_df.iloc[:, 1:].to_csv(fout, header=False, index=False, sep=" ")
        fout.close()

    #Write out the point cloud
    xyz_df.iloc[:, 1:4].to_csv(str(outname) + "_pointcloud.csv", header=None, index=False)

    end = timer()
    end_time = (end - start)
    st.write("Coversion took", end_time, "\n")


#Function to read in an inp and output a case
def inp_to_case_2d(in_name, outname):
    '''
    Function to read in an ascii inp file and output a point cloud, for cloud compare, and a case file,
    readable by paraview.

    :param in_name: Abaqus Ascii inp file name.
    :param outname: Output file name
    :return:
    '''

    start = timer()
    # Open the inp and find the diagnostic lines
    with open(in_name) as f:
        # Get the number of lines in the file
        size = len([0 for _ in f])
        st.write("Lines in inp file:", size)
    with open(in_name) as f:
        # Get the number of lines in the file
        start_xyz = "*NODE"
        start_elements = "******* E L E M E N T S *************"
        start_element_sets = "*ELSET"
        for num, line in enumerate(f, 1):
            if start_xyz in line:
                st.write('Node set starts at line:', num)
                st.write(line)
                xyz = num
            if start_elements in line:
                st.write('Elements start at line:', num)
                st.write(line)
                elements = num
            if start_element_sets in line:
                st.write('Element sets start at line:', num)
                st.write(line)
                sets = num
        f.close()

    # Use the line values to define the skip rows list for pandas

    # Start of file to nodes line
    range_1 = list(range(0, int(xyz)))

    # Start of element line to end of file.
    range_2 = list(range(int(elements - 1), int(size)))

    # Start of file until the end of the elements.
    range_3 = list(range(0, int(elements + 1)))

    # Start of the element set until the end of the file.
    range_4 = list(range(int(sets - 1), int(size)))

    # Create a list with the individual range lists
    xyz_range = range_1 + range_2
    element_range = range_3 + range_4
    list(range(0, int(sets)))

    # Read in individual dataframes for the portions. It isn't efficient, but it is readable.
    xyz_df = pd.read_csv(in_name, header=None, sep=",", skiprows=xyz_range)

    # Use ravel to stack the rows untop of one another
    stacked_xyz = pd.DataFrame(xyz_df.iloc[:, 1:4].values.ravel('F'))

    # Define the elements dataframe
    elements_df = pd.read_csv(in_name, header=None, sep=",", index_col=False, skiprows=element_range)

    # Define the sets dataframe, which isn't really used but who knows what will happen
    #try:
    #    set_df = pd.read_csv(in_name, header=None, sep=",", index_col=False, skiprows=set_range)
    #    set_df.to_csv(str(in_name) + "elementsets.csv", header=False, sep = " ")


    with open(outname + ".case", 'w', newline="") as fout:
        # tells where to find the geometry (i.e. .geo file) and what format
        fout.write('FORMAT\ntype: ensight gold\n\nGEOMETRY\nmodel:                           ')
        # Scalars per node for 3d geometry
        fout.write(outname + ".geo\n")
        fout.close()
    with open(outname + '.geo', 'w', newline="") as fout:
        # Case header defining the name of the scalar array
        Title = str(outname) + ".inp"
        # fout to write the header
        fout.write("Title:  " + str(Title) + "\n")
        # fout to write the shape of the tetrahedral geometry (i.e. tetra4)
        fout.write('Description 2\nnode id given\nelement id given\npart\n         1\n')
        fout.write('Description PART\ncoordinates \n     ' + str(len(xyz_df)) + "\n")
        # use pandas in conjunction with fout to append the scalars
        xyz_df[0].to_csv(fout, header=False, index=False, sep=" ")
        # Write out the case format header with fout
        stacked_xyz.to_csv(fout, header=False, index=False, sep=" ")
        # Write out the elments
        fout.write("\ntria3\n" + str(len(elements_df)) + "\n")
        elements_df[0].to_csv(fout, header=False, index=False, sep=" ")
        elements_df.iloc[:, 1:].to_csv(fout, header=False, index=False, sep=" ")
        fout.close()

    #Write out the point cloud
    xyz_df.iloc[:, 1:4].to_csv(outname + "_pointcloud.csv", header=None, index=False)

    end = timer()
    end_time = (end - start)
    st.write("Coversion took", end_time, "\n")

def load_mesh(input_name):
    mesh_2d = trimesh.load(str(input_name))
    st.write(mesh_info(mesh_2d))
    return mesh_2d


def file_selector(directory='.'):
    filenames = os.listdir(directory)
    selected_filename = st.selectbox('Select a file', filenames)
    file_name = str(selected_filename)
    return file_name

st.title('2d to 3d mesh')


directory = Path(st.sidebar.text_input('What directory is the mesh file in?', ""))
output = Path(st.sidebar.text_input("What is the output directory?", ""))
file = file_selector(directory)
bone = st.sidebar.text_input("What bone is it?",
                             "example: canonical_Ovis_Ast")

#write_format = st.sidebar.checkbox("Write out with scientific notation?")
mesh_input = Path(directory).joinpath(file)

if output == "":
    output = Path(directory)

if st.button('Load mesh'):
    st.write("mesh file is:", file)
    st.write('The input is set to', mesh_input)
    st.write('The output is set  to', output)
    with st.spinner('Loading data...'):
        mesh = load_mesh(mesh_input)
    st.success('Done!')
    st.write(mesh_input)
    external_script = Path(r'Z:\RyanLab\Projects\NStephens\AM\plugins\Trimesh_gmsh_interface.py')
    #result = subprocess.run('python'
    #                        ' "' + str(external_script) + '" '
    #                        + '"' + str(mesh_input) + '"', shell=True)
    with st.spinner(f"Converting {file} to inp..."):
        result = subprocess.Popen(['python',
                                    str(external_script),
                                    str(directory),
                                    str(file),
                                    str(output)],
                                  shell=True)

if st.button('Process inp'):
    inp_name = Path(output).joinpath(file)
    st.write("reading in ", inp_name)
    inp_name = str(inp_name)[:-4]
    st.write("Name_before_add", inp_name)
    inp_name = inp_name + "_3d.inp"
    st.write(inp_name)
    case_outname = "canonical_" + str(bone)
    inp_file = inp_to_case(in_name=inp_name, outname=case_outname, out_dir = output)



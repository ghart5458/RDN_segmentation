'''
Script to take a 2d ply/off and convert it into a 3d volume mesh. It is recommened that the mesh be viewed and fixed
(e.g. remove floating vertices, creases, etc.) in meshlab prior to conversion.

If you have not already, you will need to install the following packages:
pip install gmsh-sdk

Author: Nicholas Stephens (nbs49@psu.edu)
Date: 06/21/2019

'''

import sys
from pathlib import Path
from timeit import default_timer as timer

import trimesh


def mesh_2d_to_mesh_3d(directory, input_name, output_dir):
    '''

    :param input_name:
    :return:
    '''
    print("\n\n\n\n")
    start = timer()
    # Read in the mesh using trimesh
    #name = str(input_name).replace(".off", "")
    name = str(input_name)[:-4]
    print("\n\n\n" + str(name))
    outname = str(name) + "_3d.inp"
    save_location = Path(str(output_dir)).joinpath(outname)
    print("Output name is ", save_location)
    input_name = Path(directory).joinpath(input_name)
    mesh_2d = trimesh.load(str(input_name))
    #Redefine the input_name for writing
    input_name = str(input_name)[:-4]
    print("Coverting", input_name, "to 3d...")
    #Set up the save name and location for the inp file
    #Other file types available are Nastran (.bdf), Gmsh (.msh), Abaqus (*.inp), Diffpack (*.diff) and Inria Medit (*.mesh)
    #Use the gmsh interface to turn the 2d surface into a 3d volume and write out the inp file
    #(the_mesh, file_name=Save_name/location, max_element=Max_length_of_element, mesher_id=algortihm to use
    #1: Delaunay, 4: Frontal, 7: MMG3D, 10: HXT)
    trimesh.interfaces.gmsh.to_volume(mesh_2d, file_name=str(save_location), max_element=None, mesher_id=1)
    #test = trimesh.interfaces.gmsh.to_volume(mesh_2d, file_name=None, max_element=None, mesher_id=1)
    end = timer()
    end_time = (end - start)
    print("Coversion took", end_time, "\n")


###############################
#                             #
# This is where you do stuff  #
#                             #
################################

#Define the directory where the ply/off file is
#dir = Path(r"Z:\RyanLab\Projects\AGuerra\pointclouds\Tarsometatarus")
directory = Path(str(sys.argv[1]))
file_name = Path(str(sys.argv[2]))
output_dir = Path(str(sys.argv[3]))


print("\n\n\n\nFile name is", file_name)

#Convert 2d mesh to 3d
mesh_2d_to_mesh_3d(directory=directory, input_name=file_name, output_dir=output_dir)

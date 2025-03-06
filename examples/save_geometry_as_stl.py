import sys
import os
import shutil

# Default path of the library
sys.path.insert(0, os.path.join(os.getcwd(), "/usr/local/lib/python3.8/site-packages"))
sys.path.insert(0, os.path.join(os.getcwd(), "../muspectre/meson-build-release/language_bindings/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../muspectre/meson-build-release/language_bindings/libmufft/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../muspectre/meson-build-release/language_bindings/libmugrid/python"))

import numpy as np
import meshio

import muSpectre as µ

import muChirality.Geometries as geo

def save_geometry_as_stl(nb_grid_pts, lengths, mask, output_file):
    """ Save a geometry as an assembly of triangles (compatible with stl-format).
    Input
    -----
    nb_grid_pts: list of 3 ints
        Number of voxels in each direction
    lengths: list of 3 floats
             Length of the representative volume element in each direction
    mask: np.array((Nx, Ny, Nz)) of floats
          Representation of geometry with mask = 1 representing material,
          mask = 0 representing void.
    output_file: string
                 File in which the geometry is saved.
    """
    Nx, Ny, Nz = nb_grid_pts
    # Global node indices
    def c2i(xp, yp, zp):
        return xp + (Nx + 1) * (yp + (Ny + 1) * zp)

    # Point coordinates
    x = np.linspace(0, lengths[0], Nx + 1, endpoint=True)
    y = np.linspace(0, lengths[1], Ny + 1, endpoint=True)
    z = np.linspace(0, lengths[2], Nz + 1, endpoint=True)
    points = np.array(np.meshgrid(x, y, z))
    points = np.transpose((points[0].ravel(order='F'), points[1].ravel(order='F'),
                          points[2].ravel(order='F')))

    # Triangles
    cells = []
    mask = mask.reshape(nb_grid_pts, order='F')
    for i in range(Nx):
        for j in range(Ny):
            if mask[i, j, 0] == 1:
                cells.append([c2i(i, j, 0), c2i(i+1, j, 0), c2i(i, j+1, 0)])
                cells.append([c2i(i+1, j, 0), c2i(i, j+1, 0), c2i(i+1, j+1, 0)])
            if mask[i, j, -1] == 1:
                cells.append([c2i(i, j, Nz), c2i(i+1, j, Nz), c2i(i, j+1, Nz)])
                cells.append([c2i(i+1, j, Nz), c2i(i, j+1, Nz), c2i(i+1, j+1, Nz)])
            for k in range(Nz-1):
                if mask[i, j, k] != mask[i, j, k+1]:
                    cells.append([c2i(i, j, k+1), c2i(i+1, j, k+1), c2i(i, j+1, k+1)])
                    cells.append([c2i(i+1, j, k+1), c2i(i, j+1, k+1), c2i(i+1, j+1, k+1)])

    for i in range(Nx):
        for k in range(Nz):
            if mask[i, 0, k] == 1:
                cells.append([c2i(i, 0, k), c2i(i+1, 0, k), c2i(i, 0, k+1)])
                cells.append([c2i(i+1, 0, k+1), c2i(i+1, 0, k), c2i(i, 0, k+1)])
            if mask[i, -1, k] == 1:
                cells.append([c2i(i, Ny, k), c2i(i+1, Ny, k), c2i(i, Ny, k+1)])
                cells.append([c2i(i+1, Ny, k+1), c2i(i+1, Ny, k), c2i(i, Ny, k+1)])
            for j in range(Ny - 1):
                if mask[i, j, k] != mask[i, j+1, k]:
                    cells.append([c2i(i, j+1, k), c2i(i+1, j+1, k), c2i(i, j+1, k+1)])
                    cells.append([c2i(i+1, j+1, k+1), c2i(i+1, j+1, k), c2i(i, j+1, k+1)])

    for j in range(Ny):
        for k in range(Nz):
            if mask[0, j, k] == 1:
                cells.append([c2i(0, j, k), c2i(0, j+1, k), c2i(0, j, k+1)])
                cells.append([c2i(0, j+1, k+1), c2i(0, j+1, k), c2i(0, j, k+1)])
            if mask[-1, j, k] == 1:
                cells.append([c2i(Nx, j, k), c2i(Nx, j+1, k), c2i(Nx, j, k+1)])
                cells.append([c2i(Nx, j+1, k+1), c2i(Nx, j+1, k), c2i(Nx, j, k+1)])
            for i in range(Nx-1):
                if mask[i, j, k] != mask[i+1, j, k]:
                    cells.append([c2i(i+1, j, k), c2i(i+1, j+1, k), c2i(i+1, j, k+1)])
                    cells.append([c2i(i+1, j+1, k+1), c2i(i+1, j+1, k), c2i(i+1, j, k+1)])

    # Save
    meshio.write_points_cells(
            output_file,
            points,
            {"triangle": cells})

def save_cylinder():
    ### ----- Parameter definitions ----- ###
    # Geometry
    lengths = [1, 1, 10]
    dim = len(lengths)
    radius = 0.4

    # Discretization
    Nx = 70
    Ny = Nx
    Nz = 30
    nb_grid_pts = [Nx, Ny, Nz]


    # Folder for saving
    folder = f'results/cylinder/'

    ### ----- Save geometry ----- ###
    # Geometry
    mask = geo.cylinder(nb_grid_pts, lengths, radius)

    # Save
    name = folder + f'geometry_nb_grid_pts={Nx}x{Ny}x{Nz}.stl'
    save_geometry_as_stl(nb_grid_pts, lengths, mask, name)

def save_chiral_1():
    ### ----- Parameter definitions ----- ###
    # Geometry
    lengths = [1, 1, 1]
    radius = 0.4
    thickness = 0.1
    angle_mat = 0.25
    angle_mat2 = 0.2

    # Discretization
    Nx = 30
    Ny = Nx
    Nz = Nx
    nb_grid_pts = [Nx, Ny, Nz]

    # Folder for saving
    folder = f'results/chiral1/'

    ### ----- Save geometry - first angle ----- ###
    # Geometry
    mask = geo.chiral_metamaterial(nb_grid_pts, lengths, radius, thickness,
                                   alpha=angle_mat)

    # Save
    name = folder + f'geometry_angle={angle_mat}_nb_grid_pts={Nx}x{Ny}x{Nz}.stl'
    save_geometry_as_stl(nb_grid_pts, lengths, mask, name)

    ### ----- Save geometry - second angle ----- ###
    # Geometry
    mask = geo.chiral_metamaterial(nb_grid_pts, lengths, radius, thickness,
                                   alpha=angle_mat2)

    # Save
    name = folder + f'geometry_angle={angle_mat2}_nb_grid_pts={Nx}x{Ny}x{Nz}.stl'
    save_geometry_as_stl(nb_grid_pts, lengths, mask, name)

def save_chiral_2():
    ### ----- Parameter definitions ----- ###
    # Geometry
    a = 0.5
    thickness = 0.06 * a
    radius_out = 0.4 * a
    radius_inn = 0.34 * a
    angle_mat = np.pi * 35 / 180

    # Discretization
    Nx = 70
    Ny = Nx
    Nz = Nx
    nb_grid_pts = [Nx, Ny, Nz]

    # Folder for saving
    folder = f'results/chiral2/'

    ### ----- Save geometry ----- ###
    # Geometry
    mask, lengths = geo.chiral_metamaterial_2(nb_grid_pts, a, radius_out,
                                              radius_inn, thickness,
                                              alpha=angle_mat)

    # Save
    name = folder + f'geometry_nb_grid_pts={Nx}x{Ny}x{Nz}.stl'
    save_geometry_as_stl(nb_grid_pts, lengths, mask, name)

def save_chiral_2_mult_unit_cells():
    ### ----- Parameter definitions ----- ###
    # Geometry
    a = 0.5
    thickness = 0.06 * a
    radius_out = 0.4 * a
    radius_inn = 0.34 * a
    angle_mat = np.pi * 35 / 180
    plates = True

    # Discretization
    Nx = 70
    Ny = Nx
    Nz = Nx
    nb_grid_pts_uc = [Nx, Ny, Nz]
    nb_unit_cells = [2, 2, 1]

    # Folder for saving
    folder = f'results/chiral2/'

    ### ----- Save geometry ----- ###
    # Geometry
    if plates:
        mask, lengths =\
            geo.chiral_2_with_plate(nb_unit_cells, nb_grid_pts_uc, a,
                                    radius_out, radius_inn,
                                    thickness, alpha=angle_mat)
    else:
        mask, lengths =\
            geo.chiral_2_mult_unit_cell(nb_unit_cells, nb_grid_pts_uc, a,
                                        radius_out, radius_inn,
                                        thickness, alpha=angle_mat)

    # Save
    if plates:
        name = folder + f'geometry_plates_nb_unit_cells={nb_unit_cells[0]}'
        name += f'x{nb_unit_cells[1]}x{nb_unit_cells[2]}_nb_grid_pts={Nx}x{Ny}x{Nz}.stl'
    else:
        name = folder + f'geometry_nb_unit_cells={nb_unit_cells[0]}x{nb_unit_cells[1]}'
        name += f'x{nb_unit_cells[2]}_nb_grid_pts={Nx}x{Ny}x{Nz}.stl'
    save_geometry_as_stl(mask.shape, lengths, mask, name)


if __name__ == "__main__":
    save_chiral_1()
    #save_chiral_2_mult_unit_cells()

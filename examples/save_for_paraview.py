"""
@file   save_for_paraview.py

@author Indre  Joedicke <indre.joedicke@imtek.uni-freiburg.de>

@date   18 Dec 2023

@brief  Test how to save data for paraview

Copyright © 2018 Till Junge

µSpectre is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

µSpectre is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with µSpectre; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.

Additional permission under GNU GPL version 3 section 7

If you modify this Program, or any covered work, by linking or combining it
with proprietary FFT implementations or numerical libraries, containing parts
covered by the terms of those libraries' licenses, the licensors of this
Program grant you additional permission to convey the resulting work.
"""
import sys
import os

# Path of the muSpectre library
sys.path.insert(0, os.path.join(os.getcwd(), "../../muspectre/build/language_bindings/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../../muspectre/build/language_bindings/libmufft/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../../muspectre/build/language_bindings/libmugrid/python"))
sys.path.insert(0, os.path.join(os.getcwd(), ".."))

import numpy as np
import matplotlib.pyplot as plt

import muSpectre as µ

import muChirality.Geometries as geo
import muChirality.Plotting as plot

def save_different_geometries():
    ### ----- Parameter definitions ----- ###
    # Unit cell
    nb_grid_pts_1 = [50, 50, 50]
    nb_grid_pts_2 = [21, 21, 21]
    case = 'even'
    # case = 'odd'
    lengths = [1, 1, 1]

    # muSpectre parameters
    formulation = µ.Formulation.small_strain
    gradient, weights = µ.linear_finite_elements.gradient_3d_5tet
    F0 = np.eye(3)

    # Cylinders
    radius = 0.3
    n = 2 # thickness in voxels
    thickness = lengths[0] / nb_grid_pts_1[0] * n

    # Rectangular beam
    Lbx = 0.6
    Lby = 0.3

    # Chiral materials
    a = 0.9
    radius_out = 0.25
    radius_inn = radius_out - thickness
    alpha = 0.2
    lengths = [1, 1, a - 1.5 * thickness]

    # Names of geometries
    names_geo = ['Minimal', 'Minimal_much_void', 'cylinder',
                 'hollow_cylinder', 'beam', 'not-chiral', 'chiral',
                 'chiral 2', 'chiral 2 (old)']

    ### ---- Save different geometries ----- ###
    if case == 'even':
        nb_grid_pts = nb_grid_pts_1
    elif case == 'odd':
        nb_grid_pts = nb_grid_pts_2
    else:
        raise ValueError(f'ERROR: Case {case} is not defined.')

    ### ----- Define geometries ----- ###
    masks = np.empty((len(names_geo), *nb_grid_pts))
    masks[0] = geo.minimal_structure(nb_grid_pts)
    if len(names_geo) > 1:
        masks[1] = geo.minimal_structure_much_void(nb_grid_pts)
    if len(names_geo) > 2:
        masks[2] = geo.cylinder(nb_grid_pts, lengths, radius)
    if len(names_geo) > 3:
        masks[3] = geo.hollow_cylinder(nb_grid_pts, lengths,
                                       radius, thickness)
    if len(names_geo) > 4:
        masks[4] = geo.rectangular_beam(nb_grid_pts, lengths, Lbx, Lby)
    if len(names_geo) > 5:
        masks[5] = geo.chiral_metamaterial(nb_grid_pts, lengths, radius, thickness)
    if len(names_geo) > 6:
        masks[6] = geo.chiral_metamaterial(nb_grid_pts, lengths, radius, thickness,
                                           alpha=alpha)
    if len(names_geo) > 7:
        masks[7] = geo.chiral_metamaterial_2(nb_grid_pts, lengths, radius_out,
                                             radius_inn, thickness, alpha=alpha)
    if len(names_geo) > 8:
        masks[8] = geo.chiral_metamaterial_2_old(nb_grid_pts, lengths, radius_out,
                                                 radius_inn, thickness, alpha=alpha)

    ### ----- Helper cell ----- ###
    # Define muSpectre cell to use functions for saving data
    cell = µ.Cell(nb_grid_pts, lengths, formulation, gradient, weights)

    # Initialize cell
    mat = µ.material.MaterialLinearElastic1_3d.make(cell, "hard", 10, 0)
    vac = µ.material.MaterialLinearElastic1_3d.make(cell, "vacuum", 0, 0)
    mask = masks[0].flatten(order='F')
    for pixel_id, pixel in cell.pixels.enumerate():
        if mask[pixel_id] == 1:
            mat.add_pixel(pixel_id)
        else:
            vac.add_pixel(pixel_id)

    cell.initialise()
    mask.reshape(nb_grid_pts_1, order='F')

    ### ----- Save geometries ----- ###
    # What to save
    cell_data = {}
    for i, name in enumerate(names_geo): #[:-1]):
        mask = masks[i]
        material = np.stack((mask, mask, mask, mask, mask), axis=0)
        material = material.flatten(order='F')
        material = material.reshape((1, -1))
        cell_data[name] = material

    # Saving
    if case == 'even':
        name = 'examples/plots/different_geometries_even.xdmf'
    elif case == 'odd':
        name = 'examples/plots/different_geometries_odd.xdmf'
    else:
        print('ERROR: Case is not defined.')
    µ.linear_finite_elements.write_3d(name, cell, cell_data=cell_data, point_data=None,
                                      F0=F0, displacement_field=True)


def save_chiral_2():
    ### ----- Parameter definitions ----- ###
    # Unit cell
    nb_grid_pts = [40, 40, 40] # Only for one unit cell!

    # Geometry
    a = 0.5
    thickness = 0.06 * a
    radius_out = 0.4 * a
    radius_inn = 0.34 * a
    angle_mat = np.pi * 35 / 180
    N_unit_cell_list = [1, 2, 3]

    b = 1.5 * thickness

    # muSpectre parameters
    formulation = µ.Formulation.small_strain
    gradient, weights = µ.linear_finite_elements.gradient_3d_5tet
    F0 = np.eye(3)

    # Show?
    show = True

    ### ----- Helper plots ----- ###
    mask, lengths = geo.chiral_metamaterial_2(nb_grid_pts, a,
                                              radius_out, radius_inn,
                                              thickness, alpha=angle_mat)
    fig2, ax = plt.subplots(1, 2)
    fig2.suptitle(f'z = 0')
    plot.plot_2D_cut(ax[0], mask, lengths, index = 0, plane = 2)
    plot.plot_2D_cut(ax[1], mask, lengths, index = 0, plane = 2)
    #helper = boundary
    #ax[0].plot([helper, helper], [0, lengths[1]], color='red')
    #helper = lengths[0] - boundary
    #ax[0].plot([helper, helper], [0, lengths[1]], color='red')
    #helper = boundary
    #ax[1].plot([helper, helper], [0, lengths[1]], color='red')
    #helper = lengths[0] - boundary - thickness
    #ax[1].plot([helper, helper], [0, lengths[1]], color='red')
    #helper = lengths[0] - boundary - b
    #ax[1].plot([helper, helper], [0, lengths[1]], '--', color='red')
    #if show:
    #   plt.show()

    ### ----- Define geometry ----- ###
    mask_list = []
    for N_uc in N_unit_cell_list:
        # Define geometry
        nb_unit_cells = [N_uc, N_uc]
        mask, lengths =\
            geo.chiral_2_mult_unit_cell(nb_unit_cells, nb_grid_pts, a,
                                        radius_out, radius_inn,
                                        thickness, alpha=angle_mat)
        mask_list.append(mask)
        shape = mask.shape
        hx = lengths[0] / shape[0]
        hy = lengths[1] / shape[1]
        hz = lengths[2] / shape[2]
        print(f'nb_unit_cells = {N_uc} x {N_uc}')
        print(f'nb_grid_pts = {shape[0]}x{shape[1]}x{shape[2]}')
        print(f'lengths = {lengths[0]}x{lengths[1]}x{lengths[2]}')
        print(f'hx = {hx}')
        print(f'hy = {hy}')
        print(f'hz = {hz}')

        ### ----- Plotting ----- ###
        fig, ax = plt.subplots()
        fig.suptitle(f'z = 0')
        plot.plot_2D_cut(ax, mask, lengths, index = 0, plane = 2)
        if show:
            plt.show()
        plt.close(fig)

    ### ----- Save for paraview ----- ###
    nb_grid_pts = mask_list[-1].shape

    # Define muSpectre cell to use functions for saving data
    cell = µ.Cell(nb_grid_pts, lengths, formulation, gradient, weights)
    mat = µ.material.MaterialLinearElastic1_3d.make(cell, "hard", 10, 0)
    vac = µ.material.MaterialLinearElastic1_3d.make(cell, "vacuum", 0, 0)
    mask = mask.flatten(order='F')
    for pixel_id, pixel in cell.pixels.enumerate():
        if mask[pixel_id] == 1:
            mat.add_pixel(pixel_id)
        else:
            vac.add_pixel(pixel_id)
    cell.initialise()
    mask.reshape(nb_grid_pts, order='F')

    # What to save
    cell_data = {}
    for i, mask in enumerate(mask_list): #[:-1]):
        N_uc = N_unit_cell_list[i]
        helper = np.zeros(nb_grid_pts)
        shape = mask.shape
        helper[0:shape[0], 0:shape[1], 0:shape[2]] = mask
        material = np.stack((helper, helper, helper, helper, helper), axis=0)
        material = material.flatten(order='F')
        material = material.reshape((1, -1))
        name = f'nb_unit_cells={N_uc}x{N_uc}'
        cell_data[name] = material

    # Saving
    name = 'plots/different_chiral_metamaterials_2.xdmf'
    µ.linear_finite_elements.write_3d(name, cell, cell_data=cell_data, point_data=None,
                                      F0=F0, displacement_field=True)

if __name__ == "__main__":
    #save_different_geometries()
    save_chiral_2()

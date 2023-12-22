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

# Default path of the library
sys.path.insert(0, os.path.join(os.getcwd(), "./muspectre/build/language_bindings/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "./muspectre/build/language_bindings/libmufft/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "./muspectre/build/language_bindings/libmugrid/python"))

import numpy as np
import matplotlib.pyplot as plt

import muSpectre as µ

import muChirality.Geometries as geo

def save_different_geometries():
    ### ----- Parameter definitions ----- ###
    # Unit cell
    nb_grid_pts_1 = [20, 20, 20]
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
                 'chiral 2']

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

if __name__ == "__main__":
    save_different_geometries()

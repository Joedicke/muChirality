"""
@file   CalculationsTorsion.py

@author Indre  Joedicke <indre.joedicke@imtek.uni-freiburg.de>

@date   17 Aug 2023

@brief  Functions for calculating interesting data from the muSpectre
        result for torsion.

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

import muSpectre as µ
from muSpectre.gradient_integration import get_complemented_positions

def calculations(strain, stress, cell, eigen_class, detailed=False):
    """ Calculate interesting data for torsion.
    Parameters
    ----------
    strain: np.ndarray of floats
            Strain field. Shape = [3, 3, cell.nb_quad_pts,
                                      cell.nb_domain_grid_pts]
    stress: np.ndarray of floats
            Stress field. Shape = [3, 3, cell.nb_quad_pts,
                                      cell.nb_domain_grid_pts]
    cell: muSpectre cell object
    detailed: boolean
              If True, more detailed data is returned. Default is False.
    Returns
    -------
    pos: np.ndarray(3, cell.nb_domain_grid_pts+1) of floats
         Original position of the grid pts
    displ: np.ndarray(3, cell.nb_domain_grid_pts+1) of floats
           Displacement field of the grid pts
    force_z: float
             Average force in z-direction.
    moment: float
            Average moment. Is only calculated if detailed=True.
    moment_of_z: np.ndarray(10*cell.nb_domain_grid_pts[2]) of floats
                 Moment depending on z. Is only calculated if
                 detailed=True.
    stiffness: float
               Stiffness. Is only calculated if detailed=True.
    """
    ### ----- Parameters ----- ###
    lengths = cell.domain_lengths
    nb_grid_pts = cell.nb_domain_grid_pts
    angle = eigen_class.angle
    x_rot_axis = eigen_class.x_rot_axis
    y_rot_axis = eigen_class.y_rot_axis
    x = np.linspace(0, lengths[0], nb_grid_pts[0]+1, endpoint=True)
    y = np.linspace(0, lengths[1], nb_grid_pts[1]+1, endpoint=True)
    z = np.linspace(0, lengths[2], nb_grid_pts[2]+1, endpoint=True)
    hx = lengths[0] / nb_grid_pts[0]
    hy = lengths[1] / nb_grid_pts[1]
    hz = lengths[2] / nb_grid_pts[2]

    ### ----- Displacement and force ----- ###
    # Calculate position + displacement
    F0=np.eye(3)
    strain_no_eigen = strain.copy()
    eigen_class.remove_eigen_strain_func(strain_no_eigen)
    [x_0, y_0, z_0], [x_displ, y_displ, z_displ] \
        = get_complemented_positions("0d", cell, strain_array=strain_no_eigen, F0=F0,
                                     periodically_complemented=True)
    helper = - angle * np.einsum('i,j->ij', y-y_rot_axis, z)
    x_displ += helper[None, :, :]
    helper = angle * np.einsum('i,j->ij', x-x_rot_axis, z)
    y_displ += helper[:, None, :]
    pos = np.asarray([x_0, y_0, z_0])
    displ = np.asarray([x_displ, y_displ, z_displ])

    # Calculate force_z
    force_z = hx * hy * hz / 6 * np.sum(stress[2, 2])
    force_z += hx * hy * hz / 6 * np.sum(stress[2, 2, 0])
    force_z = force_z / lengths[2]

    ### ----- Additional data, only calculated if detailed=True ----- ###
    if detailed:
        x = np.arange(nb_grid_pts[0]) * hx # x-coordinate of voxel
        y = np.arange(nb_grid_pts[1]) * hy # y-coordinate of voxel
        z = np.arange(nb_grid_pts[2]) * hz # z-coordinate of voxel
        volume = hx * hy * hz / 6 # Volume of one of the four corner tetrahedras

        # Coordinates of the centers of the 5 tetradras
        X, Y, Z = np.meshgrid(x, y, z)
        # Format X[i, j, k, l] with i: quad_pt, j: x-coord, k: y-coord, l: z-coord
        X = X.transpose((1, 0, 2))
        Y = Y.transpose((1, 0, 2))
        X = np.stack((X, X, X, X, X))
        Y = np.stack((Y, Y, Y, Y, Y))
        X[0] += 0.5 * hx
        X[1] += 0.25 * hx
        X[2] += 0.75 * hx
        X[3] += 0.75 * hx
        X[4] += 0.25 * hx
        Y[0] += 0.5 * hy
        Y[1] += 0.25 * hy
        Y[2] += 0.75 * hy
        Y[3] += 0.25 * hy
        Y[4] += 0.75 * hy

        # Calculate average moment
        helper = - stress[0, 2] * (Y - y_rot_axis)
        helper += stress[1, 2] * (X - x_rot_axis)
        moment = volume * np.sum(helper)
        moment += volume * np.sum(helper[0])
        moment = moment / lengths[2]

        # Calculate moment - details
        z = np.arange(10*nb_grid_pts[2])
        z = z % 10
        z = z * hz / 10
        z = z.reshape((-1, 10))
        surface1 = 2 * hx * hy * z / hz * (1 - z / hz)
        surface2 = 0.5 * hx * hy * (1 - z / hz) ** 2
        surface3 = 0.5 * hx * hy * (z / hz) ** 2
        moment_of_z = np.sum(helper[0], axis=(0, 1))[:, None] * surface1
        moment_of_z += np.sum(helper[1], axis=(0, 1))[:, None] * surface2
        moment_of_z += np.sum(helper[2], axis=(0, 1))[:, None] * surface2
        moment_of_z += np.sum(helper[3], axis=(0, 1))[:, None] * surface3
        moment_of_z += np.sum(helper[4], axis=(0, 1))[:, None] * surface3
        moment_of_z = moment_of_z.reshape(-1)

        # Calculate stiffness
        stiffness = moment / angle
        return pos, displ, force_z, moment, moment_of_z, stiffness
    else:
        return pos, displ, force_z

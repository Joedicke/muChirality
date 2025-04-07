"""
@file   EigenStrainTorsion.py

@author Indre  Joedicke <indre.joedicke@imtek.uni-freiburg.de>

@date   18 Jul 2023

@brief  Class for eigenstrains due to loading with twist

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
import numpy as np

class EigenStrain:
    """
    Class for describing the eigenstrain due to imposed torsion along z-axis
    for a five tetrahedra discretization.

    Attributes
    ----------
    nb_grid_pts: list of 3 ints
                 Number of grid pts in each direction
    pixels: µSpectre pixels object
            pixels of the discretized unit cell
    angle: float
           imposed rotation angle
    hx: float
        voxel length in x-direction
    hy: float
        voxel length in y-direction
    x_rot_axis: float
                x-coordinate of the rotation axis
    y_rot_axis: float
                y-coordinate of the rotation axis

    Methods
    -------
    eigen_strain_func(step_nb, strain_field):
          Change the strain_field to account for the eigen strain.
    remove_eigen_strain_func(strain_field):
          Inverse of eigen_strain_func.
    """
    def __init__(self, pixels, angle, lengths, nb_grid_pts, x_rot_axis, y_rot_axis):
        """
        Parameters
        ----------
        pixels: µSpectre pixels object
                pixels of the discretized unit cell
        angle: float
               imposed rotation angle
        lengths: List of 3 floats
                 Lengths of the unit cell in cartesian directions
        nb_grid_pts: List of 3 ints
             Number of voxels in cartesian directions
        x_rot_axis: float
             x-coordinate of the rotation axis
        y_rot_axis: float
             y-coordinate of the rotation axis
        """
        self.nb_grid_pts = nb_grid_pts
        self.pixels = pixels
        self.angle = angle
        self.hx = lengths[0] / nb_grid_pts[0]
        self.hy = lengths[1] / nb_grid_pts[1]
        self.x_rot_axis = x_rot_axis
        self.y_rot_axis = y_rot_axis

    def __call__(self, nb_steps, strain_field):
        self.eigen_strain_func(step_nb, strain_field)

    def eigen_strain_func(self, step_nb, strain_field):
        """
        Change the strain_field to account for the eigen strain.

        Parameters
        ----------
        step_nb: int
                 Number of load step
        strain_field: np.array(3, 3, 5, *nb_grid_pts) of floats
                      Strain field
        """
        hx = self.hx
        hy = self.hy
        x_rot_axis = self.x_rot_axis
        y_rot_axis = self.y_rot_axis
        angle = self.angle

        # Coordinates of voxel
        x = np.arange(self.nb_grid_pts[0]) * hx # x-coordinate of voxel
        y = np.arange(self.nb_grid_pts[1]) * hy # y-coordinate of voxel

        # Difference between voxel coordinate and quadrature point
        # coordinate
        delta_x = np.array([0.5, 0.25, 0.75, 0.75, 0.25]) * hx
        delta_y = np.array([0.5, 0.25, 0.75, 0.25, 0.75]) * hy

        # Eigenstrain
        strain_field[0, 2] -= 0.5 * angle * (y[None, None, :, None] + delta_y[:, None, None, None] - y_rot_axis)
        strain_field[2, 0] -= 0.5 * angle * (y[None, None, :, None] + delta_y[:, None, None, None] - y_rot_axis)
        strain_field[1, 2] += 0.5 * angle * (x[None, :, None, None] + delta_x[:, None, None, None] - x_rot_axis)
        strain_field[2, 1] += 0.5 * angle * (x[None, :, None, None] + delta_x[:, None, None, None] - x_rot_axis)

    def eigen_strain_func_old(self, step_nb, strain_field):
        """
        Change the strain_field to account for the eigen strain.
        Note: Saves the complete position fields for all quad points and voxels.
        Therefore, it is not memory-efficient

        Parameters
        ----------
        step_nb: int
                 Number of load step
        strain_field: np.array(3, 3, 5, *nb_grid_pts) of floats
                      Strain field
        """
        hx = self.hx
        hy = self.hy
        x = np.arange(self.nb_grid_pts[0]) * hx # x-coordinate of voxel
        y = np.arange(self.nb_grid_pts[1]) * hy # y-coordinate of voxel
        x_rot_axis = self.x_rot_axis
        y_rot_axis = self.y_rot_axis
        angle = self.angle

        # x-y-coordinates of the centers of the 5 tetradras
        X, Y = np.meshgrid(x, y)
        X = X.transpose() # Format X[i, j, k] with i: quad_pt, j: x-coord, k: y-coord
        Y = Y.transpose()
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

        # Eigenstrain
        strain_field[0, 2] -= 0.5 * angle * (Y[:, :, :, None] - y_rot_axis)
        strain_field[2, 0] -= 0.5 * angle * (Y[:, :, :, None] - y_rot_axis)
        strain_field[1, 2] += 0.5 * angle * (X[:, :, :, None] - x_rot_axis)
        strain_field[2, 1] += 0.5 * angle * (X[:, :, :, None] - x_rot_axis)

    def remove_eigen_strain_func(self, strain_field):
        """
        Change the strain_field to remove the influence of the eigen strain.

        Parameters
        ----------
        strain_field: np.array(3, 3, 5, *nb_grid_pts) of floats
                      Strain field
        """
        hx = self.hx
        hy = self.hy
        x = np.arange(self.nb_grid_pts[0]) * hx # x-coordinate of voxel
        y = np.arange(self.nb_grid_pts[1]) * hy # y-coordinate of voxel
        x_rot_axis = self.x_rot_axis
        y_rot_axis = self.y_rot_axis
        angle = self.angle

        # x-y-coordinates of the centers of the 5 tetradras
        X, Y = np.meshgrid(x, y)
        X = X.transpose() # Format X[i, j, k] with i: quad_pt, j: x-coord, k: y-coord
        Y = Y.transpose()
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

        # Remove eigenstrain
        strain_field[0, 2] += 0.5 * angle * (Y[:, :, :, None] - y_rot_axis)
        strain_field[2, 0] += 0.5 * angle * (Y[:, :, :, None] - y_rot_axis)
        strain_field[1, 2] -= 0.5 * angle * (X[:, :, :, None] - x_rot_axis)
        strain_field[2, 1] -= 0.5 * angle * (X[:, :, :, None] - x_rot_axis)

"""
@file   EigenStrainBending.py

@author Indre  Joedicke <indre.joedicke@imtek.uni-freiburg.de>

@date   24 Jan 2025

@brief  Class for eigenstrains due to loading with bending

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

class EigenStrainBending3D:
    """
    Class for describing the eigenstrain due to imposed bending around y-axis
    for a five tetrahedra discretization.

    Attributes
    ----------
    nb_grid_pts: list of 3 ints
                 Number of grid pts in each direction
    pixels: µSpectre pixels object
            pixels of the discretized unit cell
    curvature: float
               imposed bending curvature
    hz: float
        voxel length in z-direction

    Methods
    -------
    eigen_strain_func(step_nb, strain_field):
          Change the strain_field to account for the eigen strain.
    remove_eigen_strain_func(strain_field):
          Inverse of eigen_strain_func.
    """
    def __init__(self, pixels, curvature, lengths, nb_grid_pts, Poisson):
        """
        Parameters
        ----------
        pixels: µSpectre pixels object
                pixels of the discretized unit cell
        curvature: float
                   Imposed curvature
        lengths: List of 3 floats
                 Lengths of the unit cell in cartesian directions
        nb_grid_pts: List of 3 ints
             Number of voxels in cartesian directions
        """
        self.nb_grid_pts = nb_grid_pts
        self.pixels = pixels
        self.curvature = curvature
        self.hz = lengths[2] / nb_grid_pts[2]
        self.Poisson = Poisson

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
        # z-coordinates of the centers of the 5 tetradras
        z = np.arange(self.nb_grid_pts[2]) * self.hz
        Z = np.stack((z, z, z, z, z))
        Z[0] += 0.5 * self.hz
        Z[1] += 0.25 * self.hz
        Z[2] += 0.25 * self.hz
        Z[3] += 0.75 * self.hz
        Z[4] += 0.75 * self.hz

        # Eigenstrain
        strain_field[0, 0] += self.curvature * Z[:, None, None, :]
        strain_field[1, 1] -= self.curvature * self.Poisson * Z[:, None, None, :]
        strain_field[2, 2] -= self.curvature * self.Poisson * Z[:, None, None, :]

    def remove_eigen_strain_func(self, strain_field):
        """
        Change the strain_field to remove the influence of the eigen strain.

        Parameters
        ----------
        strain_field: np.array(3, 3, 5, *nb_grid_pts) of floats
                      Strain field
        """
        # z-coordinates of the centers of the 5 tetradras
        z = np.arange(self.nb_grid_pts[2]) * self.hz
        Z = np.stack((z, z, z, z, z))
        Z[0] += 0.5 * self.hz
        Z[1] += 0.25 * self.hz
        Z[2] += 0.25 * self.hz
        Z[3] += 0.75 * self.hz
        Z[4] += 0.75 * self.hz

        # Remove Eigenstrain
        strain_field[0, 0] -= self.curvature * Z[:, None, None, :]
        strain_field[1, 1] += self.curvature * self.Poisson * Z[:, None, None, :]
        strain_field[2, 2] += self.curvature * self.Poisson * Z[:, None, None, :]

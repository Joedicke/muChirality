"""
@file   torsion_square_beam.py

@author Indre  Joedicke <indre.joedicke@imtek.uni-freiburg.de>

@date   29 Aug 2023

@brief  Test torsion for a beam with a square cross section

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
sys.path.insert(0, os.path.join(os.getcwd(), "../muspectre/meson-build-release/language_bindings/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../muspectre/meson-build-release/language_bindings/libmufft/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../muspectre/meson-build-release/language_bindings/libmugrid/python"))

import numpy as np
import matplotlib.pyplot as plt

import muSpectre as µ
#import muGrid

from muChirality.EigenStrainTorsion import EigenStrain
from muChirality.Geometries import rectangular_beam
from muChirality.CalculationsTorsion import calculations

def test_torsion_square_beam():
    """ Test a square beam under torsion by comparing
        it with the analytical solution.
    """
    ### ----- Parameter definitions ----- ###
    # Geometry
    lengths = [1, 1, 3]
    L_beam = 0.6

    # Discretization
    #Nxy_list = [20, 30, 40]
    Nxy_list = [30, 50, 70, 90, 110]
    Nz = 30
    gradient, weights = µ.linear_finite_elements.gradient_3d_5tet

    # Material
    Young = 100
    Poisson = 0

    # Loading
    angle = 0.15
    x_rot_axis = lengths[0] / 2
    y_rot_axis = lengths[1] / 2
    delta_F = np.zeros((3, 3))

    # Formulation
    formulation = µ.Formulation.small_strain

    # muSpectre solver parameters
    newton_tol       = 1e-7
    cg_tol           = 1e-7 # tolerance for cg algo
    equil_tol        = 1e-7 # tolerance for equilibrium
    maxiter          = 10000
    verbose          = µ.Verbosity.Silent

    # Folder for saving
    folder = 'examples/plots/square_beam/'

    # What to test
    abs_err_stress = np.empty(len(Nxy_list))
    rel_err_stress = np.empty(len(Nxy_list))
    rel_err_displ = np.empty(len(Nxy_list))
    abs_err_force = np.empty(len(Nxy_list))
    rel_err_moment = np.empty(len(Nxy_list))
    rel_err_stiff = np.empty(len(Nxy_list))

    ### ----- Calculations ----- ###
    for ind_N, Nxy in enumerate(Nxy_list):
        nb_grid_pts = [Nxy, Nxy, Nz]
        message = f'Test {ind_N+1} of {len(Nxy_list)}: '
        message += f'Nxy = {Nxy}'
        print(message)

        ### ----- muSpectre calculation ----- ###
        # Geometry
        mask = rectangular_beam(nb_grid_pts, lengths, L_beam, L_beam)
        hx = lengths[0] / nb_grid_pts[0]
        hy = lengths[1] / nb_grid_pts[1]
        hz = lengths[2] / nb_grid_pts[2]

        x = np.arange(nb_grid_pts[0]) * hx # x-coordinate of voxel
        y = np.arange(nb_grid_pts[1]) * hy # y-coordinate of voxel
        z = np.arange(nb_grid_pts[2]) * hz # z-coordinate of voxel

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

        # Initialize muSpectre cell
        cell = µ.Cell(nb_grid_pts, lengths, formulation, gradient, weights)
        mat = µ.material.MaterialLinearElastic1_3d.make(cell, "hard", Young, Poisson)
        vac = µ.material.MaterialLinearElastic1_3d.make(cell, "vacuum", 0, 0)
        mask = mask.flatten(order='F')
        for pixel_id, pixel in cell.pixels.enumerate():
            if mask[pixel_id] == 1:
                mat.add_pixel(pixel_id)
            else:
                vac.add_pixel(pixel_id)

        cell.initialise()
        solver = µ.solvers.KrylovSolverCG(cell, cg_tol, maxiter, verbose)

        # Initialize Eigen class
        eigen_class = EigenStrain(cell.pixels, angle, lengths, nb_grid_pts,
                                  x_rot_axis, y_rot_axis)

        # Solving
        res = µ.solvers.newton_cg(cell, delta_F, solver, newton_tol, equil_tol,
                                  verbose, μ.solvers.IsStrainInitialised.No,
                                  µ.StoreNativeStress.No, eigen_class.eigen_strain_func)
        shape = (3, 3, 5, *nb_grid_pts)
        stress_num = res.stress.reshape(shape, order='F')
        strain_num = res.grad.reshape(shape, order='F')

        results = calculations(strain_num, stress_num, cell, eigen_class, detailed=True)
        pos, displ, force, moment, moment_of_z, stiff = results

        ### ----- Comparison with analytical solution ----- ###
        # Stiffness
        mu = Young / 2 / (1 + Poisson)
        stiffness_ana = mu * 0.141 * L_beam ** 4
        rel_err_stiff[ind_N] = np.linalg.norm(stiff-stiffness_ana) / abs(stiffness_ana) * 100

    ### ----- Plotting ----- ###
    # Plot stiffness error
    fig, ax = plt.subplots()
    fig.suptitle('Error of stiffness')
    ax.set_xlabel('Nx=Ny')
    ax.set_ylabel('Relative err (%)')
    ax.plot(Nxy_list, rel_err_stiff, marker='x')
    name = folder + 'stiffness_error.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)

    # Save file for paraview
    F0 = np.eye(3)
    cell_data = {}
    material = np.stack((mask, mask, mask, mask, mask), axis=0)
    material = material.flatten(order='F')
    material = material.reshape((1, -1))
    stress = stress_num.reshape((3, 3, -1), order='F').T.swapaxes(1, 2)
    stress = stress.reshape((1, -1, 3, 3)).copy()
    strain = strain_num.reshape((3, 3, -1), order='F').T.swapaxes(1, 2)
    strain = strain.reshape((1, -1, 3, 3)).copy()
    cell_data = {"material": material, "stress_field": stress, "strain_field": strain}
    point_data = {"displ": displ.reshape((3, -1), order='F').T}
    name = folder + 'square_beam_torsion.xdmf'
    µ.linear_finite_elements.write_3d(name, cell, cell_data=cell_data, point_data=point_data,
                                      F0=F0, displacement_field=True)


if __name__ == "__main__":
    test_torsion_square_beam()

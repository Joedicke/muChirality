"""
@file   torsion_cylinder.py

@author Indre  Joedicke <indre.joedicke@imtek.uni-freiburg.de>

@date   15 Aug 2023

@brief  Test torsion for a cylinder

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
from muChirality.Geometries import cylinder
from muChirality.CalculationsTorsion import calculations

def test_torsion_cylinder():
    """ Test a cylinder under torsion by comparison
        with the analytical solution.
    """
    ### ----- Parameter definitions ----- ###
    # Geometry
    lengths = [1, 1, 10]
    radius = 0.4

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
    folder = 'examples/plots/cylinder/'

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
        mask = cylinder(nb_grid_pts, lengths, radius)
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
        mask = mask.reshape(nb_grid_pts, order='F')

        # Analytical stress
        mu = Young / 2 / (1 + Poisson) # Shear modulus
        stress_ana = np.zeros(shape)
        stress_ana[0, 2] = - angle * mu * (Y - y_rot_axis)
        stress_ana[0, 2] *= mask[None, :, :, :] # No stress in void
        stress_ana[1, 2] = angle * mu * (X - x_rot_axis)
        stress_ana[1, 2] *= mask[None, :, :, :]
        stress_ana[2, 0] = stress_ana[0, 2]
        stress_ana[2, 1] = stress_ana[1, 2]

        # Stress error
        err = np.linalg.norm(stress_ana - stress_num)
        abs_err_stress[ind_N] = err
        rel_err_stress[ind_N] = err / np.linalg.norm(stress_ana) * 100
        if ind_N == 0:
            stress_coarse = stress_num.copy()
            err_stress_coarse = stress_coarse - stress_ana
        elif ind_N == len(Nxy_list)-1:
            stress_fine = stress_num.copy()
            err_stress_fine = stress_fine - stress_ana

        # Displacements
        x = np.linspace(0, lengths[0], nb_grid_pts[0]+1, endpoint=True)
        y = np.linspace(0, lengths[1], nb_grid_pts[1]+1, endpoint=True)
        z = np.linspace(0, lengths[2], nb_grid_pts[2]+1, endpoint=True)
        displ_ana = np.zeros(displ.shape)
        helper = - angle * np.einsum('i,j->ij', y-y_rot_axis, z)
        displ_ana[0] += helper[None, :, :]
        helper = angle * np.einsum('i,j->ij', x-x_rot_axis, z)
        displ_ana[1] += helper[:, None, :]
        err = np.linalg.norm(displ_ana - displ)
        rel_err_displ[ind_N] = err / np.linalg.norm(displ_ana) * 100

        # Force
        abs_err_force[ind_N] = np.linalg.norm(force)

        # Moment and stiffness
        stiffness_ana = mu * np.pi * radius ** 4 / 2
        moment_ana = stiffness_ana * angle
        rel_err_moment[ind_N] = np.linalg.norm(moment-moment_ana) / moment_ana * 100
        rel_err_stiff[ind_N] = np.linalg.norm(stiff-stiffness_ana) / stiffness_ana * 100
        if ind_N == 0:
            moment_of_z_coarse = moment_of_z.copy()
            moment_coarse = moment
        elif ind_N == len(Nxy_list)-1:
            moment_of_z_fine = moment_of_z.copy()
            moment_fine = moment

    ### ----- Plotting ----- ###
    stress_labels = ['xx', 'xy', 'xz', 'yy', 'yz', 'zz']
    stress_indices = [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]]

    # Plot stress distribution
    index = 0
    for i in range(6):
        # Prepare figure
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        title = 'Stress_' + stress_labels[i] + f' (0.th quad pt) at'
        title += f' z={z[index]}'
        fig.suptitle(title)
        axes[0, 0].set_title(f'Value (Nx=Ny={Nxy_list[0]})')
        axes[0, 1].set_title(f'Value (Nx=Ny={Nxy_list[-1]})')
        axes[1, 0].set_title(f'Error (Nx=Ny={Nxy_list[0]})')
        axes[1, 1].set_title(f'Error (Nx=Ny={Nxy_list[-1]})')
        axes[0, 0].set_aspect('equal')
        axes[0, 1].set_aspect('equal')
        axes[1, 0].set_aspect('equal')
        axes[1, 1].set_aspect('equal')

        # Plot stress + err coarse
        stress = stress_coarse[stress_indices[i][0], stress_indices[i][1],
                               0, :, :, index]
        x = np.linspace(0, lengths[0], Nxy_list[0]+1)
        y = np.linspace(0, lengths[1], Nxy_list[0]+1)
        im = axes[0, 0].pcolormesh(x, y, stress.T, rasterized=True)
        fig.colorbar(im, ax=axes[0, 0])
        stress = err_stress_coarse[stress_indices[i][0], stress_indices[i][1],
                               0, :, :, index]
        im = axes[1, 0].pcolormesh(x, y, stress.T, rasterized=True)
        fig.colorbar(im, ax=axes[1, 0])

        # Plot stress + err fine
        stress = stress_fine[stress_indices[i][0], stress_indices[i][1],
                             0, :, :, index]
        x = np.linspace(0, lengths[0], Nxy_list[-1]+1)
        y = np.linspace(0, lengths[1], Nxy_list[-1]+1)
        im = axes[0, 1].pcolormesh(x, y, stress.T, rasterized=True)
        fig.colorbar(im, ax=axes[0, 1])
        stress = err_stress_fine[stress_indices[i][0], stress_indices[i][1],
                             0, :, :, index]
        im = axes[1, 1].pcolormesh(x, y, stress.T, rasterized=True)
        fig.colorbar(im, ax=axes[1, 1])

        # Save figure
        name = folder + 'stress_' + stress_labels[i] + '_distribution.pdf'
        fig.savefig(name, bbox_inches='tight')
        plt.close(fig)

    # Plot stress error
    fig, ax = plt.subplots(2)
    fig.suptitle('Error of stress')
    ax[0].set_ylabel('Absolute')
    ax[0].set_xlabel('Nx=Ny')
    ax[1].set_ylabel('Relative (%)')
    ax[1].set_xlabel('Nx=Ny')
    ax[0].plot(Nxy_list, abs_err_stress, marker='x')
    ax[1].plot(Nxy_list, rel_err_stress, marker='x')
    name = folder + 'stress_error.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)

    # Plot displ error
    fig, ax = plt.subplots()
    fig.suptitle('Error of displacement')
    ax.set_xlabel('Nx=Ny')
    ax.set_ylabel('Relative err (%)')
    ax.plot(Nxy_list, rel_err_displ, marker='x')
    name = folder + 'displ_error.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)

    # Plot force error
    fig, ax = plt.subplots()
    fig.suptitle('Error of force_z')
    ax.set_xlabel('Nx=Ny')
    ax.set_ylabel('Absolute err')
    ax.plot(Nxy_list, abs_err_force, marker='x')
    name = folder + 'force_error.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)

    # Plot moment error
    fig, ax = plt.subplots()
    fig.suptitle('Error of moment')
    ax.set_xlabel('Nx=Ny')
    ax.set_ylabel('Relative err (%)')
    ax.plot(Nxy_list, rel_err_moment, marker='x')
    name = folder + 'moment_error.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)

    # Plot stiffness error
    fig, ax = plt.subplots()
    fig.suptitle('Error of stiffness')
    ax.set_xlabel('Nx=Ny')
    ax.set_ylabel('Relative err (%)')
    ax.plot(Nxy_list, rel_err_stiff, marker='x')
    name = folder + 'stiffness_error.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)

    # Plot dependence of moment and z
    fig, ax = plt.subplots()
    fig.suptitle('Moment depending on z')
    ax.set_xlabel('z')
    ax.set_ylabel('Moment')
    ax.plot([0, lengths[2]], [moment_ana, moment_ana],
            label='analyt',
            linestyle=':', color='black', linewidth=4)
    z = np.linspace(0, lengths[2], moment_of_z_coarse.size)
    ax.plot(z, moment_of_z_coarse, label=f'Nx=Ny={Nxy_list[0]}',
            linestyle='-', color='blue')
    ax.plot([0, lengths[2]], [moment_coarse, moment_coarse],
            label=f'Nx=Ny={Nxy_list[0]} (aver.)',
            linestyle='--', color='blue')
    z = np.linspace(0, lengths[2], moment_of_z_fine.size)
    ax.plot(z, moment_of_z_fine, label=f'Nx=Ny={Nxy_list[-1]}',
            linestyle='-', color='red')
    ax.plot([0, lengths[2]], [moment_fine, moment_fine],
            label=f'Nx=Ny={Nxy_list[-1]} (aver.)',
            linestyle='--', color='red')
    ax.legend()
    name = folder + 'moment_of_z.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)

    plt.show()

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
    name = folder + 'cylinder.xdmf'
    µ.linear_finite_elements.write_3d(name, cell, cell_data=cell_data, point_data=point_data,
                                      F0=F0, displacement_field=True)


if __name__ == "__main__":
    test_torsion_cylinder()

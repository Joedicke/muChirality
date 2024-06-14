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
import shutil

# Default path of the library
sys.path.insert(0, os.path.join(os.getcwd(), "/usr/local/lib/python3.8/site-packages"))
sys.path.insert(0, os.path.join(os.getcwd(), "../muspectre/meson-build-release/language_bindings/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../muspectre/meson-build-release/language_bindings/libmufft/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../muspectre/meson-build-release/language_bindings/libmugrid/python"))

import numpy as np
import matplotlib.pyplot as plt

import muSpectre as µ
from NuMPI import MPI
from NuMPI.Tools import Reduction

from muChirality.EigenStrainTorsion import EigenStrain
from muChirality.Geometries import cylinder
from muChirality.CalculationsTorsion import calculations
from muSpectre.gradient_integration import get_complemented_positions

def plotting(folder, lengths, angle_per_length, displ=False):
    ### ----- Read data ----- ###
    # Read main data
    data = np.loadtxt((folder + 'data.txt'), skiprows=1)
    Nx = data[:, 0].astype(int)
    Ny = data[:, 1].astype(int)
    Nz = data[:, 2].astype(int)
    rel_err_stress = data[:, 3]
    moment = data[:, 4]
    num_stiff = data[:, 5]
    ana_stiff = data[:, 6]
    err_stiff = data[:, 7]

    # Read special data: stress error
    name = folder + f'data_stress_error_nb_grid_pts={Nx[0]}x'
    name += f'{Ny[0]}x{Nz[0]}.txt'
    err_stress_coarse = np.empty((6, Nx[0], Ny[0]))
    err_stress_coarse[0] = np.loadtxt(name, skiprows=1, max_rows=Nx[0])
    err_stress_coarse[1] = np.loadtxt(name, skiprows=2 + Nx[0],
                                      max_rows=Nx[0])
    err_stress_coarse[2] = np.loadtxt(name, skiprows=3 + 2 * Nx[0],
                                      max_rows=Nx[0])
    err_stress_coarse[3] = np.loadtxt(name, skiprows=4 + 3 * Nx[0],
                                      max_rows=Nx[0])
    err_stress_coarse[4] = np.loadtxt(name, skiprows=5 + 4 * Nx[0],
                                      max_rows=Nx[0])
    err_stress_coarse[5] = np.loadtxt(name, skiprows=6 + 5 * Nx[0],
                                      max_rows=Nx[0])

    name = folder + f'data_stress_error_nb_grid_pts={Nx[-1]}x'
    name += f'{Ny[-1]}x{Nz[-1]}.txt'
    err_stress_fine = np.empty((6, Nx[-1], Ny[-1]))
    err_stress_fine[0] = np.loadtxt(name, skiprows=1,
                                      max_rows=Nx[-1])
    err_stress_fine[1] = np.loadtxt(name, skiprows=2 + Nx[-1],
                                      max_rows=Nx[-1])
    err_stress_fine[2] = np.loadtxt(name, skiprows=3 + 2 * Nx[-1],
                                      max_rows=Nx[-1])
    err_stress_fine[3] = np.loadtxt(name, skiprows=4 + 3 * Nx[-1],
                                      max_rows=Nx[-1])
    err_stress_fine[4] = np.loadtxt(name, skiprows=5 + 4 * Nx[-1],
                                      max_rows=Nx[-1])
    err_stress_fine[5] = np.loadtxt(name, skiprows=6 + 5 * Nx[-1],
                                      max_rows=Nx[-1])

    # Read special data: moment
    name = folder + f'data_moment_nb_grid_pts={Nx[0]}x'
    name += f'{Ny[0]}x{Nz[0]}.txt'
    z_moment_coarse = np.loadtxt(name, skiprows=1)

    name = folder + f'data_moment_nb_grid_pts={Nx[-1]}x'
    name += f'{Ny[-1]}x{Nz[-1]}.txt'
    z_moment_fine = np.loadtxt(name, skiprows=1)

    ### ----- Plot values depending on discretization ----- ###
    # Relative error of stress
    fig, ax = plt.subplots()
    fig.suptitle('Norm of Error of stress')
    ax.set_xlabel('Nx=Ny')
    ax.set_ylabel('Relative err (%)')
    ax.plot(Nx, rel_err_stress, marker='x')
    name = folder + 'plot_rel_error_stress.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)

    # Relative error of stiffness
    fig, ax = plt.subplots()
    fig.suptitle('Error of torsion stiffness')
    ax.set_xlabel('Nx=Ny')
    ax.set_ylabel('Relative err (%)')
    ax.plot(Nx, err_stiff, marker='x')
    name = folder + 'plot_rel_error_stiff.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)

    # Stiffness
    fig, ax = plt.subplots()
    fig.suptitle('Torsion stiffness')
    ax.set_xlabel('Nx=Ny')
    ax.set_ylabel('Relative err (%)')
    ax.plot(Nx, num_stiff, marker='x', label='numerical')
    ax.plot(Nx, ana_stiff, marker='x', label='analytical')
    ax.legend()
    name = folder + 'plot_rel_error_stress.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)

    ### ----- Plot error of stress ----- ###
    stress_labels = ['xx', 'xy', 'xz', 'yy', 'yz', 'zz']
    stresses = []
    for i in range(6):
        # Prepare figure
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(('Error of stress_' + stress_labels[i]))
        axes[0].set_title(f'Nx=Ny={Nx[0]}')
        axes[1].set_title(f'Nx=Ny={Nx[-1]}')
        axes[0].set_aspect('equal')
        axes[1].set_aspect('equal')

        # Plot err coarse
        x = np.linspace(0, lengths[0], Nx[0]+1)
        y = np.linspace(0, lengths[1], Ny[0]+1)
        im = axes[0].pcolormesh(x, y, err_stress_coarse[i].T, rasterized=True)
        fig.colorbar(im, ax=axes[0])

        # Plot err fine
        x = np.linspace(0, lengths[0], Nx[-1]+1)
        y = np.linspace(0, lengths[1], Ny[-1]+1)
        im = axes[1].pcolormesh(x, y, err_stress_fine[i].T, rasterized=True)
        fig.colorbar(im, ax=axes[1])

        name = folder + 'plot_error_stress_' + stress_labels[i] + '.pdf'
        fig.savefig(name, bbox_inches='tight')
        plt.close(fig)

    ### ----- Plot moment depending on z ----- ###
    fig, ax = plt.subplots()
    fig.suptitle('Moment')
    ax.set_xlabel('z')
    ax.set_ylabel('Moment')
    z = [z_moment_coarse[0, 0], z_moment_coarse[0, -1]]
    moment_ana = ana_stiff * angle_per_length
    ax.plot(z, [moment_ana, moment_ana], label='analytical',
            linestyle = ':', color='black')
    ax.plot(z, [moment[0], moment[0]],
            label=f'Nx=Ny={Nx[0]} (aver.)', linestyle = '--', color='blue')
    ax.plot(z_moment_coarse[0], z_moment_coarse[1],
            label=f'Nx=Ny={Nx[0]}', linestyle = '-', color='blue')
    ax.plot(z, [moment[-1], moment[-1]],
            label=f'Nx=Ny={Nx[-1]} (aver.)', linestyle = '--', color='red')
    ax.plot(z_moment_fine[0], z_moment_fine[1],
            label=f'Nx=Ny={Nx[-1]}', linestyle = '-', color='red')
    ax.legend()
    name = folder + 'plot_moment.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)

    ### ----- Plot error of displacement if possible ----- ###
    if displ:
        # Read displacement
        data = np.loadtxt((folder + 'data_displacement.txt'), skiprows=1)
        Nx = data[:, 0].astype(int)
        Ny = data[:, 1].astype(int)
        Nz = data[:, 2].astype(int)
        rel_err_stress = data[:, 3]

        # Relative error of displacement
        fig, ax = plt.subplots()
        fig.suptitle('Error of displacement')
        ax.set_xlabel('Nx=Ny')
        ax.set_ylabel('Relative err (%)')
        ax.plot(Nx, err_stiff, marker='x')
        name = folder + 'plot_rel_error_displ.pdf'
        fig.savefig(name, bbox_inches='tight')
        plt.close(fig)


def calculation():
    """ Test a cylinder under torsion by comparison
        with the analytical solution. Parallel calculation.
    """
    ### ----- Parameter definitions ----- ###
    # Geometry
    lengths = [1, 1, 10]
    dim = len(lengths)
    radius = 0.4

    # Discretization
    #Nxy_list = [20, 30]
    Nxy_list = [30, 50, 70, 90]
    Nz = 50
    gradient, weights = µ.linear_finite_elements.gradient_3d_5tet

    # Material
    Young = 100
    Poisson = 0

    # Loading
    twist = 0.05
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
    fft = 'mpi' # Parallel fft

    # Folder for saving
    folder = f'results/cylinder/method1_twist={twist}/'

    ### ----- Prepare saving ----- ###
    if MPI.COMM_WORLD.rank == 0:
        # Create or clear folder
        if not os.path.exists(folder):
            os.makedirs(folder)
        else:
            shutil.rmtree(folder)
            os.makedirs(folder)

        # Create file for saving data
        name = folder + 'data.txt'
        with open(name, 'w') as f:
            title = 'nb_grid_pts   rel_norm_of_error_of_stress(%)'
            title += '   moment   numerical_torsion_stiffness   analytical_'
            title += f'torsion_stiffness   error_torsional_stiffness(%)'
            print(title, file=f)

        # Copy this file into the folder
        helper = 'cp torsion_cylinder.py ' + folder
        os.system(helper)

    if MPI.COMM_WORLD.size == 1:
        name2 = folder + 'data_displacement.txt'
        with open(name2, 'w') as f:
            title = 'nb_grid_pts   rel_norm_of_error_of_displ(%)'
            print(title, file=f)

    ### ----- Calculations ----- ###
    for ind_N, Nxy in enumerate(Nxy_list):
        nb_grid_pts = [Nxy, Nxy, Nz]
        if MPI.COMM_WORLD.rank == 0:
            message = f'Test {ind_N+1} of {len(Nxy_list)}: '
            message += f'Nxy = {Nxy}'
            print(message)

        ### ----- muSpectre calculation ----- ###
        # Geometry
        mask = cylinder(nb_grid_pts, lengths, radius)
        hx = lengths[0] / nb_grid_pts[0]
        hy = lengths[1] / nb_grid_pts[1]
        hz = lengths[2] / nb_grid_pts[2]

        # Initialize muSpectre cell
        cell = µ.Cell(nb_grid_pts, lengths, formulation, gradient,
                      weights=weights, fft=fft, communicator=MPI.COMM_WORLD)
        mask = mask[cell.fft_engine.subdomain_slices]
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
        eigen_class = EigenStrain(cell.pixels, twist, lengths, nb_grid_pts,
                                  x_rot_axis, y_rot_axis)

        # Solving
        res = µ.solvers.newton_cg(cell, delta_F, solver, newton_tol, equil_tol,
                                  verbose, μ.solvers.IsStrainInitialised.No,
                                  µ.StoreNativeStress.No, eigen_class.eigen_strain_func)
        shape = (dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts)
        stress_num = res.stress.reshape(shape, order='F')
        strain_num = res.grad.reshape(shape, order='F')

        ### ----- Further numerial calculations ----- ###
        # Coordinates of the centers of the 5 tetradras
        x = np.arange(nb_grid_pts[0]) * hx # x-coordinate of voxel
        y = np.arange(nb_grid_pts[1]) * hy # y-coordinate of voxel
        z = np.arange(nb_grid_pts[2]) * hz # z-coordinate of voxel
        x = x[cell.fft_engine.subdomain_slices[0]]
        y = y[cell.fft_engine.subdomain_slices[1]]
        z = z[cell.fft_engine.subdomain_slices[2]]

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
        #print(cell.fft_engine.subdomain_slices)
        #X = X[cell.fft_engine.subdomain_slices]
        #Y = Y[cell.fft_engine.subdomain_slices]

        # Calculate numerical moment
        helper = - stress_num[0, 2] * (Y - y_rot_axis)
        helper += stress_num[1, 2] * (X - x_rot_axis)
        moment = hx * hy * hz / 6 * np.sum(helper) # Average moment
        moment += hx * hy * hz / 6 * np.sum(helper[0])
        moment = Reduction(MPI.COMM_WORLD).sum(moment) / lengths[2]

        # Calculate detailed numerical moment in two cases
        if ((ind_N == 0) or (ind_N == len(Nxy_list)-1)) and\
           (MPI.COMM_WORLD.rank == 0):
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

        # Calculate numerical stiffness
        angle = np.arctan(twist * lengths[2])
        angle_per_length = angle / lengths[2]
        stiffness = moment / angle_per_length # stiffness = moment / twist

        ### ----- Analytical results ----- ###
        mask = mask.reshape(cell.nb_subdomain_grid_pts, order='F')
        # Analytical stress
        mu = Young / 2 / (1 + Poisson) # Shear modulus
        stress_ana = np.zeros(shape)
        stress_ana[0, 2] = - angle * mu * (Y - y_rot_axis)
        stress_ana[0, 2] *= mask[None, :, :, :] # No stress in void
        stress_ana[1, 2] = angle * mu * (X - x_rot_axis)
        stress_ana[1, 2] *= mask[None, :, :, :]
        stress_ana[2, 0] = stress_ana[0, 2]
        stress_ana[2, 1] = stress_ana[1, 2]

        # Moment and stiffness
        stiffness_ana = mu * np.pi * radius ** 4 / 2
        moment_ana = stiffness_ana * angle_per_length

        ### ----- Save results ----- ###
        # Stress error
        error_stress = stress_ana - stress_num
        norm_stress_err = np.linalg.norm(error_stress) / np.linalg.norm(stress_ana) * 100
        if (MPI.COMM_WORLD.rank == 0) and ((ind_N == 0) or (ind_N == len(Nxy_list)-1)):
            f = folder + f'data_stress_error_nb_grid_pts={nb_grid_pts[0]}x'
            f += f'{nb_grid_pts[1]}x{nb_grid_pts[2]}.txt'
            # Average stress over quad pts
            error_stress = error_stress[:, :, :, :, :, 0]
            error_stress = np.average(error_stress, axis=2,
                                      weights=[1/3, 1/6, 1/6, 1/6, 1/6])
            with open(f, 'w') as f:
                print('stress_00', file=f)
                np.savetxt(f, error_stress[0, 0]) #, newline=' ')
                print('stress_01', file=f)
                np.savetxt(f, error_stress[0, 1]) #, newline=' ')
                print('stress_02', file=f)
                np.savetxt(f, error_stress[0, 2]) #, newline=' ')
                print('stress_11', file=f)
                np.savetxt(f, error_stress[1, 1]) #, newline=' ')
                print('stress_12', file=f)
                np.savetxt(f, error_stress[1, 2]) #, newline=' ')
                print('stress_22', file=f)
                np.savetxt(f, error_stress[2, 2]) #, newline=' ')

        # Error stiffness
        error_stiff = abs((stiffness_ana - stiffness) / stiffness_ana) * 100

        # Save results in file
        if MPI.COMM_WORLD.rank == 0:
            with open(name, 'a') as f:
                np.savetxt(f, nb_grid_pts, newline=' ')
                np.savetxt(f, [norm_stress_err, moment,
                               stiffness, stiffness_ana, error_stiff], newline=' ')
                print('', file=f)

        # Save detailed informations of moment
        if (MPI.COMM_WORLD.rank == 0) and ((ind_N == 0) or (ind_N == len(Nxy_list)-1)):
            f = folder + f'data_moment_nb_grid_pts={nb_grid_pts[0]}x'
            f += f'{nb_grid_pts[1]}x{nb_grid_pts[2]}.txt'
            with open(f, 'w') as f:
                print('Contains: z   moment', file=f)
                np.savetxt(f, z, newline=' ')
                print('', file=f)
                np.savetxt(f, moment_of_z, newline=' ')

        # Print
        if MPI.COMM_WORLD.rank == 0:
            print(f'  stiff_num = {stiffness}')
            print(f'  stiff_ana = {stiffness_ana}')
            print(f'  error stiff (%) = {error_stiff}')

        ### ----- Displacement ----- ###
        # Can only be calculated in serial
        if MPI.COMM_WORLD.size == 1:
            x = np.linspace(0, lengths[0], nb_grid_pts[0]+1, endpoint=True)
            y = np.linspace(0, lengths[1], nb_grid_pts[1]+1, endpoint=True)
            z = np.linspace(0, lengths[2], nb_grid_pts[2]+1, endpoint=True)

            # Calculate numerical displacement
            F0=np.eye(3)
            strain_no_eigen = strain_num.copy()
            eigen_class.remove_eigen_strain_func(strain_no_eigen)
            [x_0, y_0, z_0], [x_displ, y_displ, z_displ] \
                = get_complemented_positions("0d", cell, strain_array=strain_no_eigen,
                                             F0=F0,  periodically_complemented=True)
            helper = - twist * np.einsum('i,j->ij', y-y_rot_axis, z)
            x_displ += helper[None, :, :]
            helper = twist * np.einsum('i,j->ij', x-x_rot_axis, z)
            y_displ += helper[:, None, :]
            pos = np.asarray([x_0, y_0, z_0])
            displ = np.asarray([x_displ, y_displ, z_displ])

            # Analytical displacement
            displ_ana = np.zeros(displ.shape)
            helper = - twist * np.einsum('i,j->ij', y-y_rot_axis, z)
            displ_ana[0] += helper[None, :, :]
            helper = twist * np.einsum('i,j->ij', x-x_rot_axis, z)
            displ_ana[1] += helper[:, None, :]

            # Error
            err = np.linalg.norm(displ_ana - displ)
            err = err / np.linalg.norm(displ_ana) * 100

            # Save in file
            with open(name2, 'a') as f:
                np.savetxt(f, nb_grid_pts, newline=' ')
                np.savetxt(f, [err], newline=' ')
                print('', file=f)

    ### ----- Plot data ----- ###
    if MPI.COMM_WORLD.size == 1:
        plotting(folder, lengths, angle_per_length, displ=True)
    else:
        plotting(folder, lengths, angle_per_length)

    # Save deformed geometry if calculation is serial
    if MPI.COMM_WORLD.size == 1:
        cell_data = {}
        material = np.stack((mask, mask, mask, mask, mask), axis=0)
        material = material.flatten(order='F')
        material = material.reshape((1, -1))
        stress = stress_num.reshape((3, 3, -1), order='F').T.swapaxes(1, 2)
        stress = stress.reshape((1, -1, 3, 3)).copy()
        strain = strain_num.reshape((3, 3, -1), order='F').T.swapaxes(1, 2)
        strain = strain.reshape((1, -1, 3, 3)).copy()
        cell_data = {"material": material, "stress_field": stress,
                     "strain_field": strain}
        point_data = {"displ": displ.reshape((3, -1), order='F').T}
        name = folder + f'deformation_nb_grid_pts={nb_grid_pts[0]}x'
        name += f'{nb_grid_pts[1]}x{nb_grid_pts[2]}.xdmf'
        µ.linear_finite_elements.write_3d(name, cell, cell_data=cell_data,
                                          point_data=point_data,
                                          F0=F0, displacement_field=True)



def calculation_serial():
    """ Test a cylinder under torsion by comparison
        with the analytical solution. Serial calculation.
        Serves for testing the calculation.
    """
    ### ----- Parameter definitions ----- ###
    # Geometry
    lengths = [1, 1, 10]
    radius = 0.4

    # Discretization
    Nxy_list = [20, 30]
    #Nxy_list = [30, 50, 70, 90, 110]
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
    folder = 'results/cylinder/data/'

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

        print(f'  stiff_num = {stiff}')
        print(f'  stiff_ana = {stiffness_ana}')
        print(f'  error stiff (%) = {rel_err_stiff[ind_N]}')
    a = b

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
    calculation()

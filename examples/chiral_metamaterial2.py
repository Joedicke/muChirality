"""
@file   chiral_metamaterial2.py

@author Indre  Joedicke <indre.joedicke@imtek.uni-freiburg.de>

@date   24 Nov 2023

@brief  Test a chiral metameterial

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
sys.path.insert(0, os.path.join(os.getcwd(), "../../muspectre/builddir/language_bindings/libmugrid/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../../muspectre/builddir/language_bindings/libmufft/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../../muspectre/builddir/language_bindings/python"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from time import time

import muSpectre as µ
from NuMPI import MPI
from NuMPI.Tools import Reduction
from NuMPI.IO import save_npy

from muChirality.EigenStrainTorsion import EigenStrain
from muChirality.CalculationsTorsion import calculations
from muSpectre.gradient_integration import get_complemented_positions
import muChirality.Geometries as geo

def plot_with_paper(folder):
    ### ----- Data ----- ###
    # Read own data
    data_file = folder + 'data.txt'
    data = np.loadtxt(data_file, skiprows=1)
    N_uc_list = data[:, 0]
    E_eff_z = data[:, 2]
    twist_per_strain = data[:, 4]

    # Paper (approximative)
    paper_N_uc = [1, 2, 3, 4, 5]
    paper_twist_exp = [2, 2, 1.5, 1.1, 0.9]
    paper_twist_num = [2, 2, 1.6, 1.4, 1.2]
    paper_E_exp = [5, 22, 25, 30, 35]
    paper_E_num = [5, 20, 24, 28, 29]

    ### ----- Plot ----- ###
    # Prepare figure
    fig, ax = plt.subplots(1, 2)
    ax[0].set_xlabel('Nb_unit_cells in RVE (one direction)')
    ax[1].set_xlabel('Nb_unit_cells in RVE (one direction)')
    ax[0].set_ylabel('Twist/strain (degree/%)')
    ax[1].set_ylabel('Eff E-Modul (MPa)')

    # Plot twist / strain
    ax[0].plot(paper_N_uc, paper_twist_exp, color='red', marker='o', label='paper (exp)')
    ax[0].plot(paper_N_uc, paper_twist_num, color='red', marker='x', label='paper (num)')
    ax[0].plot(N_uc_list, twist_per_strain, color='blue', marker='x', label='muSpectre')
    ax[0].legend()

    # Plot E-modul
    ax[1].plot(paper_N_uc, paper_E_exp, color='red', marker='o', label='paper (exp)')
    ax[1].plot(paper_N_uc, paper_E_num, color='red', marker='x', label='paper (num)')
    ax[1].plot(N_uc_list, E_eff_z, color='blue', marker='x', label='muSpectre')
    ax[1].legend()

    # Save plot
    name = folder + 'plot_data.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)

def plot_comp_cylinder(folder):
    # Read data
    name = folder + 'data.txt'
    twists = np.loadtxt(name, skiprows=1, max_rows=1)
    forces_cyl = np.loadtxt(name, skiprows=3, max_rows=1)
    forces_chiral = np.loadtxt(name, skiprows=5, max_rows=1)

    # Prepare figure
    fig, ax = plt.subplots()
    ax.set_xlabel('Twist')
    ax.set_ylabel('Force in z-direction')

    # Plot
    ax.plot(twists, forces_cyl, color='red', marker='x', label='cylinder')
    ax.plot(twists, forces_chiral, color='blue', marker='x', label='chiral mat')
    ax.legend()

    # Save plot
    name = folder + 'plot_data.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)

###################################################################################################
###################################################################################################
###################################################################################################
def calculation_mult_unit_cells():
    ### ----- Parameter definitions ----- ###
    t = time()

    # Geometry
    a = 0.5 # in mm
    thickness = 0.06 * a
    radius_out = 0.4 * a
    radius_inn = 0.34 * a
    angle_mat = np.pi * 35 / 180

    # Nb of unit cells in RVE
    N_uc_list = [1, 3]
    restart = False
    plates = False
    Nz_changes = False
    Nz = 1

    # Discretization
    dim = 3
    N = 30
    nb_grid_pts_uc = [N, N, N]
    gradient, weights = µ.linear_finite_elements.gradient_3d_5tet

    if MPI.COMM_WORLD.rank == 0:
        print(f'--- Parameters ---')
        print(f'MPI size = {MPI.COMM_WORLD.size}')
        helper = f'nb_grid_pts_uc = {nb_grid_pts_uc[0]}x{nb_grid_pts_uc[1]}'
        helper += f'x{nb_grid_pts_uc[2]}'
        print(helper)
        print('List nb_unit_cells =', N_uc_list)
        print('Restarted:', restart)
        print('With plates:', plates)
        if Nz_changes:
            print('Nz changes.')
        else:
            print(f'Nz = {Nz}')
        print()

    # Material
    Young = 2600 # in MPa
    Poisson = 0.4

    # Loading
    twist = 0.05 # in 1/mm
    #x_rot_axis = lengths[0] / 2
    #y_rot_axis = lengths[1] / 2
    delta_eps1 = np.zeros((3, 3))
    delta_eps2 = np.zeros((dim, dim))
    delta_eps2[2, 2] = 0.01

    # Formulation
    formulation = µ.Formulation.small_strain

    # muSpectre solver parameters
    newton_tol       = 1e-7
    cg_tol           = 1e-7 # tolerance for cg algo
    equil_tol        = 1e-7 # tolerance for equilibrium
    maxiter          = 10000
    verbose          = µ.Verbosity.Silent
    fft = 'mpi' # Parallel fft

    # For saving
    F0 = np.eye(3)
    folder = 'results_paper/chiral_Nuc_z={Nz}_Nxyz={N}/'
    if plates:
        folder += 'with_plates/'
    else:
        folder += 'mult_unit_cells/'
    if Nz_changes:
        folder += 'Nucz_changes_'
    else:
        folder += f'Nuc_z={Nz}_'
    folder += f'Nxyz={nb_grid_pts_uc[0]}_twist={twist}'
    if MPI.COMM_WORLD.size == 1:
        folder += '/'
    else:
        folder += '_parall/'
    name = folder + 'data.txt'

    ### ----- Prepare saving ----- ###
    if (MPI.COMM_WORLD.rank == 0) and (not restart):
        # Create or clear folder
        if not os.path.exists(folder):
            os.makedirs(folder)
        else:
            shutil.rmtree(folder)
            os.makedirs(folder)

        # Copy this file into the folder
        helper = 'cp chiral_metamaterial2.py ' + folder
        os.system(helper)

        # File  for saving data
        with open(name, 'w') as f:
            title = 'Number of unit cells in RVE    Stiffness in z direction    '
            title += 'Effective Youngs modulus in z direction    '
            title += 'Force_z (with rot)    Twist per strain (degree/%)'
            print(title, file=f)

    # Calculation
    for index, N_uc in enumerate(N_uc_list):
        t = time()
        ### ----- Define geometry ----- ###
        if Nz_changes:
            nb_unit_cells = [N_uc, N_uc, N_uc]
        else:
            nb_unit_cells = [N_uc, N_uc, Nz]
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
        nb_grid_pts = mask.shape

        ### ----- muSpectre cell initialization ----- ###
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

        ### ----- Calculate homogenized stiffness in z-direction  ----- ###
        # Solve muSpectre
        solver = µ.solvers.KrylovSolverCG(cell, cg_tol, maxiter, verbose)
        result = µ.solvers.newton_cg(cell, delta_eps2, solver, newton_tol,
                                     equil_tol, verbose)
        shape = (dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts)
        stress = result.stress.reshape(shape, order='F')

        # Force in z-direction
        hx = lengths[0] / nb_grid_pts[0]
        hy = lengths[1] / nb_grid_pts[1]
        hz = lengths[2] / nb_grid_pts[2]
        force_z = hx * hy * hz / 6 * np.sum(stress[2, 2])
        force_z += hx * hy * hz / 6 * np.sum(stress[2, 2, 0])
        force_z = Reduction(MPI.COMM_WORLD).sum(force_z) / lengths[2]

        # Stiffness in z-direction
        stiff_z = force_z / delta_eps2[2, 2]
        E_eff_z = stiff_z / a / a
        if MPI.COMM_WORLD.rank == 0:
            print(f'nb_unit_cells = {N_uc}')
            print(f'force_z = {force_z}')
            print(f'stiff_z = {stiff_z}')
            print(f'E_eff_z = {E_eff_z}')

        ### ----- Calculate Twist per Strain ----- ###
        solver = µ.solvers.KrylovSolverCG(cell, cg_tol, maxiter, verbose)

        # Initialize Eigen class
        eigen_class = EigenStrain(cell.pixels, twist, lengths, nb_grid_pts,
                                  lengths[0]/2, lengths[1]/2)

        # Solve muSpectre
        res = µ.solvers.newton_cg(cell, delta_eps1, solver, newton_tol, equil_tol,
                                  verbose, μ.solvers.IsStrainInitialised.No,
                                  µ.StoreNativeStress.No, eigen_class.eigen_strain_func)
        stress = res.stress.reshape(shape, order='F')
        strain = res.grad.reshape(shape, order='F')

        # Force in z-direction
        force = hx * hy * hz / 6 * np.sum(stress[2, 2])
        force += hx * hy * hz / 6 * np.sum(stress[2, 2, 0])
        force = Reduction(MPI.COMM_WORLD).sum(force) / lengths[2]

        # Comparison with paper
        twist_in_degree = np.arctan(twist * lengths[2] * 2 * N_uc) / np.pi * 180
        strain_zz_in_percent = force / stiff_z * 100

        twist_per_strain = twist_in_degree / strain_zz_in_percent

        if MPI.COMM_WORLD.rank == 0:
            print('Twist (degree):', twist_in_degree)
            print('strain (%):', strain_zz_in_percent)
            print('Force:', force)
            print('Twist/strain (degree/%):', twist_per_strain)
            print(f'Finished calculation {index+1} of {len(N_uc_list)}')
            t = (time() - t) / 60
            print(f'Time for calculation = {t:.2} min')

        ### ----- Save results ----- ###
        t = time()
        # Save effective parameters
        if MPI.COMM_WORLD.rank == 0:
            with open(name, 'a') as f:
                np.savetxt(f, [N_uc, stiff_z, E_eff_z, force, twist_per_strain], newline=' ')
                print('', file=f)

        # Save strain
        folder_strain = folder + f'N_uc={N_uc}_strains/'
        if (MPI.COMM_WORLD.rank == 0):
            # Create folder
            if not os.path.exists(folder_strain):
                os.makedirs(folder_strain)
        for i_quad in range(cell.nb_quad_pts):
            name_strain = folder_strain + f'quad_pt_{i_quad}_entry_'
            save_npy((name_strain + '00.npy'), strain[0, 0, i_quad], tuple(cell.subdomain_locations),
                     tuple(cell.nb_domain_grid_pts), MPI.COMM_WORLD)
            save_npy((name_strain + '01.npy'), strain[0, 1, i_quad], tuple(cell.subdomain_locations),
                     tuple(cell.nb_domain_grid_pts), MPI.COMM_WORLD)
            save_npy((name_strain + '02.npy'), strain[0, 2, i_quad], tuple(cell.subdomain_locations),
                     tuple(cell.nb_domain_grid_pts), MPI.COMM_WORLD)
            save_npy((name_strain + '11.npy'), strain[1, 1, i_quad], tuple(cell.subdomain_locations),
                     tuple(cell.nb_domain_grid_pts), MPI.COMM_WORLD)
            save_npy((name_strain + '12.npy'), strain[1, 2, i_quad], tuple(cell.subdomain_locations),
                     tuple(cell.nb_domain_grid_pts), MPI.COMM_WORLD)
            save_npy((name_strain + '22.npy'), strain[2, 2, i_quad], tuple(cell.subdomain_locations),
                     tuple(cell.nb_domain_grid_pts), MPI.COMM_WORLD)

        folder_stress = folder + f'N_uc={N_uc}_stresses/'
        if (MPI.COMM_WORLD.rank == 0):
            # Create folder
            if not os.path.exists(folder_stress):
                os.makedirs(folder_stress)
        for i_quad in range(cell.nb_quad_pts):
            name_stress = folder_stress + f'quad_pt_{i_quad}_entry_'
            save_npy((name_stress + '00.npy'), stress[0, 0, i_quad], tuple(cell.subdomain_locations),
                     tuple(cell.nb_domain_grid_pts), MPI.COMM_WORLD)
            save_npy((name_stress + '01.npy'), stress[0, 1, i_quad], tuple(cell.subdomain_locations),
                     tuple(cell.nb_domain_grid_pts), MPI.COMM_WORLD)
            save_npy((name_stress + '02.npy'), stress[0, 2, i_quad], tuple(cell.subdomain_locations),
                     tuple(cell.nb_domain_grid_pts), MPI.COMM_WORLD)
            save_npy((name_stress + '11.npy'), stress[1, 1, i_quad], tuple(cell.subdomain_locations),
                     tuple(cell.nb_domain_grid_pts), MPI.COMM_WORLD)
            save_npy((name_stress + '12.npy'), stress[1, 2, i_quad], tuple(cell.subdomain_locations),
                     tuple(cell.nb_domain_grid_pts), MPI.COMM_WORLD)
            save_npy((name_stress + '22.npy'), stress[2, 2, i_quad], tuple(cell.subdomain_locations),
                     tuple(cell.nb_domain_grid_pts), MPI.COMM_WORLD)

        if (MPI.COMM_WORLD.rank == 0):
            t = (time() - t) / 60
            print(f'Time for saving = {t:.2} min')
            print()

        # Save deformed state only if the calculation is not parallel
        save_displacement = False
        if (MPI.COMM_WORLD.size == 1) and save_displacement:
            # Calculate position + displacement
            x = np.linspace(0, lengths[0], nb_grid_pts[0]+1, endpoint=True)
            y = np.linspace(0, lengths[1], nb_grid_pts[1]+1, endpoint=True)
            z = np.linspace(0, lengths[2], nb_grid_pts[2]+1, endpoint=True)
            strain_no_eigen = strain.copy()
            eigen_class.remove_eigen_strain_func(strain_no_eigen)
            [x_0, y_0, z_0], [x_displ, y_displ, z_displ] \
                = get_complemented_positions("0d", cell, strain_array=strain_no_eigen, F0=F0,
                                             periodically_complemented=True)
            x_rot_axis = lengths[0] / 2
            y_rot_axis = lengths[1] / 2
            helper = - twist * np.einsum('i,j->ij', y-y_rot_axis, z)
            x_displ += helper[None, :, :]
            helper = twist * np.einsum('i,j->ij', x-x_rot_axis, z)
            y_displ += helper[:, None, :]
            pos = np.asarray([x_0, y_0, z_0])
            displ = np.asarray([x_displ, y_displ, z_displ])

            # Save as xdmf-file
            cell_data = {}
            material = np.stack((mask, mask, mask, mask, mask), axis=0)
            material = material.flatten(order='F')
            material = material.reshape((1, -1))
            stress = stress.reshape((3, 3, -1), order='F').T.swapaxes(1, 2)
            stress = stress.reshape((1, -1, 3, 3)).copy()
            strain = strain.reshape((3, 3, -1), order='F').T.swapaxes(1, 2)
            strain = strain.reshape((1, -1, 3, 3)).copy()
            cell_data = {"material": material, "stress_field": stress, "strain_field": strain}
            point_data = {"displ": displ.reshape((3, -1), order='F').T}
            name2 = folder + f'Nuc={N_uc}_deformed_geometry.xdmf'
            µ.linear_finite_elements.write_3d(name2, cell, cell_data=cell_data, point_data=point_data,
                                              F0=F0, displacement_field=False)

    ### ----- Plot results ----- ###
    if MPI.COMM_WORLD.rank == 0:
        plot_with_paper(folder)

###################################################################################################
###################################################################################################
###################################################################################################
def calculation_with_cylinder():
    t = time()
    ### ----- Parameter definitions ----- ###
    # Geometry
    a = 1
    thickness = 0.06
    radius_out = 0.4
    radius_inn = radius_out - thickness
    angle_mat = np.pi * 35 / 180
    lengths_cyl = [1, 1, a]

    # Discretization
    Nxyz = 30
    nb_grid_pts = [Nxyz, Nxyz, Nxyz]
    gradient, weights = µ.linear_finite_elements.gradient_3d_5tet

    # Material
    Young = 100
    Poisson = 0

    # Loading
    twists = [-0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, -0.2]
    #twists = [-0.1, 0.1, 0.2]
    delta_F = np.zeros((3, 3))

    # Formulation
    formulation = µ.Formulation.small_strain

    # muSpectre solver parameters
    newton_tol       = 1e-7
    cg_tol           = 1e-7 # tolerance for cg algo
    equil_tol        = 1e-7 # tolerance for equilibrium
    maxiter          = 10000
    verbose          = µ.Verbosity.Silent
    fft = 'mpi'

    # For saving
    folder = f'results/chiral2/comp_cylinder_Nxyz={Nxyz}/'
    F0 = np.eye(3)

    ### ----- Prepare saving ----- ###
    if (MPI.COMM_WORLD.rank == 0):
        # Create or clear folder
        if not os.path.exists(folder):
            os.makedirs(folder)
        else:
            shutil.rmtree(folder)
            os.makedirs(folder)

        # Copy this file into the folder
        helper = 'cp chiral_metamaterial2_parallel.py ' + folder
        os.system(helper)

    ### ----- Calculations ----- ###
    forces_chiral = np.empty(len(twists))
    forces_cyl = np.empty(len(twists))

    for i_twist, twist in enumerate(twists):
        if (MPI.COMM_WORLD.rank == 0):
            message = f'Test {i_twist+1} of {len(twists)}'
            print(message)

        ### ----- Calculation (cylinder) ----- ###
        # Define geometry
        mask = geo.cylinder(nb_grid_pts, lengths_cyl, radius_out)
        hx = lengths_cyl[0] / nb_grid_pts[0]
        hy = lengths_cyl[1] / nb_grid_pts[1]
        hz = lengths_cyl[2] / nb_grid_pts[2]
        x_rot_axis = lengths_cyl[0] / 2
        y_rot_axis = lengths_cyl[1] / 2

        # muSpectre cell initialization
        cell = µ.Cell(nb_grid_pts, lengths_cyl, formulation, gradient,
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
        eigen_class = EigenStrain(cell.pixels, twist, lengths_cyl, nb_grid_pts,
                                  x_rot_axis, y_rot_axis)

        # Solving
        res = µ.solvers.newton_cg(cell, delta_F, solver, newton_tol, equil_tol,
                                  verbose, μ.solvers.IsStrainInitialised.No,
                                  µ.StoreNativeStress.No, eigen_class.eigen_strain_func)
        shape = (3, 3, 5, *nb_grid_pts)
        stress = res.stress.reshape(shape, order='F')
        strain = res.grad.reshape(shape, order='F')

        # Calculate force in z-direction
        force_z = hx * hy * hz / 6 * np.sum(stress[2, 2])
        force_z += hx * hy * hz / 6 * np.sum(stress[2, 2, 0])
        force_z = force_z / lengths_cyl[2]
        forces_cyl[i_twist] = force_z

        # Delete parameters
        del cell
        del mat
        del vac
        del mask
        del solver
        del eigen_class
        del res
        del stress
        del strain

        ### ----- Calculation (chiral) ----- ###
        # Define geometry
        mask, lengths = geo.chiral_metamaterial_2(nb_grid_pts, a, radius_out,
                                                  radius_inn, thickness,
                                                  alpha=angle_mat)
        hx = lengths[0] / nb_grid_pts[0]
        hy = lengths[1] / nb_grid_pts[1]
        hz = lengths[2] / nb_grid_pts[2]
        x_rot_axis = lengths[0] / 2
        y_rot_axis = lengths[1] / 2

        # muSpectre cell initialization
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
        shape = (3, 3, 5, *nb_grid_pts)
        stress = res.stress.reshape(shape, order='F')
        strain = res.grad.reshape(shape, order='F')

        # Calculate force in z-direction
        force_z = hx * hy * hz / 6 * np.sum(stress[2, 2])
        force_z += hx * hy * hz / 6 * np.sum(stress[2, 2, 0])
        force_z = force_z / lengths[2]
        forces_chiral[i_twist] = force_z

        # Save one state for paraview (Only for serial calculations possible)
        if (i_twist is (len(twists)-1)) and (MPI.COMM_WORLD.size == 1):
            # Calculate position + displacement
            x = np.linspace(0, lengths[0], nb_grid_pts[0]+1, endpoint=True)
            y = np.linspace(0, lengths[1], nb_grid_pts[1]+1, endpoint=True)
            z = np.linspace(0, lengths[2], nb_grid_pts[2]+1, endpoint=True)
            strain_no_eigen = strain.copy()
            eigen_class.remove_eigen_strain_func(strain_no_eigen)
            [x_0, y_0, z_0], [x_displ, y_displ, z_displ] \
                = get_complemented_positions("0d", cell, strain_array=strain_no_eigen, F0=F0,
                                             periodically_complemented=True)
            x_rot_axis = lengths[0] / 2
            y_rot_axis = lengths[1] / 2
            helper = - twist * np.einsum('i,j->ij', y-y_rot_axis, z)
            x_displ += helper[None, :, :]
            helper = twist * np.einsum('i,j->ij', x-x_rot_axis, z)
            y_displ += helper[:, None, :]
            pos = np.asarray([x_0, y_0, z_0])
            displ = np.asarray([x_displ, y_displ, z_displ])

            cell_data = {}
            material = np.stack((mask, mask, mask, mask, mask), axis=0)
            material = material.flatten(order='F')
            material = material.reshape((1, -1))
            stress = stress.reshape((3, 3, -1), order='F').T.swapaxes(1, 2)
            stress = stress.reshape((1, -1, 3, 3)).copy()
            strain = strain.reshape((3, 3, -1), order='F').T.swapaxes(1, 2)
            strain = strain.reshape((1, -1, 3, 3)).copy()
            cell_data = {"material": material, "stress_field": stress, "strain_field": strain}
            point_data = {"displ": displ.reshape((3, -1), order='F').T}
            name = folder + f'twist={twist}_deformed_geometry.xdmf'
            µ.linear_finite_elements.write_3d(name, cell, cell_data=cell_data, point_data=point_data,
                                              F0=F0, displacement_field=True)

        # Delete Parameters
        del cell
        del mat
        del vac
        del mask
        del solver
        del eigen_class
        del res
        del stress
        del strain

    ### ----- Save results ----- ###
    # Save results
    name = folder + 'data.txt'
    if MPI.COMM_WORLD.rank == 0:
        with open(name, 'w') as f:
            print('Twists', file=f)
            np.savetxt(f, twists, newline=' ')
            print('', file=f)
            print('Force in z-direction (cylinder)', file=f)
            np.savetxt(f, forces_cyl, newline=' ')
            print('', file=f)
            print('Force in z-direction (chiral mat)', file=f)
            np.savetxt(f, forces_chiral, newline=' ')

    # Plot results
    if MPI.COMM_WORLD.rank == 0:
        plot_comp_cylinder(folder)

def check_geometries():
    ### ----- Parameter definitions ----- ###
    # Geometry
    a = 1
    thickness = 0.06
    radius_out = 0.4
    radius_inn = radius_out - thickness
    angle_mat = np.pi * 35 / 180
    lengths_cyl = [1, 1, a]
    N_uc = [2, 2, 3]

    # Discretization
    Nxyz = 30
    nb_grid_pts = [Nxyz, Nxyz, Nxyz]

    helper_title = f'{N_uc[0]}x{N_uc[1]}x{N_uc[2]} unit cells: '

    ### ----- Define geometries ----- ###
    # Original geometry (one unit cell)
    mask, lengths =\
        geo.chiral_metamaterial_2(nb_grid_pts, a, radius_out,
                                  radius_inn, thickness,
                                  alpha=angle_mat)
    x = np.linspace(0, lengths[0], nb_grid_pts[0]+1, endpoint=True)
    y = np.linspace(0, lengths[1], nb_grid_pts[1]+1, endpoint=True)
    z = np.linspace(0, lengths[2], nb_grid_pts[2]+1, endpoint=True)

    # Mirrored geometry (one unit cell)
    mask_mir, lengths_mir =\
        geo.chiral_metamaterial_2_mirrored(nb_grid_pts, a, radius_out,
                                           radius_inn, thickness,
                                           alpha=angle_mat)
    x_mir = np.linspace(0, lengths_mir[0], nb_grid_pts[0]+1, endpoint=True)
    y_mir = np.linspace(0, lengths_mir[1], nb_grid_pts[1]+1, endpoint=True)
    z_mir = np.linspace(0, lengths_mir[2], nb_grid_pts[2]+1, endpoint=True)

    # Original geometry (1x1x1 unit cells)
    mask_1x1x1, lengths_1x1x1 =\
        geo.chiral_2_mult_unit_cell([1, 1, 1], nb_grid_pts, a,
                                            radius_out, radius_inn,
                                            thickness, alpha=angle_mat)
    x_1x1x1 = np.linspace(0, lengths_1x1x1[0], nb_grid_pts[0]+1, endpoint=True)
    y_1x1x1 = np.linspace(0, lengths_1x1x1[1], nb_grid_pts[1]+1, endpoint=True)
    z_1x1x1 = np.linspace(0, lengths_1x1x1[2], nb_grid_pts[2]+1, endpoint=True)

    # Mirrored geometry (1x1x1 unit cells)
    mask_1x1x1_mir, lengths_1x1x1_mir =\
        geo.chiral_2_mult_unit_cell_mirrored([1, 1, 1], nb_grid_pts, a,
                                             radius_out, radius_inn,
                                             thickness, alpha=angle_mat)
    x_1x1x1_mir = np.linspace(0, lengths_1x1x1_mir[0], nb_grid_pts[0]+1, endpoint=True)
    y_1x1x1_mir = np.linspace(0, lengths_1x1x1_mir[1], nb_grid_pts[1]+1, endpoint=True)
    z_1x1x1_mir = np.linspace(0, lengths_1x1x1_mir[2], nb_grid_pts[2]+1, endpoint=True)

    # Original geometry (N_uc unit cells)
    mask_mult, lengths_mult =\
        geo.chiral_2_mult_unit_cell(N_uc, nb_grid_pts, a,
                                            radius_out, radius_inn,
                                            thickness, alpha=angle_mat)
    nb_grid_pts_mult = mask_mult.shape
    x_mult = np.linspace(0, lengths_mult[0], nb_grid_pts_mult[0]+1, endpoint=True)
    y_mult = np.linspace(0, lengths_mult[1], nb_grid_pts_mult[1]+1, endpoint=True)
    z_mult = np.linspace(0, lengths_mult[2], nb_grid_pts_mult[2]+1, endpoint=True)

    # Mirrored geometry (N_uc unit cells)
    mask_mult_mir, lengths_mult_mir =\
        geo.chiral_2_mult_unit_cell_mirrored(N_uc, nb_grid_pts, a,
                                             radius_out, radius_inn,
                                             thickness, alpha=angle_mat)
    nb_grid_pts_mult = mask_mult_mir.shape
    x_mult_mir = np.linspace(0, lengths_mult_mir[0], nb_grid_pts_mult[0]+1, endpoint=True)
    y_mult_mir = np.linspace(0, lengths_mult_mir[1], nb_grid_pts_mult[1]+1, endpoint=True)
    z_mult_mir = np.linspace(0, lengths_mult_mir[2], nb_grid_pts_mult[2]+1, endpoint=True)

    ### ----- Plots: helper functions ----- ###
    def prepare_figure_2x2(title):
        fig, axes = plt.subplots(2, 2, figsize=(6, 6))
        fig.suptitle(title)
        axes[0, 0].set_title('Original (one unit cell)')
        axes[0, 1].set_title('Mirrored (one unit cell)')
        axes[1, 0].set_title('Original (mult unit cells)')
        axes[1, 1].set_title('Mirrored (mult unit cells)')
        axes[0, 0].set_aspect('equal')
        axes[0, 1].set_aspect('equal')
        axes[1, 0].set_aspect('equal')
        axes[1, 1].set_aspect('equal')
        return fig, axes

    def prepare_figure_1x2(title):
        fig, axes = plt.subplots(1, 2, figsize=(7, 4))
        fig.suptitle(title)
        axes[0].set_title('Original')
        axes[1].set_title('Mirrored')
        axes[0].set_aspect('equal')
        axes[1].set_aspect('equal')
        return fig, axes

    ### ----- Plot xy-plane (bottom) ----- ###
    i = 0
    # Only for checking pcolormesh
    #mask[0, Nxyz//2, i] = mask_mir[0, Nxyz//2, i] = 0.5
    #mask_1x1x1[0, Nxyz//2, i] = mask_1x1x1_mir[0, Nxyz//2, i] = 0.5

    title = f'One unit cell: xy-plane (i_z={i})'
    fig_one, axes = prepare_figure_2x2(title)
    axes[0, 0].pcolormesh(x, y, mask[:, :, i].T, shading='flat')
    axes[0, 1].pcolormesh(x_mir, y_mir, mask_mir[:, :, i].T, shading='flat')
    axes[1, 0].pcolormesh(x_1x1x1, y_1x1x1, mask_1x1x1[:, :, i].T, shading='flat')
    axes[1, 1].pcolormesh(x_1x1x1_mir, y_1x1x1_mir, mask_1x1x1_mir[:, :, i].T, shading='flat')

    title = helper_title + f'xy-plane (i_z={i})'
    fig_mult, axes = prepare_figure_1x2(title)
    axes[0].pcolormesh(x_mult, y_mult, mask_mult[:, :, i].T, shading='flat')
    axes[1].pcolormesh(x_mult_mir, y_mult_mir, mask_mult_mir[:, :, i].T, shading='flat')

    #plt.show()
    plt.close(fig_one)
    plt.close(fig_mult)

    ### ------ Plot xy-plane (top) ----- ###
    i = -1

    title = f'One unit cell: xy-plane (i_z={i})'
    fig_one, axes = prepare_figure_2x2(title)
    axes[0, 0].pcolormesh(x, y, mask[:, :, i].T, shading='flat')
    axes[0, 1].pcolormesh(x_mir, y_mir, mask_mir[:, :, i].T, shading='flat')
    axes[1, 0].pcolormesh(x_1x1x1, y_1x1x1, mask_1x1x1[:, :, i].T, shading='flat')
    axes[1, 1].pcolormesh(x_1x1x1_mir, y_1x1x1_mir, mask_1x1x1_mir[:, :, i].T, shading='flat')

    title = helper_title + f'xy-plane (i_z={i})'
    fig_mult, axes = prepare_figure_1x2(title)
    axes[0].pcolormesh(x_mult, y_mult, mask_mult[:, :, i].T, shading='flat')
    axes[1].pcolormesh(x_mult_mir, y_mult_mir, mask_mult_mir[:, :, i].T, shading='flat')

    #plt.show()
    plt.close(fig_one)
    plt.close(fig_mult)

    ### ----- Plot xz-plane (left) ------ ###
    i = 1
    # Only for checking pcolormesh
    #mask[Nxyz//2, i, 0] = mask_mir[Nxyz//2, i, 0] = 0.5
    #mask_1x1x1[Nxyz//2, i, 0] = mask_1x1x1_mir[Nxyz//2, i, 0] = 0.5

    title = f'One unit cell: xz-plane (i_y={i})'
    fig_one, axes = prepare_figure_2x2(title)
    axes[0, 0].pcolormesh(x, z, mask[:, i, :].T, shading='flat')
    axes[0, 1].pcolormesh(x_mir, z_mir, mask_mir[:, i, :].T, shading='flat')
    axes[1, 0].pcolormesh(x_1x1x1, z_1x1x1, mask_1x1x1[:, i, :].T, shading='flat')
    axes[1, 1].pcolormesh(x_1x1x1_mir, z_1x1x1_mir, mask_1x1x1_mir[:, i, :].T, shading='flat')

    title = helper_title + f'xz-plane (i_y={i})'
    fig_mult, axes = prepare_figure_1x2(title)
    axes[0].pcolormesh(x_mult, z_mult, mask_mult[:, i, :].T, shading='flat')
    axes[1].pcolormesh(x_mult_mir, z_mult_mir, mask_mult_mir[:, i, :].T, shading='flat')

    #plt.show()
    plt.close(fig_one)
    plt.close(fig_mult)

    ### ----- Plot xz-plane (right) ------ ###
    i = -2

    title = f'One unit cell: xz-plane (i_y={i})'
    fig_one, axes = prepare_figure_2x2(title)
    axes[0, 0].pcolormesh(x, z, mask[:, i, :].T, shading='flat')
    axes[0, 1].pcolormesh(x_mir, z_mir, mask_mir[:, i, :].T, shading='flat')
    axes[1, 0].pcolormesh(x_1x1x1, z_1x1x1, mask_1x1x1[:, i, :].T, shading='flat')
    axes[1, 1].pcolormesh(x_1x1x1_mir, z_1x1x1_mir, mask_1x1x1_mir[:, i, :].T, shading='flat')

    title = helper_title + f'xz-plane (i_y={i})'
    fig_mult, axes = prepare_figure_1x2(title)
    axes[0].pcolormesh(x_mult, z_mult, mask_mult[:, i, :].T, shading='flat')
    axes[1].pcolormesh(x_mult_mir, z_mult_mir, mask_mult_mir[:, i, :].T, shading='flat')

    #plt.show()
    plt.close(fig_one)
    plt.close(fig_mult)

    ### ----- Plot yz-plane (front) ------ ###
    i = 1

    title = f'One unit cell: yz-plane (i_x={i})'
    fig_one, axes = prepare_figure_2x2(title)
    axes[0, 0].pcolormesh(y, z, mask[i, :, :].T, shading='flat')
    axes[0, 1].pcolormesh(y_mir, z_mir, mask_mir[i, :, :].T, shading='flat')
    axes[1, 0].pcolormesh(y_1x1x1, z_1x1x1, mask_1x1x1[i, :, :].T, shading='flat')
    axes[1, 1].pcolormesh(y_1x1x1_mir, z_1x1x1_mir, mask_1x1x1_mir[i, :, :].T, shading='flat')

    title = helper_title + f'yz-plane (i_x={i})'
    fig_mult, axes = prepare_figure_1x2(title)
    axes[0].pcolormesh(y_mult, z_mult, mask_mult[i, :, :].T, shading='flat')
    axes[1].pcolormesh(y_mult_mir, z_mult_mir, mask_mult_mir[i, :, :].T, shading='flat')

    #plt.show()
    plt.close(fig_one)
    plt.close(fig_mult)

    ### ----- Plot yz-plane (back) ------ ###
    i = -2

    title = f'One unit cell: yz-plane (i_x={i})'
    fig_one, axes = prepare_figure_2x2(title)
    axes[0, 0].pcolormesh(y, z, mask[i, :, :].T, shading='flat')
    axes[0, 1].pcolormesh(y_mir, z_mir, mask_mir[i, :, :].T, shading='flat')
    axes[1, 0].pcolormesh(y_1x1x1, z_1x1x1, mask_1x1x1[i, :, :].T, shading='flat')
    axes[1, 1].pcolormesh(y_1x1x1_mir, z_1x1x1_mir, mask_1x1x1_mir[i, :, :].T, shading='flat')

    title = helper_title + f'yz-plane (i_x={i})'
    fig_mult, axes = prepare_figure_1x2(title)
    axes[0].pcolormesh(y_mult, z_mult, mask_mult[i, :, :].T, shading='flat')
    axes[1].pcolormesh(y_mult_mir, z_mult_mir, mask_mult_mir[i, :, :].T, shading='flat')

    plt.show()
    plt.close(fig_one)
    plt.close(fig_mult)

def plot_border_of_geometry(ax, nb_grid_pts_xy, X_xy, Y_xy, mask_xy,
                            linewidth=1., linecolor='black'):
    """
    Plot the border of the geometry described by mask.

    Input
    -----
    ax: matplotlib.pyplot.ax object
        Axis on which the border should be plotted.
    nb_grid_pts_xy: list with two ints
                    Number of pixels in each direction
    X_xy: np.array([nb_grid_pts[0]+1, nb_grid_pts[1]+1]) of floats
          x-coordinate for each node.
    Y_xy: np.array([nb_grid_pts[0]+1, nb_grid_pts[1]+1]) of floats
          y-coordinate for each node.
    mask_xy: np.array(nb_grid_pts) of ints
             Describes the geometry. 1 corresponds to material, 0 to void.
    linewidth: float
               Linewidth of the plotted lines. Default is 1
    linecolor: string
               Color of the plotted lines. Must be color recognized by
               matplotlib. Default is 'black'.
    """
    # Plot geometry in the interior of the plot
    for ix in range(nb_grid_pts_xy[0]-1):
        for iy in range(nb_grid_pts_xy[1]-1):
            if mask_xy[ix, iy] != mask_xy[ix+1, iy]:
                ax.plot([X_xy[ix+1, iy], X_xy[ix+1, iy+1]],
                        [Y_xy[ix+1, iy], Y_xy[ix+1, iy+1]],
                        linewidth=linewidth, color=linecolor)
            if mask_xy[ix, iy] != mask_xy[ix, iy+1]:
                ax.plot([X_xy[ix, iy+1], X_xy[ix+1, iy+1]],
                        [Y_xy[ix, iy+1], Y_xy[ix+1, iy+1]],
                        linewidth=linewidth, color=linecolor)

    # Plot geometry at the borders of the plot
    for ix in range(nb_grid_pts_xy[0]-1):
        if mask_xy[ix, -1] != mask_xy[ix+1, -1]:
            ax.plot([X_xy[ix+1, -2], X_xy[ix+1, -1]],
                    [Y_xy[ix+1, -2], Y_xy[ix+1, -1]],
                    linewidth=linewidth, color=linecolor)
        if mask_xy[ix, 0] == 1:
            ax.plot([X_xy[ix, 0], X_xy[ix+1, 0]], [Y_xy[ix, 0], Y_xy[ix+1, 0]],
                    linewidth=linewidth, color=linecolor)
        if mask_xy[ix, -1] == 1:
            ax.plot([X_xy[ix, -1], X_xy[ix+1, -1]], [Y_xy[ix, -1], Y_xy[ix+1, -1]],
                    linewidth=linewidth, color=linecolor)
    for iy in range(nb_grid_pts_xy[1]-1):
        if mask_xy[-1, iy] != mask_xy[-1, iy+1]:
            ax.plot([X_xy[-2, iy+1], X_xy[-1, iy+1]],
                    [Y_xy[-2, iy+1], Y_xy[-1, iy+1]],
                    linewidth=linewidth, color=linecolor)
        if mask_xy[0, iy] == 1:
            ax.plot([X_xy[0, iy], X_xy[0, iy+1]], [Y_xy[0, iy], Y_xy[0, iy+1]],
                    linewidth=linewidth, color=linecolor)
        if mask_xy[-1, iy] == 1:
            ax.plot([X_xy[-1, iy], X_xy[-1, iy+1]], [Y_xy[-1, iy], Y_xy[-1, iy+1]],
                    linewidth=linewidth, color=linecolor)

    # Plot geometry at the corners of the plot
    if mask_xy[-1, 0] == 1:
        ax.plot([X_xy[-2, 0], X_xy[-1, 0]], [Y_xy[-2, 0], Y_xy[-1, 0]],
                linewidth=linewidth, color=linecolor)
    if mask_xy[0, -1] == 1:
        ax.plot([X_xy[0, -2], X_xy[0, -1]], [Y_xy[0, -2], Y_xy[0, -1]],
                linewidth=linewidth, color=linecolor)
    if mask_xy[-1, -1] == 1:
        ax.plot([X_xy[-2, -1], X_xy[-1, -1]], [Y_xy[-2, -1], Y_xy[-1, -1]],
                linewidth=linewidth, color=linecolor)
        ax.plot([X_xy[-1, -2], X_xy[-1, -1]], [Y_xy[-1, -2], Y_xy[-1, -1]],
                linewidth=linewidth, color=linecolor)

def check_2D_plot():
    ### ----- Parameter definitions ----- ###
    # Geometry
    a = 1
    thickness = 0.06
    radius_out = 0.4
    radius_inn = radius_out - thickness
    angle_mat = np.pi * 35 / 180
    N_uc = [2, 2, 3]

    # Discretization
    Nxyz = 20
    nb_grid_pts_uc = [Nxyz, Nxyz, Nxyz]

    helper_title = f'{N_uc[0]}x{N_uc[1]}x{N_uc[2]} unit cells: '

    # Plot borders of geometry
    linewidth = 1.5
    linecolor = 'red'

    ### ----- Define geometries ----- ###
    # Original geometry (1x1x1 unit cells)
    geo_1x1x1, lengths_1x1x1 =\
        geo.chiral_2_mult_unit_cell([1, 1, 1], nb_grid_pts_uc, a,
                                            radius_out, radius_inn,
                                            thickness, alpha=angle_mat)
    x_1x1x1 = np.linspace(0, lengths_1x1x1[0], nb_grid_pts_uc[0]+1, endpoint=True)
    y_1x1x1 = np.linspace(0, lengths_1x1x1[1], nb_grid_pts_uc[1]+1, endpoint=True)
    z_1x1x1 = np.linspace(0, lengths_1x1x1[2], nb_grid_pts_uc[2]+1, endpoint=True)

    # Mirrored geometry (1x1x1 unit cells)
    geo_1x1x1_mir, lengths_1x1x1_mir =\
        geo.chiral_2_mult_unit_cell_mirrored([1, 1, 1], nb_grid_pts_uc, a,
                                            radius_out, radius_inn,
                                            thickness, alpha=angle_mat)
    x_1x1x1_mir = np.linspace(0, lengths_1x1x1_mir[0], nb_grid_pts_uc[0]+1, endpoint=True)
    y_1x1x1_mir = np.linspace(0, lengths_1x1x1_mir[1], nb_grid_pts_uc[1]+1, endpoint=True)
    z_1x1x1_mir = np.linspace(0, lengths_1x1x1_mir[2], nb_grid_pts_uc[2]+1, endpoint=True)

    # Original geometry (N_uc unit cells)
    geo_mult, lengths_mult =\
        geo.chiral_2_mult_unit_cell(N_uc, nb_grid_pts_uc, a,
                                            radius_out, radius_inn,
                                            thickness, alpha=angle_mat)
    nb_grid_pts = geo_mult.shape
    x_mult = np.linspace(0, lengths_mult[0], nb_grid_pts[0]+1, endpoint=True)
    y_mult = np.linspace(0, lengths_mult[1], nb_grid_pts[1]+1, endpoint=True)
    z_mult = np.linspace(0, lengths_mult[2], nb_grid_pts[2]+1, endpoint=True)

    # Mirrored geometry (N_uc unit cells)
    geo_mult_mir, lengths_mult_mir =\
        geo.chiral_2_mult_unit_cell_mirrored(N_uc, nb_grid_pts_uc, a,
                                            radius_out, radius_inn,
                                            thickness, alpha=angle_mat)
    nb_grid_pts = geo_mult_mir.shape
    x_mult_mir = np.linspace(0, lengths_mult_mir[0], nb_grid_pts[0]+1, endpoint=True)
    y_mult_mir = np.linspace(0, lengths_mult_mir[1], nb_grid_pts[1]+1, endpoint=True)
    z_mult_mir = np.linspace(0, lengths_mult_mir[2], nb_grid_pts[2]+1, endpoint=True)

    ### ----- Plot voxel data on xz-plane: One unit cell ----- ###
    # Plot data like stress, strain, material, ...
    iy = 1 # y-postion of plotted plane

    # Prepare figure
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(f'Plot voxel data (one unit cell)')
    axes[0].set_title('Original')
    axes[1].set_title('Mirrored')
    for i in range(2):
        axes[i].set_aspect('equal')
        axes[i].set_xlabel('Position x')
        axes[i].set_ylabel('Position z')
        axes[i].set_ylim((-0.01, 1.01))


    # Only for checking pcolormesh
    #geo_1x1x1[Nxyz//2, iy, 0] = geo_1x1x1_mir[Nxyz//2, iy, 0] = 0.5

    # Mask vacuum
    mask_el = geo_1x1x1[:, iy, :].T
    mask_el = (mask_el == 0)
    mask_el_mir = geo_1x1x1_mir[:, iy, :].T
    mask_el_mir = (mask_el_mir == 0)

    # Plot Geometry
    X, Z = np.meshgrid(x_1x1x1, z_1x1x1)
    n = np.ma.masked_array(geo_1x1x1[:, iy, :].T, mask_el)
    pm = axes[0].pcolormesh(X, Z, n, shading='flat')
    cbar = fig.colorbar(pm, ax=axes[0])
    n = np.ma.masked_array(geo_1x1x1_mir[:, iy, :].T, mask_el_mir)
    pm = axes[1].pcolormesh(x_1x1x1_mir, z_1x1x1_mir, n, shading='flat')
    cbar = fig.colorbar(pm, ax=axes[1])

    # Plot Border of geometry
    nb_grid_pts = [nb_grid_pts_uc[0], nb_grid_pts_uc[1]]
    plot_border_of_geometry(axes[0], nb_grid_pts, X.T, Z.T, geo_1x1x1[:, iy, :],
                            linewidth=linewidth, linecolor=linecolor)
    X, Z = np.meshgrid(x_1x1x1_mir, z_1x1x1_mir)
    plot_border_of_geometry(axes[1], nb_grid_pts, X.T, Z.T, geo_1x1x1_mir[:, iy, :],
                            linewidth=linewidth, linecolor=linecolor)

    plt.show()
    plt.close(fig)

     ### ----- Plot voxel data on xz-plane: Multiple unit cells ----- ###
    # Plot data like stress, strain, material, ...
    iy = 1 # y-postion of plotted plane

    # Prepare figure
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(f'Plot voxel data (mult unit cell)')
    axes[0].set_title('Original')
    axes[1].set_title('Mirrored')
    for i in range(2):
        axes[i].set_aspect('equal')
        axes[i].set_xlabel('Position x')
        axes[i].set_ylabel('Position z')

    # Mask vacuum
    mask_el = geo_mult[:, iy, :].T
    mask_el = (mask_el == 0)
    mask_el_mir = geo_mult_mir[:, iy, :].T
    mask_el_mir = (mask_el_mir == 0)

    # Plot Geometry
    n = np.ma.masked_array(geo_mult[:, iy, :].T, mask_el)
    pm = axes[0].pcolormesh(x_mult, z_mult, n, shading='flat')
    cbar = fig.colorbar(pm, ax=axes[0])
    n = np.ma.masked_array(geo_mult_mir[:, iy, :].T, mask_el_mir)
    pm = axes[1].pcolormesh(x_mult_mir, z_mult_mir, n, shading='flat')
    cbar = fig.colorbar(pm, ax=axes[1])

    # Plot Border of geometry
    X, Z = np.meshgrid(x_mult_mir, z_mult_mir)
    nb_grid_pts = [geo_mult.shape[0], geo_mult.shape[2]]
    plot_border_of_geometry(axes[0], nb_grid_pts, X.T, Z.T, geo_mult[:, iy, :],
                            linewidth=linewidth, linecolor=linecolor)
    X, Z = np.meshgrid(x_mult_mir, z_mult_mir)
    plot_border_of_geometry(axes[1], nb_grid_pts, X.T, Z.T, geo_mult_mir[:, iy, :],
                            linewidth=linewidth, linecolor=linecolor)

    #plt.show()
    plt.close(fig)

    ### ----- Plot node data on xz-plane: One unit cell ----- ###
    # Plot data like displacement
    iy = 1 # y-postion of plotted plane
    Nx = nb_grid_pts_uc[0]
    Nz = nb_grid_pts_uc[2]

    # Create data
    data_1 = np.empty((Nx+1, Nz+1))
    data_1[:-1, :-1] = geo_1x1x1[:, iy, :]
    for i in range(Nx):
        if geo_1x1x1[i, iy, -1] == 1:
            data_1[i, -1] = 1
        else:
            data_1[i, -1] = 0
    for i in range(Nz):
        if geo_1x1x1[-1, iy, i] == 1:
            data_1[-1, i] = 1
        else:
            data_1[-1, i] = 0
    if geo_1x1x1[-1, iy, -1] == 1:
        data_1[-1, -1] = 1
    else:
        data_1[-1, -1] = 0
    data_2 = np.arange((Nx+1) * (Nz+1)).reshape((Nx+1, Nz+1))
    data_2[-1, Nz//2] = -50

    # Prepare figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f'Plot node data with gouraud shading (one unit cell)')
    axes[0].set_title('Material')
    axes[1].set_title('Material (void is masked)')
    axes[2].set_title('Values (void is masked)')
    for i in range(3):
        axes[i].set_aspect('equal')
        axes[i].set_xlabel('Position x')
        axes[i].set_ylabel('Position z')


    # Plot material (with void)
    n = data_1.T
    pm = axes[0].pcolormesh(x_1x1x1, z_1x1x1, n, shading='gouraud')
    cbar = fig.colorbar(pm, ax=axes[0])
    X, Z = np.meshgrid(x_1x1x1, z_1x1x1)
    plot_border_of_geometry(axes[0], [Nx, Nz], X.T, Z.T, geo_1x1x1[:, iy, :],
                            linewidth=linewidth, linecolor=linecolor)

    # Masks to ignore void
    mask = geo_1x1x1[:, iy, :].T
    helper = (mask != 1)
    mask_points = np.full([Nx+1, Nz+1], True)
    mask_points[:-1, :-1] = helper
    mask_points[1:, :-1] = np.logical_and(helper, mask_points[1:, :-1])
    mask_points[:-1, 1:] = np.logical_and(helper, mask_points[:-1, 1:])
    mask_points[1:, 1:] = np.logical_and(helper, mask_points[1:, 1:])

    # Plot material (while masking void)
    n = data_1.T # Both valid
    n = np.ma.masked_array(data_1.T, mask_points) # Both valid
    pm = axes[1].pcolormesh(x_1x1x1, z_1x1x1, n, shading='gouraud')
    cbar = fig.colorbar(pm, ax=axes[1])
    X, Z = np.meshgrid(x_1x1x1, z_1x1x1)
    plot_border_of_geometry(axes[1], [Nx, Nz], X.T, Z.T, geo_1x1x1[:, iy, :],
                            linewidth=linewidth, linecolor=linecolor)
    # Note: Due to the gouraud shading, a few elements in corners are not
    # masked correctly with mask_points. The next line paints these white.
    # This line is sufficient to make all void elements white, however,
    # only with mask_points is the automatic colorbar range adjusted.
    # Either use both masks or adjust colorbar manually.
    axes[1].pcolormesh(X, Z, np.ma.masked_array(mask, mask),
                       cmap=mpl.colors.ListedColormap(['white']))

    # Plot values (while masking void)
    #n = data_2.T # Will result in badly adjusted colorbar
    n = np.ma.masked_array(data_2.T, mask_points)
    pm = axes[2].pcolormesh(x_1x1x1, z_1x1x1, n, shading='gouraud')
    cbar = fig.colorbar(pm, ax=axes[2])
    X, Z = np.meshgrid(x_1x1x1, z_1x1x1)
    plot_border_of_geometry(axes[2], [Nx, Nz], X.T, Z.T, geo_1x1x1[:, iy, :],
                            linewidth=linewidth, linecolor=linecolor)
    axes[2].pcolormesh(X, Z, np.ma.masked_array(mask, mask),
                       cmap=mpl.colors.ListedColormap(['white']))

    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    #calculation_mult_unit_cells()
    #calculation_with_cylinder()
    #folder = 'results/results_nemo/chiral2/mult_unit_cells/'
    #folder = folder + 'Nuc_z=1_Nxyz=50_twist=0.05_strain=0.0/'
    #plot_with_paper(folder)
    #check_geometries()
    check_2D_plot()

"""
@file   test_chiral_metamaterial2.py

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
sys.path.insert(0, os.path.join(os.getcwd(), "/usr/local/lib/python3.8/site-packages"))

import numpy as np
import matplotlib.pyplot as plt
from time import time

import muSpectre as µ
from NuMPI import MPI
from NuMPI.Tools import Reduction

from muChirality.EigenStrainTorsion import EigenStrain
from muChirality.CalculationsTorsion import calculations
from muSpectre.gradient_integration import get_complemented_positions
import muChirality.Geometries as geo

def plot_on_paper_results(folder):
    imfile = 'results_paper.jpg'
    # Read data
    data_file = folder + 'data.txt'
    data = np.loadtxt(data_file, skiprows=1)
    N_uc_list = data[:, 0]
    E_eff_z = data[:, 2]
    twist_per_strain = data[:, 4]

    # Other parameters
    dpi = 100
    color = 'green'
    marker = 'x'
    markersize = 12
    linewidth = 4

    ### ----- Plot data ----- ###
    # Prepare figure
    im = image.imread(imfile)
    fig, axim = plt.subplots(figsize=(im.shape[1]/dpi, im.shape[0]/dpi), dpi=dpi)
    axim.imshow(im, aspect='equal')
    axim.axis('off')

    # Plot E_eff_z
    ax = fig.add_axes([0.25, 0.17, 0.635, 0.295], facecolor='None')
    ax.set_xlim([0.49, 5.49])
    ax.set_ylim([0, 45])
    ax.plot(nb_unit_cells, E_eff_z, marker=marker, linewidth=linewidth,
            markersize=markersize, color=color)
    ax.axis('off')

    # Plot twist per strain
    ax = fig.add_axes([0.25, 0.474, 0.635, 0.297], facecolor='None')
    ax.set_xlim([0.49, 5.49])
    ax.set_ylim([-0.21, 2.48])
    ax.plot(nb_unit_cells, twist_per_strain, marker=marker, linewidth=linewidth,
        markersize=markersize, color=color)
    ax.axis('off')

    name = folder + 'plot_on_paper.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)

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
    a = 0.5
    thickness = 0.06 * a
    radius_out = 0.4 * a
    radius_inn = 0.34 * a
    angle_mat = np.pi * 35 / 180

    # Nb of unit cells in RVE
    N_uc_list = [1, 2]
    restart = False
    plates = False
    Nz_changes = False
    Nz = 2

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
    Young = 2600
    Poisson = 0.4

    # Loading
    twist = 0.05
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
    folder = 'results/chiral2/'
    if Nz_changes:
        folder += 'Nz_changes'
    else:
        folder += f'Nz={Nz}'
    if plates:
        folder += '_plates'
    folder += f'Nxyz={nb_grid_pts_uc[0]}/'
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
        twist_in_degree = np.arctan(twist * lengths[2]) / np.pi * 180
        strain_zz_in_percent = force / stiff_z * 100

        twist_per_strain = twist_in_degree / strain_zz_in_percent

        if MPI.COMM_WORLD.rank == 0:
            print('Twist (degree):', twist_in_degree)
            print('strain (%):', strain_zz_in_percent)
            print('Force:', force)
            print('Twist/strain (degree/%):', twist_per_strain)
            print(f'Finished calculation {index+1} of {len(N_uc_list)}')
            print()

        ### ----- Save results ----- ###
        if MPI.COMM_WORLD.rank == 0:
            with open(name, 'a') as f:
                np.savetxt(f, [N_uc, stiff_z, E_eff_z, force, twist_per_strain], newline=' ')
                print('', file=f)

        # Save deformed state only if the calculation is not parallel
        if MPI.COMM_WORLD.size == 1:
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
    if MPI.COMM_WORLD. rank == 0:
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

if __name__ == "__main__":
    calculation_mult_unit_cells()
    #calculation_with_cylinder()

"""
@file   chiral_metamaterial_paper.py

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
from time import time

import muSpectre as µ
import muGrid
from NuMPI import MPI
from NuMPI.Tools import Reduction

from muChirality.EigenStrainTorsion import EigenStrain
from muChirality.CalculationsTorsion import calculations
from muSpectre.gradient_integration import get_complemented_positions
import muChirality.Geometries as geo

###################################################################################################
###################################################################################################
###################################################################################################
def mesh_refinement(test_case=False):
    ### ----- Parameter definitions ----- ###
    restart = False

    # Geometry
    a = 0.5 # in mm
    thickness = 0.06 * a
    radius_out = 0.4 * a
    radius_inn = 0.34 * a
    angle_mat = np.pi * 35 / 180

    # Nb of unit cells in RVE
    N_uc = 1
    nb_unit_cells = [N_uc, N_uc, N_uc]

    # Discretization
    dim = 3
    if test_case:
        N_list = [16, 30]
    else:
        N_list = [60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300,
                  320, 340, 360, 380, 400, 420, 440, 460, 480, 500]
    gradient, weights = µ.linear_finite_elements.gradient_3d_5tet

    if MPI.COMM_WORLD.rank == 0:
        print(f'--- Mesh refinement: Parameters ---')
        print('Restart:', restart)
        print(f'MPI size = {MPI.COMM_WORLD.size}')
        print(f'List of nb_grid_pts:', N_list)
        print(f'nb_unit_cells = {N_uc}x{N_uc}x{N_uc}')
        print(µ.version.info())

    # Material
    Young = 2600 # in MPa
    Poisson = 0.4

    # Loading
    twist = 0.05 # in 1/mm
    delta_eps = np.zeros((3, 3))

    # Formulation
    formulation = µ.Formulation.small_strain

    # muSpectre solver parameters
    newton_tol       = 1e-7
    cg_tol           = 1e-7 # tolerance for cg algo
    equil_tol        = 1e-7 # tolerance for equilibrium
    maxiter          = 10000
    verbose          = µ.Verbosity.Silent
    if MPI.COMM_WORLD.size == 1:
        fft = 'fftw'
    else:
        fft = 'pfft' # Parallel fft

    # For saving
    F0 = np.eye(3)
    if test_case:
        folder = f'results_testing/chiral_mesh_refinement_mpi{MPI.COMM_WORLD.size}/'
    else:
        folder = f'chiral_mesh_refinement_mpi{MPI.COMM_WORLD.size}/'
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
        helper = 'cp chiral_metamaterial_paper.py ' + folder
        os.system(helper)

        # File  for saving data
        with open(name, 'w') as f:
            title = 'nb_grid_pts_every_direction  average_force_z (N)  '
            title += 'nb_elements_without_vacuum  time_calculation (s)  twist (1/mm)'
            title += '  angle_per_unit_cell (rad/1)'
            print(title, file=f)

    ### ----- Calculation ----- ###
    for index, N in enumerate(N_list):
        if (MPI.COMM_WORLD.rank == 0):
            print(f'Calculation {index + 1} of {len(N_list)}: N={N}')

        # Define geometry
        nb_grid_pts_uc = [N, N, N]
        mask, lengths =\
            geo.chiral_2_mult_unit_cell(nb_unit_cells, nb_grid_pts_uc, a, radius_out, radius_inn,
                                        thickness, alpha=angle_mat)
        nb_grid_pts = mask.shape
        nb_elements_without_vacuum = np.sum(mask)

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

        shape = (dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts)
        solver = µ.solvers.KrylovSolverCG(cell, cg_tol, maxiter, verbose)

        # EigenStrain initialization
        eigen_class = EigenStrain(twist, lengths, nb_grid_pts, cell.fft_engine.subdomain_slices,
                                  lengths[0]/2, lengths[1]/2)

        # Solve muSpectre
        t = time()
        res = µ.solvers.newton_cg(cell, delta_eps, solver, newton_tol, equil_tol,
                                  verbose, μ.solvers.IsStrainInitialised.No,
                                  µ.StoreNativeStress.No, eigen_class.eigen_strain_func)
        stress = res.stress.reshape(shape, order='F')

        # Calculate force in z-direction
        hx = lengths[0] / nb_grid_pts[0]
        hy = lengths[1] / nb_grid_pts[1]
        hz = lengths[2] / nb_grid_pts[2]
        force = hx * hy * hz / 6 * np.sum(stress[2, 2])
        force += hx * hy * hz / 6 * np.sum(stress[2, 2, 0])
        force = Reduction(MPI.COMM_WORLD).sum(force) / lengths[2]

        # Angle per unit cell
        angle_per_uc = np.arctan(twist * lengths[2] / nb_unit_cells[2])

        # Save result
        if (MPI.COMM_WORLD.rank == 0):
            with open(name, 'a') as f:
                to_save = f'{N}  {force}  {nb_elements_without_vacuum}  '
                to_save += f'{time() - t}  {twist}  {angle_per_uc}'
                print(to_save, file=f)

###################################################################################################
###################################################################################################
###################################################################################################
def calculation_mult_unit_cells(test_case=False, N_uc=1):
    ### ----- Parameter definitions ----- ###
    t = time()

    # Geometry
    a = 0.5 # in mm
    thickness = 0.06 * a
    radius_out = 0.4 * a
    radius_inn = 0.34 * a
    angle_mat = np.pi * 35 / 180

    # Nb of unit cells in RVE
    if test_case:
        N_uc = 1
        N_uc_z_list = [1, 2]
    else:
        N_uc_z_list = [1, 2, 3, 4]

    # Discretization
    dim = 3
    if test_case:
        N = 30
    else:
        N = 100
    nb_grid_pts_uc = [N, N, N]
    gradient, weights = µ.linear_finite_elements.gradient_3d_5tet

    if MPI.COMM_WORLD.rank == 0:
        print(f'--- Change Nz: Parameters ---')
        print(f'MPI size = {MPI.COMM_WORLD.size}')
        print(f'nb_unit_cells = {N_uc}x{N_uc}xN_uc_z')
        helper = f'nb_grid_pts_uc = {nb_grid_pts_uc[0]}x{nb_grid_pts_uc[1]}'
        helper += f'x{nb_grid_pts_uc[2]}'
        print(helper)
        print('List nb_unit_cells_z =', N_uc_z_list)
        print(µ.version.info())

    # Material
    Young = 2600 # in MPa
    Poisson = 0.4

    # Loading
    twist = 0.05 # in 1/mm
    delta_eps = np.zeros((3, 3))

    # Formulation
    formulation = µ.Formulation.small_strain

    # muSpectre solver parameters
    newton_tol       = 1e-7
    cg_tol           = 1e-7 # tolerance for cg algo
    equil_tol        = 1e-7 # tolerance for equilibrium
    maxiter          = 10000
    verbose          = µ.Verbosity.Silent
    if MPI.COMM_WORLD.size == 1:
        fft = 'fftw'
    else:
        fft = 'pfft'

    # For saving
    F0 = np.eye(3)
    if test_case:
        folder = 'results_testing/' + f'chiral_Nuc={N_uc}x{N_uc}xNucz_mpi{MPI.COMM_WORLD.size}/'
    else:
        folder = f'chiral_Nuc={N_uc}x{N_uc}xNucz_mpi{MPI.COMM_WORLD.size}/'
    name = folder + 'data.txt'

    ### ----- Prepare saving ----- ###
    if (MPI.COMM_WORLD.rank == 0):
        # Create or clear folder
        if not os.path.exists(folder):
            os.makedirs(folder)
        else:
            shutil.rmtree(folder)
            os.makedirs(folder)

        # Copy this file into the folder
        helper = 'cp chiral_metamaterial_paper.py ' + folder
        os.system(helper)

        # File  for saving data
        with open(name, 'w') as f:
            title = 'nb_unit_cells_z_direction    nb_grid_pts   force_z (N)    '
            title += 'twist (1/mm)    angle_per_unit_cell (rad/1)   time (s)'
            print(title, file=f)

    ### ----- Calculation ----- ###
    for index, N_uc_z in enumerate(N_uc_z_list):
        t1 = time()
        # Define geometry
        nb_unit_cells = [N_uc, N_uc, N_uc_z]
        mask, lengths =\
            geo.chiral_2_mult_unit_cell(nb_unit_cells, nb_grid_pts_uc, a,
                                        radius_out, radius_inn,
                                        thickness, alpha=angle_mat)
        nb_grid_pts = mask.shape

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
        shape = (dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts)


        # Krylov solver initialization
        solver = µ.solvers.KrylovSolverCG(cell, cg_tol, maxiter, verbose)

        # Initialize Eigen class
        eigen_class = EigenStrain(twist, lengths, nb_grid_pts, cell.fft_engine.subdomain_slices,
                                  lengths[0]/2, lengths[1]/2)

        # Solve muSpectre
        t2 = time()
        res = µ.solvers.newton_cg(cell, delta_eps, solver, newton_tol, equil_tol,
                                  verbose, μ.solvers.IsStrainInitialised.No,
                                  µ.StoreNativeStress.No, eigen_class.eigen_strain_func)
        stress = res.stress.reshape(shape, order='F')
        strain = res.grad.reshape(shape, order='F')

        # Force in z-direction
        hx = lengths[0] / nb_grid_pts[0]
        hy = lengths[1] / nb_grid_pts[1]
        hz = lengths[2] / nb_grid_pts[2]
        force = hx * hy * hz / 6 * np.sum(stress[2, 2])
        force += hx * hy * hz / 6 * np.sum(stress[2, 2, 0])
        force = Reduction(MPI.COMM_WORLD).sum(force) / lengths[2]

        t3 = time()

        # Calculate fluctuating displacement
        if N_uc_z == 1:
            strain_no_eigen = strain.copy()
            eigen_class.remove_eigen_strain_func(strain_no_eigen)
            displ = get_complemented_positions("d", cell, strain_array=strain_no_eigen,
                                               F0=F0, periodically_complemented=False)

        ### ----- Save results ----- ###
        if MPI.COMM_WORLD.rank == 0:
            print(f'Finished calculation {index+1} of {len(N_uc_z_list)}')
            t = (t3 - t1) / 60
            print(f'Time for calculation = {t:.2} min')

        # Angle per unit cell
        angle_per_uc = np.arctan(twist * lengths[2] / nb_unit_cells[2])

        # Save force
        if MPI.COMM_WORLD.rank == 0:
            with open(name, 'a') as f:
                np.savetxt(f, [N_uc_z, nb_grid_pts[0], nb_grid_pts[1], nb_grid_pts[2],
                               force, twist, angle_per_uc, t3 - t1], newline=' ')
                print('', file=f)

        if N_uc_z == 1:
            comm = muGrid.Communicator(MPI.COMM_WORLD)
            # Create file io object for saving stress and strain
            name_3D = folder + f'data_3D_Nucz=1.nc'
            if os.path.exists(name_3D):
                if comm.rank == 0:
                    os.remove(name_3D)
            MPI.COMM_WORLD.Barrier() # wait for rank 0 to delete the old netcdf file
            file_io_object = muGrid.FileIONetCDF(
                name_3D, muGrid.FileIONetCDF.OpenMode.Write, comm)

            # Register field for saving displacements
            fields = cell.get_field_collection()
            fields.register_real_field('fluctuating_displacement', nb_components=3, sub_division='pixel')
            fluct_displ = fields.get_field('fluctuating_displacement').array()
            fluct_displ[:] = displ[:, None, :, :, :]

            # Register field for saving total strain
            fields.register_real_field('strain_total', components_shape=[3, 3], sub_division='quad_point')
            helper = fields.get_field('strain_total').array()
            helper[:] = strain


            # Register global fields of the cell which you want to write
            file_io_object.register_field_collection(
                field_collection=fields,
                field_names=["strain", "stress", "strain_total",
                             "fluctuating_displacement"])

            # Write values
            file_io_object.append_frame().write(["stress"])
            file_io_object.append_frame().write(["strain"]) # Note: This is the fluctuating strain
            file_io_object.append_frame().write(["strain_total"]) # Note: This is the complete strain
            file_io_object.append_frame().write(["fluctuating_displacement"])
            file_io_object.close()

            if (MPI.COMM_WORLD.rank == 0):
                t = (time() - t3) / 60
                print(f'Time for saving = {t:.2} min')
                print()

###################################################################################################
###################################################################################################
###################################################################################################
def calculation_with_cylinder(test_case=False):
    t = time()
    ### ----- Parameter definitions ----- ###
    # Geometry
    a = 0.5 # in mm
    thickness = 0.06 * a
    radius_out = 0.4 * a
    radius_inn = 0.34 * a
    angle_mat = np.pi * 35 / 180
    lengths_cyl = [1, 1, a]
    dim = 3

    if test_case:
        N_uc_list = [1, 2]
    else:
        N_uc_list = [1, 3]

    # Discretization
    if test_case:
        Nxyz = 30
    else:
        Nxyz = 100
    nb_grid_pts_uc = [Nxyz, Nxyz, Nxyz]
    gradient, weights = µ.linear_finite_elements.gradient_3d_5tet

    # Material
    Young = 100
    Poisson = 0

    # Loading
    if test_case:
        twists = np.array([-0.1, 0.1, 0.2])
    else:
        twists = np.array([-0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, -0.2])
    delta_eps = np.zeros((3, 3))

    # Formulation
    formulation = µ.Formulation.small_strain

    # muSpectre solver parameters
    newton_tol       = 1e-7
    cg_tol           = 1e-7 # tolerance for cg algo
    equil_tol        = 1e-7 # tolerance for equilibrium
    maxiter          = 10000
    verbose          = µ.Verbosity.Silent
    if MPI.COMM_WORLD.size == 1:
        fft = 'fftw'
    else:
        fft = 'pfft'

    # For saving
    folder = f'chiral_comp_cylinder_mpi{MPI.COMM_WORLD.size}/'
    if test_case:
        folder = 'results_testing/' + folder
    name = folder + 'data.txt'

    ### ----- Prepare saving ----- ###
    if (MPI.COMM_WORLD.rank == 0):
        # Create or clear folder
        if not os.path.exists(folder):
            os.makedirs(folder)
        else:
            shutil.rmtree(folder)
            os.makedirs(folder)

        # Copy this file into the folder
        helper = 'cp chiral_metamaterial_paper.py ' + folder
        os.system(helper)

        # Save twists
        with open(name, 'w') as f:
            print('Twists (1/mm)', file=f)
            np.savetxt(f, twists, newline=' ')
            print('', file=f)

    ### ----- Calculation: Cylinder ----- ###
    if (MPI.COMM_WORLD.rank == 0):
        print('----- Cylinder -----')

    # Define geometry
    mask = geo.cylinder(nb_grid_pts_uc, lengths_cyl, radius_out)
    hx = lengths_cyl[0] / nb_grid_pts_uc[0]
    hy = lengths_cyl[1] / nb_grid_pts_uc[1]
    hz = lengths_cyl[2] / nb_grid_pts_uc[2]
    x_rot_axis = lengths_cyl[0] / 2
    y_rot_axis = lengths_cyl[1] / 2

    # muSpectre cell initialization
    cell = µ.Cell(nb_grid_pts_uc, lengths_cyl, formulation, gradient,
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

    forces = np.empty(len(twists))
    for i_twist, twist in enumerate(twists):
        if (MPI.COMM_WORLD.rank == 0):
            print(f'Test {i_twist+1} of {len(twists)}')

        # Initialize Eigen class
        eigen_class = EigenStrain(twist, lengths_cyl, nb_grid_pts_uc,
                                  cell.fft_engine.subdomain_slices,
                                  x_rot_axis, y_rot_axis)

        # Solving
        res = µ.solvers.newton_cg(cell, delta_eps, solver, newton_tol, equil_tol,
                                  verbose, μ.solvers.IsStrainInitialised.No,
                                  µ.StoreNativeStress.No, eigen_class.eigen_strain_func)
        shape = (dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts)
        stress = res.stress.reshape(shape, order='F')

        # Force in z-direction
        force = hx * hy * hz / 6 * np.sum(stress[2, 2])
        force += hx * hy * hz / 6 * np.sum(stress[2, 2, 0])
        force = Reduction(MPI.COMM_WORLD).sum(force) / lengths_cyl[2]
        forces[i_twist] = force

    # Save results
    angle_per_uc = np.arctan(twists * lengths_cyl[2])
    if MPI.COMM_WORLD.rank == 0:
        with open(name, 'a') as f:
            print(f'Rotation angle per unit cell (rad/1): Cylinder', file=f)
            np.savetxt(f, angle_per_uc, newline=' ')
            print('', file=f)
            print('Force in z-direction (N): Cylinder', file=f)
            np.savetxt(f, forces, newline=' ')
            print('', file=f)

    # Delete variables
    del cell
    del mat
    del vac
    del mask
    del solver
    del eigen_class
    del res
    del stress


    ### ----- Calculation: Metamaterial ----- ###
    for N_uc in N_uc_list:
        if (MPI.COMM_WORLD.rank == 0):
            print(f'----- Metamaterial: {N_uc}x{N_uc}x1 unit cells -----')

        # Define geometry
        mask, lengths = geo.chiral_2_mult_unit_cell([N_uc, N_uc, 1], nb_grid_pts_uc, a,
                                            radius_out, radius_inn,
                                            thickness, alpha=angle_mat)
        nb_grid_pts = mask.shape
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

        for i_twist, twist in enumerate(twists):
            if (MPI.COMM_WORLD.rank == 0):
                print(f'Test {i_twist+1} of {len(twists)}')

            # Initialize Eigen class
            eigen_class = EigenStrain(twist, lengths, nb_grid_pts, cell.fft_engine.subdomain_slices,
                                      x_rot_axis, y_rot_axis)

            # Solving
            res = µ.solvers.newton_cg(cell, delta_eps, solver, newton_tol, equil_tol,
                                      verbose, μ.solvers.IsStrainInitialised.No,
                                      µ.StoreNativeStress.No, eigen_class.eigen_strain_func)
            shape = (dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts)
            stress = res.stress.reshape(shape, order='F')

            # Force in z-direction
            force = hx * hy * hz / 6 * np.sum(stress[2, 2])
            force += hx * hy * hz / 6 * np.sum(stress[2, 2, 0])
            force = Reduction(MPI.COMM_WORLD).sum(force) / lengths[2]
            forces[i_twist] = force

        # Rotation angle per unit cell
        angle_per_uc = np.arctan(twists * lengths[2])

        # Save results
        if MPI.COMM_WORLD.rank == 0:
            with open(name, 'a') as f:
                print(f'Rotation angle per unit cell (rad/1): Metamaterial with {N_uc}x{N_uc}x1 unit cells', file=f)
                np.savetxt(f, angle_per_uc, newline=' ')
                print('', file=f)
                print(f'Force in z-direction (N): Metamaterial with {N_uc}x{N_uc}x1 unit cells', file=f)
                np.savetxt(f, forces, newline=' ')
                print('', file=f)

        # Delete variables
        del cell
        del mat
        del vac
        del mask
        del solver
        del eigen_class
        del res
        del stress

if __name__ == "__main__":
    #mesh_refinement(test_case=True)
    #calculation_mult_unit_cells(test_case=True)
    #calculation_with_cylinder(test_case=True)
    mesh_refinement(test_case=False)
    calculation_mult_unit_cells(test_case=False)
    calculation_with_cylinder(test_case=False)
    calculation_mult_unit_cells(test_case=False, N_uc=3)

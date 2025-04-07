"""
@file   test_mesh_refinements.py

@author Indre  Joedicke <indre.joedicke@imtek.uni-freiburg.de>

@date   28 Mar 2025

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
from NuMPI import MPI
from NuMPI.Tools import Reduction
from NuMPI.IO import save_npy

from muChirality.EigenStrainTorsion import EigenStrain
from muChirality.CalculationsTorsion import calculations
import muChirality.Geometries as geo

def test_convergence_rotation_worker(create_geometry, N_list, folder, name_geometry):
    """ Test the convergence of one geometry.
    Input
    -----
    create_geometry: callable
                     Takes the number of grid pts in ONE direction as input and returns
                     the discretized geometry and the lengths of the unit cell.
    N_list: list of ints
            Number of grid pts in one direction
    folder: string
            Folder where the results are saved.
    name_geometry: string
                   Name of the geometry. For saving the results.
    """
    ### ----- Parameter definitions ----- ###
    # Discretization
    dim = 3
    gradient, weights = µ.linear_finite_elements.gradient_3d_5tet

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
    fft = 'mpi' # Parallel fft

    # For saving
    F0 = np.eye(3)
    output = folder + 'general_output.txt'
    output_data = folder + name_geometry + '_rotation_data.txt'

    ### ----- Prepare saving ----- ###
    if (MPI.COMM_WORLD.rank == 0):
        # File  for saving data
        with open(output_data, 'w') as f:
            title = 'nb_grid_pts_every_direction  average_force_z (N)  '
            title += 'nb_elements_without_vacuum  time_calculation (s)'
            print(title, file=f)
        with open(output, 'a') as f:
            message = '----- Start: Test rotation of ' + name_geometry + ' -----'
            print(message, file=f)

    ### ----- Calculation ----- ###
    for index, N in enumerate(N_list):
        if (MPI.COMM_WORLD.rank == 0):
            with open(output, 'a') as f:
                print(f'Calculation {index + 1} of {len(N_list)}: N={N}', file=f)
        t1 = time()

        # Define geometry
        nb_grid_pts = [N, N, N]
        mask, lengths = create_geometry(nb_grid_pts)
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
        eigen_class = EigenStrain(cell.pixels, twist, lengths, nb_grid_pts,
                                  lengths[0]/2, lengths[1]/2)

        # Solve muSpectre
        t2 = time()
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

        # Save result
        if (MPI.COMM_WORLD.rank == 0):
            with open(output_data, 'a') as f:
                to_save = f'{N}  {force}  {nb_elements_without_vacuum}  {time() - t1}'
                print(to_save, file=f)

def test_convergence_traction_worker(create_geometry, N_list, folder, name_geometry):
    """ Test the convergence of one geometry.
    Input
    -----
    create_geometry: callable
                     Takes the number of grid pts in ONE direction as input and returns
                     the discretized geometry and the lengths of the unit cell.
    N_list: list of ints
            Number of grid pts in one direction
    folder: string
            Folder where the results are saved.
    name_geometry: string
                   Name of the geometry. For saving the results.
    """
    ### ----- Parameter definitions ----- ###
    # Discretization
    dim = 3
    gradient, weights = µ.linear_finite_elements.gradient_3d_5tet

    # Material
    Young = 2600 # in MPa
    Poisson = 0.4

    # Loading
    delta_eps = np.zeros((3, 3))
    delta_eps[2, 2] = 0.05

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
    output = folder + 'general_output.txt'
    output_data = folder + name_geometry + '_traction_data.txt'

    ### ----- Prepare saving ----- ###
    if (MPI.COMM_WORLD.rank == 0):
        # File  for saving data
        with open(output_data, 'w') as f:
            title = 'nb_grid_pts_every_direction  average_force_x  '
            title += 'time_calculation (s)'
            print(title, file=f)
        with open(output, 'a') as f:
            message = '----- Start: Test traction of ' + name_geometry + ' -----'
            print(message, file=f)

    ### ----- Calculation ----- ###
    for index, N in enumerate(N_list):
        if (MPI.COMM_WORLD.rank == 0):
            with open(output, 'a') as f:
                print(f'Calculation {index + 1} of {len(N_list)}: N={N}', file=f)
        t1 = time()

        # Define geometry
        nb_grid_pts = [N, N, N]
        mask, lengths = create_geometry(nb_grid_pts)

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


        # Solve muSpectre
        t2 = time()
        res = µ.solvers.newton_cg(cell, delta_eps, solver, newton_tol, equil_tol,
                                  verbose, μ.solvers.IsStrainInitialised.No,
                                  µ.StoreNativeStress.No)
        stress = res.stress.reshape(shape, order='F')

        # Force in z-direction
        hx = lengths[0] / nb_grid_pts[0]
        hy = lengths[1] / nb_grid_pts[1]
        hz = lengths[2] / nb_grid_pts[2]
        force_z = hx * hy * hz / 6 * np.sum(stress[2, 2])
        force_z += hx * hy * hz / 6 * np.sum(stress[2, 2, 0])
        force_z = Reduction(MPI.COMM_WORLD).sum(force_z) / lengths[2]

        # Save result
        if (MPI.COMM_WORLD.rank == 0):
            with open(output_data, 'a') as f:
                to_save = f'{N}  {force_z}  {time() - t1}'
                print(to_save, file=f)

def create_square_beam(nb_grid_pts):
    # Parameters
    lengths = [1, 1, 10]
    L_beam_x = L_beam_y = 0.9

    # Create discretized geometry
    mask = geo.rectangular_beam(nb_grid_pts, lengths, L_beam_x, L_beam_y)
    return mask, lengths

def create_cylinder(nb_grid_pts):
    # Parameters
    lengths = [1, 1, 10]
    radius = 0.4

    # Create discretized geometry
    mask = geo.cylinder(nb_grid_pts, lengths, radius)
    return mask, lengths

def create_chiral_1(nb_grid_pts):
    # Parameters
    lengths = [1, 1, 10]
    radius = 0.45
    thickness = 0.1
    alpha = 0.04

    # Create discretized geometry
    mask = geo.chiral_metamaterial(nb_grid_pts, lengths, radius, thickness, alpha=alpha)
    return mask, lengths

def create_chiral_2(nb_grid_pts):
    # Parameters
    lengths = [1, 1, 10]
    radius = 0.45# Geometry
    a = 0.5 # in mm
    thickness = 0.06 * a
    radius_out = 0.4 * a
    radius_inn = 0.34 * a
    angle_mat = np.pi * 35 / 180

    # Nb of unit cells in RVE
    nb_unit_cells = [1, 1, 1]

    # Create discretized geometry
    mask, lengths = geo.chiral_2_mult_unit_cell(nb_unit_cells, nb_grid_pts, a, radius_out, radius_inn,
                                                thickness, alpha=angle_mat)
    return mask, lengths

def test_convergences():
    ### ----- Preparation ----- ###
    # Parameters
    N_list = [16, 20]
    #N_list = [60, 80, 100, 120, 140, 160, 180, 200]
    folder = f'mesh_refinement_mpi{MPI.COMM_WORLD.size}/'

    if (MPI.COMM_WORLD.rank == 0):
        # Create or clear folder
        if not os.path.exists(folder):
            os.makedirs(folder)
        else:
            shutil.rmtree(folder)
            os.makedirs(folder)

        # Copy this file into the folder
        helper = 'cp test_mesh_refinements.py ' + folder
        os.system(helper)

    # Print some informations
    output = folder + 'general_output.txt'
    if MPI.COMM_WORLD.rank == 0:
        with open(output, 'w') as f:
            print(f'--- Mesh refinement: Parameters ---', file=f)
            print(f'MPI size = {MPI.COMM_WORLD.size}', file=f)
            print(f'List of nb_grid_pts:', N_list, file=f)
            print(µ.version.info(), file=f)

    ### ----- Tests for rotation ----- ###
    # Square beam
    name_geometry = 'square_beam'
    test_convergence_rotation_worker(create_square_beam, N_list, folder,
                                     name_geometry)

    # Cylinder
    name_geometry = 'cylinder'
    test_convergence_rotation_worker(create_cylinder, N_list, folder,
                                     name_geometry)

    # Chiral 1
    name_geometry = 'chiral_1'
    test_convergence_rotation_worker(create_chiral_1, N_list, folder, name_geometry)

    # Chiral 2 (1x1x1 unit cells)
    name_geometry = 'chiral_2'
    test_convergence_rotation_worker(create_chiral_2, N_list, folder, name_geometry)

    ### ----- Tests for traction ----- ###
    # Square beam
    name_geometry = 'square_beam'
    test_convergence_traction_worker(create_square_beam, N_list, folder,
                                     name_geometry)

    # Cylinder
    name_geometry = 'cylinder'
    test_convergence_traction_worker(create_cylinder, N_list, folder,
                                     name_geometry)

    # Chiral 1
    name_geometry = 'chiral_1'
    test_convergence_traction_worker(create_chiral_1, N_list, folder, name_geometry)

    # Chiral 2 (1x1x1 unit cells)
    name_geometry = 'chiral_2'
    test_convergence_traction_worker(create_chiral_2, N_list, folder, name_geometry)

def plot_mesh_refinements():
    ### ----- Parameters ----- ###
    # Data
    folder = 'results_nemo/mesh_refinement_mpi126/'
    names = ['square_beam', 'cylinder', 'chiral_1', 'chiral_2']
    loads = ['traction', 'rotation']
    ylabels = ['average_stress_xx', 'average force_z']

    # For plotting
    figsize = (8, 7)
    gridspec_kw = {'wspace': 0.3, 'hspace': 0.3}
    fontsize_large = 15

    ### ----- Plot ----- ###
    for i_load, load in enumerate(loads):
        # Prepare figure
        fig, axes = plt.subplots(2, 2, figsize=figsize, gridspec_kw=gridspec_kw)
        fig.suptitle(load, fontsize=fontsize_large)
        for i, ax in enumerate(axes.flatten()):
            ax.set_title(names[i])
            ax.set_xlabel('nb_grid_pts')
            ax.set_ylabel(ylabels[i_load])

            # Read data
            name = folder + names[i] + '_' + load + '_data.txt'
            data = np.loadtxt(name, skiprows=1)

            # Plot data
            ax.plot(data[:, 0], data[:, 1], marker='x')

        plt.show()
        name = 'convergence_' + load + '.pdf'
        fig.savefig(name, bbox_inches='tight')
        plt.close(fig)

if __name__ == "__main__":
    test_convergences()
    #plot_mesh_refinements()

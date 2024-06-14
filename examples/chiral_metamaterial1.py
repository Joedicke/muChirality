"""
@file   test_chiral_metamaterial2.py

@author Indre  Joedicke <indre.joedicke@imtek.uni-freiburg.de>

@date   14 June 2024

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


def plot_comp_cylinder(folder, angle_mat, angle_mat2):
    # Read data
    name = folder + 'data.txt'
    twists = np.loadtxt(name, skiprows=1, max_rows=1)
    forces_cyl = np.loadtxt(name, skiprows=3, max_rows=1)
    forces_chiral = np.loadtxt(name, skiprows=5, max_rows=1)
    forces_chiral2 = np.loadtxt(name, skiprows=7, max_rows=1)

    # Prepare figure
    fig, ax = plt.subplots()
    ax.set_xlabel('Twist')
    ax.set_ylabel('Force in z-direction')

    # Plot
    ax.plot(twists, forces_cyl, color='red', marker='x', label='cylinder')
    ax.plot(twists, forces_chiral, color='blue', marker='x',
            label=f'chiral mat (angle={angle_mat}')
    ax.plot(twists, forces_chiral2, color='blue', marker='x',
            label=f'chiral mat (angle={angle_mat2}')
    ax.legend()

    # Save plot
    name = folder + 'plot_data.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)

###################################################################################################
###################################################################################################
###################################################################################################
def calculation_with_cylinder():
    t = time()
    ### ----- Parameter definitions ----- ###
    # Geometry
    lengths = [1, 1, 1]
    radius = 0.4
    thickness = 0.1
    angle_mat = 0.4
    angle_mat2 = 0.2

    hx = lengths[0] / nb_grid_pts[0]
    hy = lengths[1] / nb_grid_pts[1]
    hz = lengths[2] / nb_grid_pts[2]

    # Discretization
    Nxyz = 30
    nb_grid_pts = [Nxyz, Nxyz, Nxyz]
    gradient, weights = µ.linear_finite_elements.gradient_3d_5tet

    # Material
    Young = 100
    Poisson = 0

    # Loading
    #twists = [-0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, -0.2]
    twists = [-0.1, 0.1, 0.2]
    delta_F = np.zeros((3, 3))
    x_rot_axis = lengths[0] / 2
    y_rot_axis = lengths[1] / 2

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
    folder = f'results/chiral1/comp_cylinder_Nxyz={Nxyz}/'
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
    forces_chiral2 = np.empty(len(twists))
    forces_cyl = np.empty(len(twists))

    for i_twist, twist in enumerate(twists):
        if (MPI.COMM_WORLD.rank == 0):
            message = f'Test {i_twist+1} of {len(twists)}'
            print(message)

        ### ----- Calculation (cylinder) ----- ###
        # Define geometry
        mask = geo.cylinder(nb_grid_pts, lengths_cyl, radius_out)

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

        ### ----- Calculation (chiral - angle 1) ----- ###
        # Define geometry
        mask = geo.chiral_metamaterial(nb_grid_pts, lengths, radius, thickness,
                                       alpha=angle_mat)

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
            name = folder + f'alpha={angle_mat}_twist={twist}_deformed_geometry.xdmf'
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

        ### ----- Calculation (chiral - angle 2) ----- ###
        # Define geometry
        mask = geo.chiral_metamaterial(nb_grid_pts, lengths, radius, thickness,
                                       alpha=angle_mat2)

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
        forces_chiral2[i_twist] = force_z

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
            name = folder + f'alpha={angle_mat2}_twist={twist}_deformed_geometry.xdmf'
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
            print(f'Force in z-direction (chiral mat - angle={angle_mat})', file=f)
            np.savetxt(f, forces_chiral, newline=' ')
            print('', file=f)
            print(f'Force in z-direction (chiral mat - angle={angle_mat2})', file=f)
            np.savetxt(f, forces_chiral2, newline=' ')

    # Plot results
    if MPI.COMM_WORLD.rank == 0:
        plot_comp_cylinder(folder, angle_mat, angle_mat2)

if __name__ == "__main__":
    calculation_with_cylinder()

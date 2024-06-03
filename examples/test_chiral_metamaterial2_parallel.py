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

import numpy as np
import matplotlib.pyplot as plt
from time import time

import muSpectre as µ
from NuMPI import MPI
from NuMPI.Tools import Reduction

from muChirality.EigenStrainTorsion import EigenStrain
from muChirality.Geometries import cylinder
from muChirality.Geometries import chiral_metamaterial_2
from muChirality.Geometries import chiral_2_mult_unit_cell
from muChirality.CalculationsTorsion import calculations
from muSpectre.gradient_integration import get_complemented_positions

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

# Discretization
dim = 3
nb_grid_pts_uc = [30, 30, 30]
gradient, weights = µ.linear_finite_elements.gradient_3d_5tet

if MPI.COMM_WORLD.rank == 0:
    print(f'--- Parameters ---')
    print(f'MPI size = {MPI.COMM_WORLD.size}')
    helper = f'nb_grid_pts_uc = {nb_grid_pts_uc[0]}x{nb_grid_pts_uc[1]}'
    helper += f'x{nb_grid_pts_uc[2]}'
    print(helper)
    print('List nb_unit_cells =', N_uc_list)
    print('Restarted:', restart)
    print()

# Material
Young = 2600
Poisson = 0.4

# Loading
angle_rot = 0.2
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
name = f'chiral_mat_2_Nxyz={nb_grid_pts_uc[0]}_data.txt'

if (MPI.COMM_WORLD.rank == 0) and (not restart):
    with open(name, 'w') as f:
        title = 'Number of unit cells in RVE    Effective Youngs modulus in'
        title += ' z direction    Twist per strain (degree/%)'
        print(title, file=f)

# What is calculated
E_eff_z_list = np.empty(len(N_uc_list))
twist_per_strain_list = np.empty(len(N_uc_list))
for index, N_uc in enumerate(N_uc_list):
    ### ----- Define geometry ----- ###
    mask, lengths =\
            chiral_2_mult_unit_cell([N_uc, N_uc], nb_grid_pts_uc,
                                    a, radius_out, radius_inn,
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
    E_eff_z_list[index] = E_eff_z
    if MPI.COMM_WORLD.rank == 0:
        print(f'nb_unit_cells = {N_uc}')
        print(f'E_eff_z = {E_eff_z}')

    ### ----- Calculate Twist per Strain ----- ###
    solver = µ.solvers.KrylovSolverCG(cell, cg_tol, maxiter, verbose)

    # Initialize Eigen class
    eigen_class = EigenStrain(cell.pixels, angle_rot, lengths, nb_grid_pts,
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
    twist_in_degree = np.arctan(angle_rot * lengths[2]) / np.pi * 180
    strain_zz_in_percent = force / stiff_z * 100

    twist_per_strain = twist_in_degree / strain_zz_in_percent
    twist_per_strain_list[index] = twist_per_strain

    if MPI.COMM_WORLD.rank == 0:
        print('Twist/strain (degree/%):', twist_per_strain)
        print(f'Finished calculation {index+1} of {len(N_uc_list)}')

    ### ----- Save results ----- ###
    if MPI.COMM_WORLD.rank == 0:
        with open(name, 'a') as f:
            np.savetxt(f, [N_uc, E_eff_z, twist_per_strain], newline=' ')
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
        helper = - angle_rot * np.einsum('i,j->ij', y-y_rot_axis, z)
        x_displ += helper[None, :, :]
        helper = angle_rot * np.einsum('i,j->ij', x-x_rot_axis, z)
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
        name = f'chiral_mat2_Nuc={N_uc}_Nxyz={nb_grid_pts_uc[0]}_deformed.xdmf'
        µ.linear_finite_elements.write_3d(name, cell, cell_data=cell_data, point_data=point_data,
                                          F0=F0, displacement_field=False)

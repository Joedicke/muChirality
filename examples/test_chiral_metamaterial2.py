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

from muChirality.EigenStrainTorsion import EigenStrain
from muChirality.Geometries import cylinder
from muChirality.Geometries import chiral_metamaterial_2
from muChirality.CalculationsTorsion import calculations
from muSpectre.gradient_integration import get_complemented_positions

def comparison_cylinder(nb_grid_pts):
    t = time()
    ### ----- Parameter definitions ----- ###
    # Geometry
    a = 1
    thickness = 0.06
    lengths = [a + thickness, a + thickness, a]
    radius_out = 0.4
    radius_inn = radius_out - thickness
    angle_mat = np.pi * 35 / 180

    # Discretization
    gradient, weights = µ.linear_finite_elements.gradient_3d_5tet

    # Material
    Young = 100
    Poisson = 0

    # Loading
    angles_rot = [-0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, -0.2]
    #angles_rot = [-0.1, 0.1, 0.2]
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

    # For saving
    folder = 'examples/plots/'
    F0 = np.eye(3)

    ### ----- Calculations ----- ###
    forces_chiral = np.empty(len(angles_rot))
    forces_cyl = np.empty(len(angles_rot))

    for i_angle, angle_rot in enumerate(angles_rot):
        message = f'Test {i_angle+1} of {len(angles_rot)}'
        print(message)

        ### ----- Calculation (cylinder) ----- ###
        # Define geometry
        mask = cylinder(nb_grid_pts, lengths, radius_out)

        # muSpectre cell initialization
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
        eigen_class = EigenStrain(cell.pixels, angle_rot, lengths, nb_grid_pts,
                                  x_rot_axis, y_rot_axis)

        # Solving
        res = µ.solvers.newton_cg(cell, delta_F, solver, newton_tol, equil_tol,
                                  verbose, μ.solvers.IsStrainInitialised.No,
                                  µ.StoreNativeStress.No, eigen_class.eigen_strain_func)
        shape = (3, 3, 5, *nb_grid_pts)
        stress = res.stress.reshape(shape, order='F')
        strain = res.grad.reshape(shape, order='F')

        # Calculate force
        pos, displ, force = calculations(strain, stress, cell,
                                         eigen_class, detailed=False)
        forces_cyl[i_angle] = force

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
        mask = chiral_metamaterial_2(nb_grid_pts, lengths, radius_out, radius_inn,
                                     thickness, alpha=angle_mat)

        # muSpectre cell initialization
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
        eigen_class = EigenStrain(cell.pixels, angle_rot, lengths, nb_grid_pts,
                                  x_rot_axis, y_rot_axis)

        # Solving
        res = µ.solvers.newton_cg(cell, delta_F, solver, newton_tol, equil_tol,
                                  verbose, μ.solvers.IsStrainInitialised.No,
                                  µ.StoreNativeStress.No, eigen_class.eigen_strain_func)
        shape = (3, 3, 5, *nb_grid_pts)
        stress = res.stress.reshape(shape, order='F')
        strain = res.grad.reshape(shape, order='F')

        # Calculate force
        pos, displ, force = calculations(strain, stress, cell,
                                         eigen_class, detailed=False)
        forces_chiral[i_angle] = force

        #twist_per_strain = angle
        #print(f'twist / strain = {twist_per_strain}')

        # Save one state for paraview
        if i_angle is (len(angles_rot)-1):
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
            name = folder + f'nb_grid_pts={nb_grid_pts[0]}'
            name += f'x{nb_grid_pts[1]}x{nb_grid_pts[2]}.xdmf'
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


    ### ----- Plot forces ----- ###
    fig, ax = plt.subplots()
    ax.set_xlabel('Angle')
    ax.set_ylabel('Force z')
    ax.plot(angles_rot, forces_cyl, label='Cylinder', marker='x')
    ax.plot(angles_rot, forces_chiral, label='metamat', marker='x')
    ax.legend()
    name = folder + f'forces_nb_grid_pts={nb_grid_pts[0]}x{nb_grid_pts[1]}x{nb_grid_pts[2]}.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)

    t = time() - t
    print()
    if t > 60 * 60:
        t = t / 60 / 60
        print(f'Time = {t:.1f} h')
    elif t > 60:
        t = t / 60
        print(f'Time = {t:.1f} min')
    else:
        print(f'Time = {t:.1f} s')

def comparison_paper_hom_E(nb_grid_pts):
    print(f'--- nb_grid_pts = {nb_grid_pts[0]}x{nb_grid_pts[1]}x{nb_grid_pts[2]} ---')
    t = time()
    ### ----- Parameter definitions ----- ###
    dim = 3

    # Geometry
    a = 0.5
    thickness = 0.06 * a
    lengths = [a + thickness, a + thickness, a]
    radius_out = 0.4 * a
    radius_inn = 0.34 * a
    angle_mat = np.pi * 35 / 180

    # Discretization
    gradient, weights = µ.linear_finite_elements.gradient_3d_5tet

    # Material
    Young = 2600
    Poisson = 0.4

    # Loading
    angle_rot = 0.2
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

    # For saving
    folder = 'examples/plots/'
    F0 = np.eye(3)

    ### ----- Calculation of homogenized stiffness in z-direction ----- ###
    delta_eps = np.zeros((dim, dim))
    delta_eps[2, 2] = 0.01

    # Define geometry
    mask = chiral_metamaterial_2(nb_grid_pts, lengths, radius_out, radius_inn,
                                 thickness, alpha=angle_mat)

    # muSpectre cell initialization
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

    # Solve muSpectre
    result = µ.solvers.newton_cg(cell, delta_eps, solver, newton_tol,
                                 equil_tol, verbose)
    shape = (dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts)
    stress = result.stress.reshape(shape, order='F')

    # Force in z-direction
    hx = lengths[0] / nb_grid_pts[0]
    hy = lengths[1] / nb_grid_pts[1]
    hz = lengths[2] / nb_grid_pts[2]
    force_z = hx * hy * hz / 6 * np.sum(stress[2, 2])
    force_z += hx * hy * hz / 6 * np.sum(stress[2, 2, 0])
    force_z = force_z / lengths[2]

    # Stiffness in z-direction
    stiff_z = force_z / delta_eps[2, 2]

    E_eff_z = stiff_z / lengths[0] / lengths[1]
    print(f'Stiffness in z-dir = {stiff_z}')
    print(f'eff E-mod in z-dir = {E_eff_z}')

    ### ----- Calculate Torsion ----- ###
    solver = µ.solvers.KrylovSolverCG(cell, cg_tol, maxiter, verbose)

    # Initialize Eigen class
    eigen_class = EigenStrain(cell.pixels, angle_rot, lengths, nb_grid_pts,
                              x_rot_axis, y_rot_axis)

    # Solving
    res = µ.solvers.newton_cg(cell, delta_F, solver, newton_tol, equil_tol,
                              verbose, μ.solvers.IsStrainInitialised.No,
                                  µ.StoreNativeStress.No, eigen_class.eigen_strain_func)
    shape = (3, 3, 5, *nb_grid_pts)
    stress = res.stress.reshape(shape, order='F')
    strain = res.grad.reshape(shape, order='F')

    # Calculate force
    pos, displ, force = calculations(strain, stress, cell,
                                     eigen_class, detailed=False)
    print('Force calculated.')

    # Comparison with paper
    twist_in_degree = np.arctan(angle_rot * lengths[2]) / np.pi * 180
    strain_zz_in_percent = force / stiff_z * 100
    print('Strain zz (%):', strain_zz_in_percent)

    twist_per_strain = twist_in_degree / strain_zz_in_percent
    print('Twist/strain (degree/%):', twist_per_strain)
    print('Force_z:', force)

    # Print time
    t = time() - t
    print()
    if t > 60 * 60:
        t = t / 60 / 60
        print(f'Time = {t:.1f} h')
    elif t > 60:
        t = t / 60
        print(f'Time = {t:.1f} min')
    else:
        print(f'Time = {t:.1f} s')


if __name__ == "__main__":
    N = 40
    nb_grid_pts = [N, N, N]
    #comparison_cylinder(nb_grid_pts)
    comparison_paper_hom_E(nb_grid_pts)

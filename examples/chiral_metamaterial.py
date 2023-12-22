"""
@file   test_chiral_metamaterial.py

@author Indre  Joedicke <indre.joedicke@imtek.uni-freiburg.de>

@date   18 Aug 2023

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

# Default path of the library
sys.path.insert(0, os.path.join(os.getcwd(), "../muspectre/meson-build-release/language_bindings/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../muspectre/meson-build-release/language_bindings/libmufft/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../muspectre/meson-build-release/language_bindings/libmugrid/python"))

import numpy as np
import matplotlib.pyplot as plt
from time import time

import muSpectre as µ

from muChirality.EigenStrainTorsion import EigenStrain
from muChirality.Geometries import cylinder
from muChirality.Geometries import chiral_metamaterial
#from muChirality.Geometries import chiral_metamaterial_2
from muChirality.CalculationsTorsion import calculations
from muChirality.Plotting import plot_2D_chiral
#from muChirality.Plotting import plot_2D_metamaterial_2


def comparison_cylinder(nb_grid_pts):
    t = time()
    ### ----- Parameter definitions ----- ###
    # Geometry
    lengths = [1, 1, 1]
    radius = 0.4
    thickness = 0.1
    angle_mat = 0.4
    angle_mat2 = 0.2

    # Discretization
    gradient, weights = µ.linear_finite_elements.gradient_3d_5tet

    # Material
    Young = 100
    Poisson = 0

    # Loading
    #angles_rot = [-0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, -0.2]
    angles_rot = [-0.1, 0.1, 0.2]
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
    folder = 'plots_chiral_material/'
    F0 = np.eye(3)

    ### ----- Calculations ----- ###
    forces_chiral = np.empty(len(angles_rot))
    forces_chiral2 = np.empty(len(angles_rot))
    forces_cyl = np.empty(len(angles_rot))

    for i_angle, angle_rot in enumerate(angles_rot):
        message = f'Test {i_angle+1} of {len(angles_rot)}'
        print(message)

        ### ----- Calculation (cylinder) ----- ###
        # Define geometry
        mask = cylinder(nb_grid_pts, lengths, radius)

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
        mask = chiral_metamaterial(nb_grid_pts, lengths, radius, thickness,
                                   alpha=angle_mat)

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
            name = folder + f'angle={angle_mat}/nb_grid_pts={nb_grid_pts[0]}'
            name += f'x{nb_grid_pts[1]}x{nb_grid_pts[2]}.xdmf'
            µ.linear_finite_elements.write_3d(name, cell, cell_data=cell_data, point_data=point_data,
                                              F0=F0, displacement_field=True)
            name = folder + f'angle={angle_mat}/nb_grid_pts={nb_grid_pts[0]}'
            name += f'x{nb_grid_pts[1]}x{nb_grid_pts[2]}.xmf'
            µ.linear_finite_elements.write_3d(name, cell, cell_data=cell_data, point_data=point_data,
                                              F0=F0, displacement_field=False)
            name = folder + f'angle={angle_mat}/nb_grid_pts={nb_grid_pts[0]}'
            name += f'x{nb_grid_pts[1]}x{nb_grid_pts[2]}.vtu'
            µ.linear_finite_elements.write_3d(name, cell, cell_data=cell_data, point_data=point_data,
                                              F0=F0, displacement_field=False)

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

        ### ----- Calculation (chiral 2) ----- ###
        # Define geometry
        mask2 = chiral_metamaterial(nb_grid_pts, lengths, radius, thickness,
                                    alpha=angle_mat2)

        # muSpectre cell initialization
        cell2 = µ.Cell(nb_grid_pts, lengths, formulation, gradient, weights)
        mat = µ.material.MaterialLinearElastic1_3d.make(cell2, "hard", Young, Poisson)
        vac = µ.material.MaterialLinearElastic1_3d.make(cell2, "vacuum", 0, 0)
        mask2 = mask2.flatten(order='F')
        for pixel_id, pixel in cell2.pixels.enumerate():
            if mask2[pixel_id] == 1:
                mat.add_pixel(pixel_id)
            else:
                vac.add_pixel(pixel_id)

        cell2.initialise()
        solver = µ.solvers.KrylovSolverCG(cell2, cg_tol, maxiter, verbose)

        # Initialize Eigen class
        eigen_class2 = EigenStrain(cell2.pixels, angle_rot, lengths, nb_grid_pts,
                                   x_rot_axis, y_rot_axis)

        # Solving
        res = µ.solvers.newton_cg(cell2, delta_F, solver, newton_tol, equil_tol,
                                  verbose, μ.solvers.IsStrainInitialised.No,
                                  µ.StoreNativeStress.No, eigen_class2.eigen_strain_func)
        shape = (3, 3, 5, *nb_grid_pts)
        stress2 = res.stress.reshape(shape, order='F')
        strain2 = res.grad.reshape(shape, order='F')

        # Calculate force
        pos2, displ2, force2 = calculations(strain2, stress2, cell2,
                                            eigen_class2, detailed=False)
        forces_chiral2[i_angle] = force2

        # Save one state for paraview
        if i_angle is (len(angles_rot)-1):
            cell_data = {}
            material = np.stack((mask2, mask2, mask2, mask2, mask2), axis=0)
            material = material.flatten(order='F')
            material = material.reshape((1, -1))
            stress = stress2.reshape((3, 3, -1), order='F').T.swapaxes(1, 2)
            stress = stress.reshape((1, -1, 3, 3)).copy()
            strain = strain2.reshape((3, 3, -1), order='F').T.swapaxes(1, 2)
            strain = strain.reshape((1, -1, 3, 3)).copy()
            cell_data = {"material": material, "stress_field": stress, "strain_field": strain}
            point_data = {"displ": displ2.reshape((3, -1), order='F').T}
            name = folder + f'angle={angle_mat2}/nb_grid_pts={nb_grid_pts[0]}'
            name += f'x{nb_grid_pts[1]}x{nb_grid_pts[2]}.xdmf'
            µ.linear_finite_elements.write_3d(name, cell2, cell_data=cell_data, point_data=point_data,
                                              F0=F0, displacement_field=True)

        # Delete parameters
        del cell2
        del mat
        del vac
        del mask2
        del solver
        del eigen_class2
        del res
        del stress2
        del strain2


    ### ----- Plot forces ----- ###
    fig, ax = plt.subplots()
    ax.set_xlabel('Angle')
    ax.set_ylabel('Force z')
    ax.plot(angles_rot, forces_cyl, label='Cylinder', marker='x')
    ax.plot(angles_rot, forces_chiral, label=f'metamat angle={angle_mat}', marker='x')
    ax.plot(angles_rot, forces_chiral2, label=f'metamat angle={angle_mat2}', marker='x')
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

def geometry_plots(nb_grid_pts, saving=True, showing=False):
    ### ----- Parameter definitions ----- ###
    # Geometry
    lengths = [1, 1, 1]
    radius = 0.4
    thickness = 0.1
    angle_mat = 0.4
    angle_mat2 = 0.2

    # For saving
    folder = 'plots_chiral_material/'

    ### ----- Chiral metamaterial 1 ----- ###
    fig1, fig2, fig3 = plot_2D_metamaterial(nb_grid_pts, lengths, radius,
                                            thickness, angle_mat)

    # Save plots
    if saving:
        name = folder + f'angle={angle_mat}/nb_grid_pts='
        name += f'{nb_grid_pts[0]}x{nb_grid_pts[1]}x{nb_grid_pts[2]}'
        name += '_projection_on_xz.pdf'
        fig1.savefig(name, bbox_inches='tight')

        name = folder + f'angle={angle_mat}/nb_grid_pts='
        name += f'{nb_grid_pts[0]}x{nb_grid_pts[1]}x{nb_grid_pts[2]}'
        name += '_projection_on_yz.pdf'
        fig2.savefig(name, bbox_inches='tight')

        name = folder + f'angle={angle_mat}/nb_grid_pts='
        name += f'{nb_grid_pts[0]}x{nb_grid_pts[1]}x{nb_grid_pts[2]}'
        name += '_projection_on_xy.pdf'
        fig3.savefig(name, bbox_inches='tight')

    if showing:
        plt.show()
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)

    ### ----- Chiral metamaterial 2 ----- ###
    # Create plots
    fig1, fig2, fig3 = plot_2D_metamaterial(nb_grid_pts, lengths, radius,
                                            thickness, angle_mat2)

    # Save plots
    if saving:
        name = folder + f'angle={angle_mat2}/nb_grid_pts='
        name += f'{nb_grid_pts[0]}x{nb_grid_pts[1]}x{nb_grid_pts[2]}'
        name += '_projection_on_xz.pdf'
        fig1.savefig(name, bbox_inches='tight')

        name = folder + f'angle={angle_mat2}/nb_grid_pts='
        name += f'{nb_grid_pts[0]}x{nb_grid_pts[1]}x{nb_grid_pts[2]}'
        name += '_projection_on_yz.pdf'
        fig2.savefig(name, bbox_inches='tight')

        name = folder + f'angle={angle_mat2}/nb_grid_pts='
        name += f'{nb_grid_pts[0]}x{nb_grid_pts[1]}x{nb_grid_pts[2]}'
        name += '_projection_on_xy.pdf'
        fig3.savefig(name, bbox_inches='tight')

    if showing:
        plt.show()
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)

if __name__ == "__main__":
    nb_grid_pts = [90, 90, 90]
    #nb_grid_pts = [80, 80, 80]
    #nb_grid_pts = [70, 70, 70]
    #nb_grid_pts = [30, 30, 30]
    geometry_plots(nb_grid_pts, saving=True, showing=False)
    comparison_cylinder(nb_grid_pts)
    #comparison_cylinder2(nb_grid_pts)
    #comparison_paper2_relaxation(nb_grid_pts)

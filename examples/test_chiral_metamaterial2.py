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

# Default path of the library
sys.path.insert(0, os.path.join(os.getcwd(), "./muspectre/meson-build-release/language_bindings/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "./muspectre/meson-build-release/language_bindings/libmufft/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "./muspectre/meson-build-release/language_bindings/libmugrid/python"))

import numpy as np
import matplotlib.pyplot as plt
from time import time

import muSpectre as µ

from EigenStrainTorsion import EigenStrain
from Geometries import cylinder
#from Geometries import chiral_metamaterial
from Geometries import chiral_metamaterial_2
from CalculationsTorsion import calculations
#from test_geometries import plot_2D_metamaterial
from test_geometries import plot_2D_metamaterial_2

def calculate_hom_stiffness_3d_small_strain(cell, weights):
    #print(dir(cell))
    ### ----- Parameter definitions ----- ###
    # From cell
    dim = cell.dim
    formulation = cell.formulation

    if dim != 3:
        message = 'This stiffness function only works for 3D cases.'
        raise ValueError(message)
    if formulation != µ.Formulation.small_strain:
        message = 'This stiffness function only works for small strain cases.'
        raise ValueError(message)

    # muSpectre solver parameters
    newton_tol       = 1e-7
    cg_tol           = 1e-7 # tolerance for cg algo
    equil_tol        = 1e-7 # tolerance for equilibrium
    maxiter          = 10000
    verbose          = µ.Verbosity.Silent

    # Loading
    strains = [np.zeros((dim, dim)), np.zeros((dim, dim)), np.zeros((dim, dim)),
               np.zeros((dim, dim)), np.zeros((dim, dim)), np.zeros((dim, dim))]
    strains[0][0, 0] = 0.01
    strains[1][1, 1] = 0.01
    strains[2][2, 2] = 0.01
    strains[3][0, 1] = strains[3][1, 0] = 0.01
    strains[4][0, 2] = strains[4][2, 0] = 0.01
    strains[5][1, 2] = strains[5][2, 1] = 0.01

    ### ----- muSpectre calculation ----- ###
    shape = (dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts)
    stresses = []

    # Initialization
    cell.initialise()
    solver = µ.solvers.KrylovSolverCG(cell, cg_tol, maxiter, verbose)

    # Solving
    for i, strain in enumerate(strains):
        result = µ.solvers.newton_cg(cell, [strain], solver, newton_tol,
                                     equil_tol, verbose)
        stress = result[0].stress.reshape(shape, order='F')
        stress = np.average(stress, axis=(3, 4, 5))
        stress = np.average(stress, axis=2, weights = weights)
        stresses.append(stress)
        print(f'Finished {i+1}.th case')
        del result

    ### ----- Homogenized stiffness ----- ###
    # Average stresses
    #for result in results:

    # Numerical homogenized stiffness
    stiff = np.empty((dim, dim, dim, dim))
    stiff[:, :, 0, 0] = stresses[0] / strains[0][0, 0]
    stiff[:, :, 1, 1] = stresses[1] / strains[1][1, 1]
    stiff[:, :, 2, 2] = stresses[2] / strains[2][2, 2]
    stiff[:, :, 0, 1] = stiff[:, :, 1, 0] = stresses[3] / strains[3][0, 1] / 2
    stiff[:, :, 0, 2] = stiff[:, :, 2, 0] = stresses[4] / strains[4][0, 2] / 2
    stiff[:, :, 1, 2] = stiff[:, :, 2, 1] = stresses[5] / strains[5][1, 2] / 2

    # Delete parameters
    #del cell
    #del mat
    #del vac
    #del mask
    del solver
    #del eigen_class
    #del results
    #del stress
    #del strain

    return stiff

def test_hom_stiffness():
    ### ----- Parameter definitions ----- ###
    # Geometry
    lengths = [1, 1, 1]
    dim = len(lengths)

    # Discretization
    nb_grid_pts = [3, 3, 3]
    gradient, weights = µ.linear_finite_elements.gradient_3d_5tet

    # Material
    Young = 2600
    Poisson = 0.4

    # Formulation
    formulation = µ.Formulation.small_strain

    # muSpectre solver parameters
    newton_tol       = 1e-7
    cg_tol           = 1e-7 # tolerance for cg algo
    equil_tol        = 1e-7 # tolerance for equilibrium
    maxiter          = 10000
    verbose          = µ.Verbosity.Silent

    # Loading
    strain = np.full((dim, dim), 0.1)

    ### ----- Calculation ----- ###
    mask = np.ones(nb_grid_pts).flatten(order='F')

    # Initialization of material
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
    stiff = calculate_hom_stiffness_3d_small_strain(cell, weights)

    ### ----- Comparison ----- ###
    # Analytical solution (for homogenous material)
    solver = µ.solvers.KrylovSolverCG(cell, cg_tol, maxiter, verbose)
    result = µ.solvers.newton_cg(cell, strain, solver, newton_tol,
                                 equil_tol, verbose)

    ### ----- Homogenized stiffness ----- ###
    # Average stresses
    shape = (dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts)
    stress, stiff_muSpectre = cell.evaluate_stress_tangent(result.grad.reshape(shape, order='F'))
    stiff_muSpectre = stiff_muSpectre[:, :, :, :, 0, 0, 0, 0]
    #stress = np.average(stress, axis=(3, 4, 5))
    #stress = np.average(stress, axis=2, weights = weights)
    #stress2 = np.einsum('ijkl, kl', stiff_muSpectre, strain)
    #diff = np.linalg.norm(stress - stress2)
    #print(diff)

    # Difference to calculated stiffness
    diff = stiff - stiff_muSpectre
    diff = np.linalg.norm(diff)
    print(f'Difference between analytical and numerical stiffness = {diff}')

def geometry_plots(nb_grid_pts, saving=True, showing=False):
    ### ----- Parameter definitions ----- ###
    # Geometry
    a = 1
    thickness = 0.06
    lengths = [a + thickness, a + thickness, a]
    radius_out = 0.4
    radius_inn = radius_out - thickness
    angle_mat = np.pi * 35 / 180

    # For saving
    folder = 'plots_chiral_material_2/'

    ### ----- Chiral metamaterial 1 ----- ###
    figures = plot_2D_metamaterial_2(nb_grid_pts, lengths, radius_out, radius_inn,
                                     thickness, angle_mat)
    fig1, fig2, fig3, fig4, fig5, fig6 = figures

    # Save plots
    if saving:
        name = folder + f'{nb_grid_pts[0]}x{nb_grid_pts[1]}x{nb_grid_pts[2]}'
        name += '_projection_on_xz.pdf'
        fig1.savefig(name, bbox_inches='tight')

        name = folder + f'{nb_grid_pts[0]}x{nb_grid_pts[1]}x{nb_grid_pts[2]}'
        name += '_projection_on_yz.pdf'
        fig2.savefig(name, bbox_inches='tight')

        name = folder + f'{nb_grid_pts[0]}x{nb_grid_pts[1]}x{nb_grid_pts[2]}'
        name += '_projection_on_xy.pdf'
        fig3.savefig(name, bbox_inches='tight')

        name = folder + f'{nb_grid_pts[0]}x{nb_grid_pts[1]}x{nb_grid_pts[2]}'
        name += '_xz_plane.pdf'
        fig4.savefig(name, bbox_inches='tight')

        name = folder + f'{nb_grid_pts[0]}x{nb_grid_pts[1]}x{nb_grid_pts[2]}'
        name += '_yz_plane.pdf'
        fig5.savefig(name, bbox_inches='tight')

        name = folder + f'{nb_grid_pts[0]}x{nb_grid_pts[1]}x{nb_grid_pts[2]}'
        name += '_xy_plane.pdf'
        fig6.savefig(name, bbox_inches='tight')

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    #plt.close(fig4)
    #plt.close(fig5)
    #plt.close(fig6)
    if showing:
        plt.show()

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
    folder = 'plots_chiral_material_2/'
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

def comparison_paper_relaxation(nb_grid_pts):
    t = time()
    ### ----- Parameter definitions ----- ###
    # Geometry
    a = 0.5 #1
    thickness = 0.06 * a
    lengths = [a + thickness, a + thickness, a]
    radius_out = 0.4 * a
    radius_inn = 0.34 * a
    angle_mat = np.pi * 35 / 180

    # Discretization
    gradient, weights = µ.linear_finite_elements.gradient_3d_5tet

    # Material
    #Young = 100
    #Poisson = 0
    Young = 2600 #2.6
    Poisson = 0.4

    # Loading
    angle_rot = 0.2
    x_rot_axis = lengths[0] / 2
    y_rot_axis = lengths[1] / 2
    delta_F = np.zeros((3, 3))
    twist_in_degree = np.arctan(angle_rot * lengths[2]) / np.pi * 180
    loading = twist_in_degree / 100 / (0.5)
    delta_F[2, 2] = loading
    print('Strain =', loading)

    # Formulation
    formulation = µ.Formulation.small_strain

    # muSpectre solver parameters
    newton_tol       = 1e-7
    cg_tol           = 1e-7 # tolerance for cg algo
    equil_tol        = 1e-7 # tolerance for equilibrium
    maxiter          = 10000
    verbose          = µ.Verbosity.Silent

    # For saving
    folder = 'plots_chiral_material_2/'
    F0 = np.eye(3)

    ### ----- Calculations ----- ###
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

    # Comparison with paper
    twist_in_degree = np.arctan(angle_rot * lengths[2]) / np.pi * 180
    nb_pixels = np.prod(nb_grid_pts)
    nb_pixels_with_material = np.sum(mask)
    strain_zz_in_percent = delta_F[2, 2] * 100
    print('pixels with mat =', nb_pixels_with_material)

    twist_per_strain = twist_in_degree / strain_zz_in_percent
    print('Twist/strain (degree/%):', twist_per_strain)
    print('Force_z:', force)
    stress_zz = force * lengths[2]
    #force_z = hx * hy * hz / 6 * np.sum(stress[2, 2])
    #force_z += hx * hy * hz / 6 * np.sum(stress[2, 2, 0])
    #force_z = force_z / lengths[2]
    print('Total stress_zz:', stress_zz)
    print('Average stress_zz:', stress_zz / nb_pixels)
    print('Average stress_zz in material:', stress_zz / nb_pixels_with_material)

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


def comparison_paper_stiffness(nb_grid_pts):
    print(f'--- nb_grid_pts = {nb_grid_pts[0]}x{nb_grid_pts[1]}x{nb_grid_pts[2]} ---')
    t = time()
    ### ----- Parameter definitions ----- ###
    dim = 3

    # Geometry
    a = 0.5 #1
    thickness = 0.06 * a
    lengths = [a + thickness, a + thickness, a]
    radius_out = 0.4 * a
    radius_inn = 0.34 * a
    angle_mat = np.pi * 35 / 180

    # Discretization
    gradient, weights = µ.linear_finite_elements.gradient_3d_5tet

    # Material
    #Young = 100
    #Poisson = 0
    Young = 2600 #2.6
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
    folder = 'plots_chiral_material_2/'
    F0 = np.eye(3)

    ### ----- Calculation of stiffness ----- ###
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

    # Calculate stiffness
    stiff = calculate_hom_stiffness_3d_small_strain(cell, weights)
    print('Stiffness calculated.')
    #print(stiff[:, :, 0, 0])
    #print(stiff[:, :, 0, 1])
    #print(stiff[:, :, 0, 2])
    #print(stiff[:, :, 1, 1])
    #print(stiff[:, :, 1, 2])
    #print(stiff[:, :, 2, 2])

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
    stress = np.average(stress, axis=(3, 4, 5))
    stress = np.average(stress, axis=2, weights=weights)
    #strain = np.linalg.solve(stiff, stress)
    strain = np.linalg.solve(stiff.reshape((dim**2, dim**2)), stress.reshape(dim**2))
    strain = strain.reshape((dim, dim))
    diff = stress - np.einsum('ijkl, kl', stiff, strain)
    print(f'Diff strain =', np.linalg.norm(diff))
    strain_zz_in_percent = strain[2, 2] * 100
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
    nb_grid_pts = [90, 90, 90]
    #nb_grid_pts = [50, 50, 50]
    #test_hom_stiffness()
    #geometry_plots(nb_grid_pts, saving=True, showing=False)
    #comparison_cylinder(nb_grid_pts)
    #comparison_paper_relaxation(nb_grid_pts)
    comparison_paper_stiffness(nb_grid_pts)

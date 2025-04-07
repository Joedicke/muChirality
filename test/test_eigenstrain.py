"""
Test the EigenStrain class for torsion problems
"""

import sys
import os

# Default path of the library
sys.path.insert(0, os.path.join(os.getcwd(), "../../muspectre/builddir/language_bindings/libmugrid/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../../muspectre/builddir/language_bindings/libmufft/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../../muspectre/builddir/language_bindings/python"))

import numpy as np

import muSpectre as µ

from muChirality.EigenStrainTorsion import EigenStrain

def test_eigen_strain():
    """ Test wether the eigen strain is correctly added.
    """
    ### ----- Parameters ----- ###
    nb_grid_pts = [6, 6, 6]
    lengths = [1, 1, 10]
    Young = 100
    Poisson = 0
    angle = 0.15

    # Formulation
    formulation = µ.Formulation.small_strain

    # Numerical derivative (FE 5 Tetraeder + weights)
    gradient, weights = µ.linear_finite_elements.gradient_3d_5tet

    # muSpectre solver parameters
    newton_tol       = 1e-7
    cg_tol           = 1e-7 # tolerance for cg algo
    equil_tol        = 1e-7 # tolerance for equilibrium
    maxiter          = 10000
    verbose          = µ.Verbosity.Silent

    # Loading is pure torsion
    delta_F = np.zeros((3, 3))
    x_torsion = lengths[0] / 2
    y_torsion = lengths[1] / 2

    ### ----- Initialization of µSpectre ----- ###
    cell = µ.Cell(nb_grid_pts, lengths, formulation, gradient, weights)
    solver = µ.solvers.KrylovSolverCG(cell, cg_tol, maxiter, verbose)

    # Eigen class
    eigen_class = EigenStrain(cell.pixels, angle, lengths, nb_grid_pts,
                              x_torsion, y_torsion)

    ### ----- Test eigen strain ----- ###
    rng = np.random.default_rng(14210451135)
    strain = rng.random((3, 3, cell.nb_quad_pts, *nb_grid_pts))
    strain_new = strain.copy()
    eigen_class.eigen_strain_func(0, strain_new)

    hx = lengths[0] / nb_grid_pts[0]
    hy = lengths[1] / nb_grid_pts[1]
    hz = lengths[2] / nb_grid_pts[2]

    # Test the strain at one pixel, quadrature point 0
    ind_x, ind_y, ind_z = [0, 3, 3]
    strain_twist_01 = - 0.5 * angle * (ind_y * hy + 0.5 * hy - y_torsion)
    strain_twist_02 = 0.5 * angle * (ind_x * hx + 0.5 * hx - x_torsion)
    strain_twist = np.array([[0, 0, strain_twist_01],
                             [0, 0, strain_twist_02],
                             [strain_twist_01, strain_twist_02, 0]])

    diff = strain_new[:, :, 0, ind_x, ind_y, ind_z]
    diff = diff - strain[:, :, 0, ind_x, ind_y, ind_z] - strain_twist
    diff = np.linalg.norm(diff)
    assert diff < 1e-10

    # Test the strain at other pixel, quadrature point 1
    ind_x, ind_y, ind_z = [2, 2, -1]
    strain_twist_01 = - 0.5 * angle * (ind_y * hy + 0.25 * hy - y_torsion)
    strain_twist_02 = 0.5 * angle * (ind_x * hx + 0.25 * hx - x_torsion)
    strain_twist = np.array([[0, 0, strain_twist_01],
                             [0, 0, strain_twist_02],
                             [strain_twist_01, strain_twist_02, 0]])

    diff = strain_new[:, :, 1, ind_x, ind_y, ind_z]
    diff = diff - strain[:, :, 1, ind_x, ind_y, ind_z] - strain_twist
    diff = np.linalg.norm(diff)
    assert diff < 1e-10

    print('Finished test_eigen_strain')

def test_remove_eigen_strain():
    """ Test wether the eigen strain can be correctly removed.
    """
    ### ----- Parameters ----- ###
    nb_grid_pts = [4, 4, 4]
    lengths = [1, 1, 10]
    Young = 100
    Poisson = 0
    angle = 0.15

    # Formulation
    formulation = µ.Formulation.small_strain

    # Numerical derivative (FE 5 Tetraeder + weights)
    gradient, weights = µ.linear_finite_elements.gradient_3d_5tet

    # muSpectre solver parameters
    newton_tol       = 1e-7
    cg_tol           = 1e-7 # tolerance for cg algo
    equil_tol        = 1e-7 # tolerance for equilibrium
    maxiter          = 10000
    verbose          = µ.Verbosity.Silent

    # Loading is pure torsion
    delta_F = np.zeros((3, 3))
    x_torsion = lengths[0] / 2
    y_torsion = lengths[1] / 2

    ### ----- Initialization of µSpectre ----- ###
    cell = µ.Cell(nb_grid_pts, lengths, formulation, gradient, weights)
    solver = µ.solvers.KrylovSolverCG(cell, cg_tol, maxiter, verbose)

    # Eigen class
    eigen_class = EigenStrain(cell.pixels, angle, lengths, nb_grid_pts,
                              x_torsion, y_torsion)

    ### ----- Testing ----- ###
    strain = np.random.random((3, 3, cell.nb_quad_pts, *nb_grid_pts))
    strain_new = strain.copy()

    # eigen_strain_func changes the strain
    eigen_class.eigen_strain_func(0, strain_new)
    diff = np.linalg.norm(strain - strain_new)
    assert(diff > 1e-10)

    # remove_eigen_strain_func should remove the change
    eigen_class.remove_eigen_strain_func(strain_new)
    diff = np.linalg.norm(strain - strain_new)
    assert(diff < 1e-10)

if __name__ == "__main__":
    test_eigen_strain()
    test_remove_eigen_strain()

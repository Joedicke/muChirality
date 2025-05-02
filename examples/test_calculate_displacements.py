import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), "../../muspectre/builddir/language_bindings/libmugrid/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../../muspectre/builddir/language_bindings/libmufft/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../../muspectre/builddir/language_bindings/python"))

import muSpectre as µ

import numpy as np
from NuMPI import MPI
from NuMPI.IO import save_npy
from NuMPI.IO import load_npy
from NuMPI.Tools import Reduction

from muSpectre.gradient_integration import get_complemented_positions
from muChirality.EigenStrainTorsion import EigenStrain

def test_vs_analytical_displ():
    comm = MPI.COMM_WORLD

    ### ----- Parameters ----- ###
    nb_grid_pts = [3, 4, 5]
    lengths = [1, 1, 1]
    dim = 3

    gradient, weights = µ.linear_finite_elements.gradient_3d_5tet
    Young = 2600 # in MPa
    Poisson = 0.4
    delta_eps = np.zeros((3, 3))
    delta_eps[0, 0] = 0.02

    formulation = µ.Formulation.small_strain

    F0 = np.eye(3)

    # muSpectre solver parameters
    newton_tol       = 1e-7
    cg_tol           = 1e-7 # tolerance for cg algo
    equil_tol        = 1e-7 # tolerance for equilibrium
    maxiter          = 10000
    verbose          = µ.Verbosity.Silent
    fft = 'mpi' # Parallel fft

    # Geometry
    mask_xyz = np.ones(nb_grid_pts)

    ### ----- Displacement and small strain tensor ----- ###
    hx = lengths[0] / nb_grid_pts[0]
    hy = lengths[1] / nb_grid_pts[1]
    hz = lengths[2] / nb_grid_pts[2]

    # Initial positions of nodes
    x_x = np.linspace(0, lengths[0], nb_grid_pts[0]+1)
    y_y = np.linspace(0, lengths[1], nb_grid_pts[1]+1)
    z_z = np.linspace(0, lengths[2], nb_grid_pts[2]+1)

    initial_ixyz = np.empty((3, nb_grid_pts[0]+1, nb_grid_pts[1]+1, nb_grid_pts[2]+1))
    initial_ixyz[0] = x_x[:, None, None]
    initial_ixyz[1] = y_y[None, :, None]
    initial_ixyz[2] = z_z[None, None, :]

    # Initial positions of quadrature points
    x_x = np.linspace(0, lengths[0], nb_grid_pts[0], endpoint=False)
    y_y = np.linspace(0, lengths[1], nb_grid_pts[1], endpoint=False)
    z_z = np.linspace(0, lengths[2], nb_grid_pts[2], endpoint=False)
    delta_x_q = np.array([0.5, 0.25, 0.75, 0.75, 0.25]) * hx
    delta_y_q = np.array([0.5, 0.25, 0.75, 0.25, 0.75]) * hy
    delta_z_q = np.array([0.5, 0.25, 0.25, 0.75, 0.75]) * hz

    pos_quad_iqxyz = np.empty((3, 5, nb_grid_pts[0], nb_grid_pts[1], nb_grid_pts[2]))
    pos_quad_iqxyz[0] = x_x[None, :, None, None]
    pos_quad_iqxyz[1] = y_y[None, None, :, None]
    pos_quad_iqxyz[2] = z_z[None, None, None, :]
    pos_quad_iqxyz[0] = pos_quad_iqxyz[0] + delta_x_q[:, None, None, None]
    pos_quad_iqxyz[1] = pos_quad_iqxyz[1] + delta_y_q[:, None, None, None]
    pos_quad_iqxyz[2] = pos_quad_iqxyz[2] + delta_z_q[:, None, None, None]

    # Displacements (at nodes)
    #displ_ixyz = np.empty(initial_ixyz.shape)
    #displ_ixyz[0] = np.sin(initial_ixyz[0] / lengths[0])
    #displ_ixyz[1] = np.sin(initial_ixyz[0] / lengths[0]) * np.sin(initial_ixyz[1] / lengths[1])
    #displ_ixyz[2] = np.sin(initial_ixyz[0] / lengths[0])
    displ_ixyz = np.zeros(initial_ixyz.shape)
    displ_ixyz[0] = np.sin(2 * np.pi * initial_ixyz[0] / lengths[0])
    #displ_ixyz[1] = np.sin(initial_ixyz[0] / lengths[0]) * np.sin(initial_ixyz[1] / lengths[1])
    #displ_ixyz[2] = np.sin(initial_ixyz[0] / lengths[0])

    # Strain (at quad pts)
    strain_iiqxyz = np.empty((3, 3, 5, nb_grid_pts[0], nb_grid_pts[1], nb_grid_pts[2]))
    #strain_iiqxyz[0, 0] = np.cos(pos_quad_iqxyz[0] / lengths[0]) / lengths[0]
    #strain_iiqxyz[0, 1] = 0.5 * np.cos(pos_quad_iqxyz[0] / lengths[0]) / lengths[0] * np.sin(pos_quad_iqxyz[1] / lengths[1])
    #strain_iiqxyz[1, 0] = strain_iiqxyz[0, 1]
    #strain_iiqxyz[0, 2] = strain_iiqxyz[2, 0] = 0.5 * np.cos(pos_quad_iqxyz[0] / lengths[0])
    #strain_iiqxyz[1, 1] = np.sin(pos_quad_iqxyz[0] / lengths[0]) * np.cos(pos_quad_iqxyz[1] / lengths[1]) / lengths[1]
    #strain_iiqxyz[1, 2] = strain_iiqxyz[2, 1] = 0
    #strain_iiqxyz[2, 2] = 0
    strain_iiqxyz = np.zeros((3, 3, 5, nb_grid_pts[0], nb_grid_pts[1], nb_grid_pts[2]))
    strain_iiqxyz[0, 0] = np.cos(2 * np.pi * pos_quad_iqxyz[0] / lengths[0]) * 2 * np.pi / lengths[0]

    #
    strain_disc = np.zeros(strain_iiqxyz.shape)
    #strain_disc[0, 0, 0] = (


    ### ----- muSpectre calculation of small strain ----- ###
    # Cell initialization
    cell = µ.Cell(nb_grid_pts, lengths, formulation, gradient,
                  weights=weights, fft=fft)
    mat = µ.material.MaterialLinearElastic1_3d.make(cell, "hard", Young, Poisson)
    for pixel_id, pixel in cell.pixels.enumerate():
        mat.add_pixel(pixel_id)
    cell.initialise()

    # Calculate periodic displacement
    pos_ini_mu_ixyz, displ_mu_ixyz \
        = get_complemented_positions("0d", cell, strain_array=strain_iiqxyz, F0=F0,
                                     periodically_complemented=True)

    # Comparison
    print('shape of positions:', pos_ini_mu_ixyz.shape)
    diff = np.linalg.norm(pos_ini_mu_ixyz - initial_ixyz)
    print(f'Difference in initial position of nodes:', diff)
    diff = np.linalg.norm(displ_mu_ixyz - displ_ixyz)
    print(f'Difference in displ of nodes:', diff)

def test_serial_vs_parallel():
    comm = MPI.COMM_WORLD

    ### ----- Parameters ----- ###
    if comm.rank == 0:
        print('MPI size =', comm.size)

    nb_grid_pts = [3, 4, 5]
    lengths = [1, 1, 1]
    dim = 3

    gradient, weights = µ.linear_finite_elements.gradient_3d_5tet
    Young = 2600 # in MPa
    Poisson = 0.4
    delta_eps = np.zeros((3, 3))
    #delta_eps[0, 0] = 0.02
    twist = 0.05 # in 1/mm

    formulation = µ.Formulation.small_strain

    F0 = np.eye(3)

    # muSpectre solver parameters
    newton_tol       = 1e-7
    cg_tol           = 1e-7 # tolerance for cg algo
    equil_tol        = 1e-7 # tolerance for equilibrium
    maxiter          = 10000
    verbose          = µ.Verbosity.Silent
    fft = 'mpi' # Parallel fft

    # Geometry
    mask = np.ones(nb_grid_pts)
    mask[1, 1, 1] = 0

    ### ----- Calculate strain field ----- ###
    # Create + initialize muSpectre cell
    cell = µ.Cell(nb_grid_pts, lengths, formulation, gradient,
                  weights=weights, fft=fft, communicator=comm)
    mat = µ.material.MaterialLinearElastic1_3d.make(cell, "hard", Young, Poisson)
    vac = µ.material.MaterialLinearElastic1_3d.make(cell, "vacuum", 0, 0)
    mask = mask[cell.fft_engine.subdomain_slices]
    mask = mask.flatten(order='F')
    for pixel_id, pixel in cell.pixels.enumerate():
        if mask[pixel_id] == 1:
            mat.add_pixel(pixel_id)
        else:
            vac.add_pixel(pixel_id)
    cell.initialise()
    shape_strain = (dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts)

    # EigenStrain class
    x_rot_axis = lengths[0] / 2
    y_rot_axis = lengths[1] / 2
    eigen_class = EigenStrain(twist, lengths, nb_grid_pts, cell.fft_engine.subdomain_slices,
                              x_rot_axis, y_rot_axis)

    # Krylov solver initialization
    solver = µ.solvers.KrylovSolverCG(cell, cg_tol, maxiter, verbose)

    # Solving
    res = µ.solvers.newton_cg(cell, delta_eps, solver, newton_tol, equil_tol,
                                  verbose, μ.solvers.IsStrainInitialised.No,
                                  µ.StoreNativeStress.No, eigen_class.eigen_strain_func)
    strain = res.grad .reshape(shape_strain, order='F')

    norm_strain = np.linalg.norm(strain)**2
    norm_strain = Reduction(comm).sum(norm_strain)
    norm_strain = norm_strain ** 0.5
    #print(f'rank {comm.rank}: shape_strain = {strain.shape}  norm_strain = {norm_strain}')
    #print(f'rank {comm.rank}: strain = {strain[0, 2, 0, 0, 0, :]}')

    ### ----- Calculate displacement field ----- ###
    # Substract non-periodic strain part
    strain = strain.copy()
    eigen_class.remove_eigen_strain_func(strain)

    # Calculate displacement of fluctuating strain
    if comm.size == 1:
        pos_ini, displ \
            = get_complemented_positions("0d", cell, strain_array=strain, F0=F0,
                                         periodically_complemented=True)
    else:
        pos_ini, displ \
            = get_complemented_positions("0d", cell, strain_array=strain, F0=F0,
                                         periodically_complemented=False)

    print(f'rank {comm.rank}: displ_shape = {displ.shape}')
    print(f'rank {comm.rank}: displ[0, 1, 1, :] = {displ[0, 1, 1, :]}')

def test_prod_vs_product():
    ### ----- Parameters ----- ###
    if MPI.COMM_WORLD.size == 1:
        raise ValueError('Difference is only apparent for comm.size > 1.')

    nb_grid_pts = [3, 4, 5]
    lengths = [1, 1, 1]
    dim = 3

    gradient, weights = µ.linear_finite_elements.gradient_3d_5tet
    Young = 2600 # in MPa
    Poisson = 0.4
    delta_eps = np.zeros((3, 3))
    twist = 0.05 # in 1/mm

    formulation = µ.Formulation.small_strain

    F0 = np.eye(3)

    # muSpectre solver parameters
    newton_tol       = 1e-7
    cg_tol           = 1e-7 # tolerance for cg algo
    equil_tol        = 1e-7 # tolerance for equilibrium
    maxiter          = 10000
    verbose          = µ.Verbosity.Silent
    fft = 'mpi' # Parallel fft

    # Geometry
    mask = np.ones(nb_grid_pts)
    mask[1, 1, 1] = 0

    name_old = 'fluctuating_displacement_test_old_numpy.npy'

    ### ----- Calculate strain field ----- ###
    # Create + initialize muSpectre cell
    cell = µ.Cell(nb_grid_pts, lengths, formulation, gradient,
                  weights=weights, fft=fft, communicator=MPI.COMM_WORLD)
    mat = µ.material.MaterialLinearElastic1_3d.make(cell, "hard", Young, Poisson)
    vac = µ.material.MaterialLinearElastic1_3d.make(cell, "vacuum", 0, 0)
    mask = mask[cell.fft_engine.subdomain_slices]
    mask = mask.flatten(order='F')
    for pixel_id, pixel in cell.pixels.enumerate():
        if mask[pixel_id] == 1:
            mat.add_pixel(pixel_id)
        else:
            vac.add_pixel(pixel_id)
    cell.initialise()
    shape_strain = (dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts)

    # EigenStrain class
    x_rot_axis = lengths[0] / 2
    y_rot_axis = lengths[1] / 2
    eigen_class = EigenStrain(twist, lengths, nb_grid_pts, cell.fft_engine.subdomain_slices,
                              x_rot_axis, y_rot_axis)

    # Krylov solver initialization
    solver = µ.solvers.KrylovSolverCG(cell, cg_tol, maxiter, verbose)

    # Solving
    res = µ.solvers.newton_cg(cell, delta_eps, solver, newton_tol, equil_tol,
                                  verbose, μ.solvers.IsStrainInitialised.No,
                                  µ.StoreNativeStress.No, eigen_class.eigen_strain_func)
    strain = res.grad .reshape(shape_strain, order='F')

    ### ----- Calculate fluctuating displacement field ----- ###
    # Substract non-periodic strain part
    strain = strain.copy()
    eigen_class.remove_eigen_strain_func(strain)
    pos_ini, displ \
            = get_complemented_positions("0d", cell, strain_array=strain, F0=F0,
                                         periodically_complemented=False)

    #np.save(name_old, displ)
    # helper = (cell.nb_subdomain_grid_pts[2], cell.nb_subdomain_grid_pts[1], cell.nb_subdomain_grid_pts[0])
    i = 0
    helper = np.ascontiguousarray(displ[0])
    save_npy(name_old, helper, tuple(cell.subdomain_locations),
             tuple(cell.nb_domain_grid_pts), MPI.COMM_WORLD)

    ### ----- Compare with old displacement field ----- ###
    displ_old = load_npy(name_old, tuple(cell.subdomain_locations),
                         tuple(cell.nb_domain_grid_pts), MPI.COMM_WORLD)
    displ_old = displ_old[cell.fft_engine.subdomain_slices]
    print(f'rank {MPI.COMM_WORLD.rank}:', displ[0, 0, 0, :], displ_old[0, 0, :])

    diff = np.linalg.norm(displ[0] - displ_old)**2
    print(f'rank {MPI.COMM_WORLD.rank}:', diff)
    diff = Reduction(MPI.COMM_WORLD).sum(diff)
    diff = diff ** 0.5
    print(f'Difference calculation with np.prod vs np.product = {diff}')

    # print(np.version.version)

if __name__ == "__main__":
    #test_prod_vs_product()
    test_serial_vs_parallel()



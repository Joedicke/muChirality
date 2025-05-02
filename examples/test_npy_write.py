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

### ----- Parameters ------ ###
nb_grid_pts = [3, 4, 5]
lengths = [1, 1, 1]
dim = 3

gradient, weights = µ.linear_finite_elements.gradient_3d_5tet
Young = 2600 # in MPa
Poisson = 0.4
delta_eps = np.zeros((3, 3))
delta_eps[0, 0] = 0.02

formulation = µ.Formulation.small_strain

# muSpectre solver parameters
newton_tol       = 1e-7
cg_tol           = 1e-7 # tolerance for cg algo
equil_tol        = 1e-7 # tolerance for equilibrium
maxiter          = 10000
verbose          = µ.Verbosity.Silent
fft = 'mpi' # Parallel fft

mask = np.ones(nb_grid_pts)
mask[1, 1, 2] = 0

#print(dir(muGrid))


### ----- muSpectre calculation ----- ###
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

# Solve muSpectre
res = µ.solvers.newton_cg(cell, delta_eps, solver, newton_tol, equil_tol,
                          verbose, μ.solvers.IsStrainInitialised.No,
                          µ.StoreNativeStress.No)
stress = res.stress.reshape(shape, order='F')

helper_stress = stress[0, 0, 0]
stress1 = helper_stress.copy()
print(f'Rank {MPI.COMM_WORLD.rank}: stress1:', stress1[0, 1])

### ----- Save helper_stress (NuMPI) ----- ###
name_npy = f'test_mpi_size={MPI.COMM_WORLD.size}.npy'
helper = cell.nb_subdomain_grid_pts
helper = (cell.nb_subdomain_grid_pts[2], cell.nb_subdomain_grid_pts[1], cell.nb_subdomain_grid_pts[0])
helper_stress = np.ascontiguousarray(helper_stress)
save_npy(name_npy, helper_stress, tuple(cell.subdomain_locations),
         tuple(cell.nb_domain_grid_pts), MPI.COMM_WORLD)

### ----- Load for special interest-case ----- ###
# Special case: Saved in parallel and loaded in serial
if MPI.COMM_WORLD.size == 1:
    name2 = f'test_mpi_size=2.npy'
    stress3 = load_npy(name2, tuple(cell.subdomain_locations),
                       tuple(cell.nb_domain_grid_pts), MPI.COMM_WORLD)
    diff = np.linalg.norm(stress1 - stress3)
    print(f'Special case: stress3:', stress3[0, 1])
    print(f'Special case: Difference saved and loaded stress = {diff}')

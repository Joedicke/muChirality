import sys
import os
import shutil
sys.path.insert(0, os.path.join(os.getcwd(), "/usr/local/lib/python3.8/site-packages"))

import numpy as np

import muSpectre as µ
from NuMPI import MPI
from NuMPI.Tools import Reduction

### ----- Define class for Eigenstrain representing twist ----- ###
class EigenStrain:
    """
    Class for describing the eigenstrain due to imposed torsion along z-axis
    for a five tetrahedra discretization.

    Attributes
    ----------
    nb_grid_pts: list of 3 ints
                 Number of grid pts in each direction
    pixels: µSpectre pixels object
            pixels of the discretized unit cell
    angle: float
           imposed rotation angle
    hx: float
        voxel length in x-direction
    hy: float
        voxel length in y-direction
    x_rot_axis: float
                x-coordinate of the rotation axis
    y_rot_axis: float
                y-coordinate of the rotation axis

    Methods
    -------
    eigen_strain_func(step_nb, strain_field):
          Change the strain_field to account for the eigen strain.
    remove_eigen_strain_func(strain_field):
          Inverse of eigen_strain_func.
    """
    def __init__(self, angle, lengths, nb_grid_pts, slices, x_rot_axis, y_rot_axis):
        """
        Parameters
        ----------
        angle: float
               imposed rotation angle
        lengths: List of 3 floats
                 Lengths of the unit cell in cartesian directions
        nb_grid_pts: List of 3 ints
             Number of voxels in cartesian directions
        slices: List of 3 slices
                Slices of the subdomain for parallel processes
        x_rot_axis: float
             x-coordinate of the rotation axis
        y_rot_axis: float
             y-coordinate of the rotation axis
        """
        self.nb_grid_pts = nb_grid_pts
        self.slices = slices
        self.angle = angle
        self.hx = lengths[0] / nb_grid_pts[0]
        self.hy = lengths[1] / nb_grid_pts[1]
        self.x_rot_axis = x_rot_axis
        self.y_rot_axis = y_rot_axis

    def __call__(self, nb_steps, strain_field):
        self.eigen_strain_func(step_nb, strain_field)

    def eigen_strain_func(self, step_nb, strain_field):
        """
        Change the strain_field to account for the eigen strain.

        Parameters
        ----------
        step_nb: int
                 Number of load step
        strain_field: np.array(3, 3, 5, *nb_grid_pts) of floats
                      Strain field
        """
        hx = self.hx
        hy = self.hy
        x_rot_axis = self.x_rot_axis
        y_rot_axis = self.y_rot_axis
        angle = self.angle
        slice_y = self.slices[1]

        # Coordinates of voxel
        x = np.arange(self.nb_grid_pts[0]) * hx # x-coordinate of voxel
        y = np.arange(self.nb_grid_pts[1]) * hy # y-coordinate of voxel

        # Difference between voxel coordinate and quadrature point
        # coordinate
        delta_x = np.array([0.5, 0.25, 0.75, 0.75, 0.25]) * hx
        delta_y = np.array([0.5, 0.25, 0.75, 0.25, 0.75]) * hy

        # If pfft is used as fft in parallel simulations, the y-dimension is sliced
        y = y[slice_y]

        # Eigenstrain
        strain_field[0, 2] -= 0.5 * angle * (y[None, None, :, None] + delta_y[:, None, None, None] - y_rot_axis)
        strain_field[2, 0] -= 0.5 * angle * (y[None, None, :, None] + delta_y[:, None, None, None] - y_rot_axis)
        strain_field[1, 2] += 0.5 * angle * (x[None, :, None, None] + delta_x[:, None, None, None] - x_rot_axis)
        strain_field[2, 1] += 0.5 * angle * (x[None, :, None, None] + delta_x[:, None, None, None] - x_rot_axis)


### ----- Parameter definitions ----- ###
# Geometry (Beam with square cross section)
dim = 3
length = 10
width = 1

# Discretization
N_xy = 10
N_z = 10
gradient, weights = µ.linear_finite_elements.gradient_3d_5tet # 3D linear finite elements (5 tetraedra per voxel)

# Material
Young = 100
Poisson = 0

# Loading
twist = 0.15
delta_F = np.zeros((3, 3))

# muSpectre solver parameters
formulation = µ.Formulation.small_strain
newton_tol       = 1e-7
cg_tol           = 1e-7 # tolerance for cg algo
equil_tol        = 1e-7 # tolerance for equilibrium
maxiter          = 10000
verbose          = µ.Verbosity.Silent
fft = 'mpi' # Parallel fft

### ----- Initialization ----- ###
# Voxel sizes
hx = width / (N_xy - 2)
hy = width / (N_xy - 2)
hz = length / N_z

# Size of the simulation domain (including void to break periodicity)
domain = [hx * N_xy, hy * N_xy, length]

x_rot_axis = domain[0] / 2 # Location of rotation axis
y_rot_axis = domain[1] / 2

# Geometry: Beam with square cross section
mask = np.ones([N_xy, N_xy, N_z])
mask[0, :, :] = 0
mask[-1, :, :] = 0
mask[:, 0, :] = 0
mask[:, -1, :] = 0

# Initialize muSpectre cell
cell = µ.Cell(mask.shape, domain, formulation, gradient,
              weights=weights, fft=fft, communicator=MPI.COMM_WORLD)
mask = mask[cell.fft_engine.subdomain_slices]

# Define material
mat = µ.material.MaterialLinearElastic1_3d.make(cell, "hard", Young, Poisson)
vac = µ.material.MaterialLinearElastic1_3d.make(cell, "vacuum", 0, 0)
mask = mask.flatten(order='F')
for pixel_id, pixel in cell.pixels.enumerate():
    if mask[pixel_id] == 1:
        mat.add_pixel(pixel_id)
    else:
        vac.add_pixel(pixel_id)
cell.initialise()

# Initialize conjugate gradient solver
solver = µ.solvers.KrylovSolverCG(cell, cg_tol, maxiter, verbose)

# Initialize Eigen class
eigen_class = EigenStrain(twist, domain, [N_xy, N_xy, N_z], cell.fft_engine.subdomain_slices,
                          x_rot_axis, y_rot_axis)

### ----- Calculation ----- ###
# Solve mechanical equilibrium
res = µ.solvers.newton_cg(cell, delta_F, solver, newton_tol, equil_tol,
                          verbose, μ.solvers.IsStrainInitialised.No,
                          µ.StoreNativeStress.No,
                          eigen_class.eigen_strain_func)
shape = (dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts)
stress_num = res.stress.reshape(shape, order='F')
strain_num = res.grad.reshape(shape, order='F')

# Coordinates of the centers of the 5 tetradras
x = np.arange(N_xy) * hx # x-coordinate of voxel
y = np.arange(N_xy) * hy # y-coordinate of voxel
z = np.arange(N_z) * hz # z-coordinate of voxel
x = x[cell.fft_engine.subdomain_slices[0]]
y = y[cell.fft_engine.subdomain_slices[1]]
z = z[cell.fft_engine.subdomain_slices[2]]

X, Y, Z = np.meshgrid(x, y, z)
# Format X[i, j, k, l] with i: quad_pt, j: x-coord, k: y-coord, l: z-coord
X = X.transpose((1, 0, 2))
Y = Y.transpose((1, 0, 2))
X = np.stack((X, X, X, X, X))
Y = np.stack((Y, Y, Y, Y, Y))
X[0] += 0.5 * hx
X[1] += 0.25 * hx
X[2] += 0.75 * hx
X[3] += 0.75 * hx
X[4] += 0.25 * hx
Y[0] += 0.5 * hy
Y[1] += 0.25 * hy
Y[2] += 0.75 * hy
Y[3] += 0.25 * hy
Y[4] += 0.75 * hy

# Calculate numerical torsion stiffness
helper = - stress_num[0, 2] * (Y - y_rot_axis)
helper += stress_num[1, 2] * (X - x_rot_axis)
moment = hx * hy * hz / 6 * np.sum(helper) # Total moment
moment += hx * hy * hz / 6 * np.sum(helper[0])
moment = Reduction(MPI.COMM_WORLD).sum(moment) / length

stiffness = moment / twist

# Calculate analytical torsion stiffness + error
mu = Young / 2 / (1 + Poisson)
stiffness_ana = mu * 0.141 * width ** 4
error = np.linalg.norm(stiffness-stiffness_ana) / abs(stiffness_ana) * 100

if (MPI.COMM_WORLD.rank == 0):
    print(f'Numerical torsion stiffness = {stiffness:.2f}')
    print(f'Analytical torsion stiffness = {stiffness_ana:.2f}')
    print(f'Error torsion stiffness = {error:.2f}%')

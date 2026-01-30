import sys
import os
import shutil
sys.path.insert(0, os.path.join(os.getcwd(), "../../muspectre/builddir/language_bindings/libmugrid/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../../muspectre/builddir/language_bindings/libmufft/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../../muspectre/builddir/language_bindings/python"))

import numpy as np

import muSpectre as µ
import netCDF4
from NuMPI import MPI

from muChirality.EigenStrainTorsion import EigenStrain
from muSpectre.gradient_integration import get_complemented_positions
import muChirality.Geometries as geo

comm = MPI.COMM_WORLD

folder =  'results_paper/chiral_Nuc=1x1xNucz/'
name_data = folder + 'data_3D_Nucz=1.nc'

# Discretization muSpectre
nb_grid_pts_uc = [100, 100, 100]
#nb_grid_pts_uc = [10, 10, 10]
nb_quad_pts = 5

# Additional parameters for muSpectre
twist = 0.05
a = 0.5
thickness = 0.06 * a
radius_out = 0.4 * a
radius_inn = 0.34 * a
angle_mat = np.pi * 35 / 180
formulation = µ.Formulation.small_strain
gradient, weights = µ.linear_finite_elements.gradient_3d_5tet
F0 = np.eye(3)
fft = 'mpi'
Young = 2600
Poisson = 0.4

### ----- Initializations ----- ###
# Geometry
mask, lengths =\
    geo.chiral_2_mult_unit_cell([1, 1, 1], nb_grid_pts_uc, a, radius_out,
                                radius_inn, thickness, alpha=angle_mat)

# muSpectre cell
cell = µ.Cell(nb_grid_pts_uc, lengths, formulation, gradient,
              weights=weights, fft=fft, communicator=comm)
mask = mask[cell.fft_engine.subdomain_slices]
print(f'rank {comm.rank}: mask_shape = {mask.shape}')

# Material
mat = µ.material.MaterialLinearElastic1_3d.make(cell, "hard", Young, Poisson)
vac = µ.material.MaterialLinearElastic1_3d.make(cell, "vacuum", 0, 0)
mask = mask.flatten(order='F')
for pixel_id, pixel in cell.pixels.enumerate():
    if mask[pixel_id] == 1:
        mat.add_pixel(pixel_id)
    else:
        vac.add_pixel(pixel_id)
cell.initialise()

# EigenStrain class
x_rot_axis = lengths[0] / 2
y_rot_axis = lengths[1] / 2
eigen_class = EigenStrain(twist, lengths, nb_grid_pts, cell.fft_engine.subdomain_slices,
                          x_rot_axis, y_rot_axis)

# Read strain
nc = netCDF4.Dataset(name_mu, 'r')
strain = nc.variables['strain'][0]
nc.close()
strain = strain[:, :, :, cell.fft_engine.subdomain_slices[0],
                cell.fft_engine.subdomain_slices[1], cell.fft_engine.subdomain_slices[2]]
print(f'rank {comm.rank}: strain_shape = {strain.shape}')

### ----- Calculate muSpectre displacement ----- ###
# Calculate displacement of periodic strain
strain_no_eigen = strain.copy()
eigen_class.remove_eigen_strain_func(strain_no_eigen)
[x_0, y_0, z_0], [x_displ, y_displ, z_displ] \
    = get_complemented_positions("0d", cell, strain_array=strain_no_eigen, F0=F0,
                                 periodically_complemented=True)

# Add displacement of rotational strain
#x = np.linspace(0, lengths[0], nb_grid_pts_uc[0]+1, endpoint=True)
#y = np.linspace(0, lengths[1], nb_grid_pts_uc[1]+1, endpoint=True)
#z = np.linspace(0, lengths[2], nb_grid_pts_uc[2]+1, endpoint=True)
#helper = - twist * np.einsum('i,j->ij', y-y_rot_axis, z)
#x_displ += helper[None, :, :]
#helper = twist * np.einsum('i,j->ij', x-x_rot_axis, z)
#y_displ += helper[:, None, :]

# Initial and final node positions
#pos_initial = np.asarray([x_0, y_0, z_0])
#displ = np.asarray([x_displ, y_displ, z_displ])
#pos_displ = pos_initial + scale * displ

### ----- Save displacements ----- ###
#name = folder + 'displacements.npy'
#np.save(name, displ)

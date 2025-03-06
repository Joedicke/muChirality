import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), "../../muspectre/builddir/language_bindings/libmugrid/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../../muspectre/builddir/language_bindings/libmufft/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../../muspectre/builddir/language_bindings/python"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image

import muSpectre as µ
from muSpectre.gradient_integration import get_complemented_positions

from muChirality.EigenStrainTorsion import EigenStrain
import muChirality.Geometries as geo

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['font.serif'] = ['Arial']
mpl.rcParams['font.cursive'] = ['Arial']
mpl.rcParams['font.size'] = '10'
mpl.rcParams['legend.fontsize'] = '10'
mpl.rcParams['xtick.labelsize'] = '9'
mpl.rcParams['ytick.labelsize'] = '9'
mpl.rcParams['svg.fonttype'] = 'none'

##################################################################
### ------------- Plot error stiffnesses ------------- ###
##################################################################
### ----- Parameters ----- ###
# Colors
color1 = 'red'
color2 = 'blue'

# Fontsize
size_large = 15
tick_size = 11

### ----- Read data ----- ###
# Folders torsion
folder_cyl = f'results/cylinder/method2/radius={0.4}_Lz={10}_Young={100}_'
folder_cyl += f'Poisson={0}_twist={0.15}_Nz={30}/'
folder_sq = f'results/square_beam/method2/thick={0.6}_Lz={10}_Young={100}_'
folder_sq += f'Poisson={0}_twist={0.15}_Nz={30}/'

# Read error torsion stiffness cylinder
data = np.loadtxt((folder_cyl + 'data.txt'), skiprows=1)
Nx_cyl = data[:, 0].astype(int)
err_stiff_cyl = data[:, 7]

# Read error torsion stiffness square beam
data = np.loadtxt((folder_sq + 'data.txt'), skiprows=1)
Nx_sq = data[:, 0].astype(int)
err_stiff_sq = data[:, 5]

# Read error bending stiffness rectangular beam
name = 'results/bending/data_L_beam=10x0.5x0.8_Young=120_Poisson=0.3_moment=4.txt'
N_bend = np.loadtxt(name, skiprows=1, max_rows=1).astype(int)
err_stiff_bend = np.loadtxt(name, skiprows=3)

### ----- Plot data ----- ###
fig, axes = plt.subplots(1, 2, figsize=(9, 5), sharey=True, layout='constrained')

# Plot torsion stiffnesses
axes[0].set_xlabel('Number of voxels $N_x$=$N_y$', fontsize=size_large)
axes[0].set_ylabel('Difference between stiffnesses (%)', fontsize=size_large)
axes[0].tick_params(axis='both', which='major', labelsize=tick_size)

axes[0].plot(Nx_cyl, err_stiff_cyl, marker='x', color=color1)
axes[0].plot(Nx_sq, err_stiff_sq, marker='x', color=color2)

x = 1.2 * Nx_cyl[0]
y = 0.8 * err_stiff_cyl[0]
axes[0].text(x, y, 'circular cross-section', color=color1, fontsize=size_large)

x = 1.02 * Nx_sq[-1]
y = -0.1 * err_stiff_sq[0]
axes[0].text(x, y, 'square cross-section', color=color2, fontsize=size_large, ha='right')

# Plot bending stiffness
axes[1].set_xlabel('Number of voxels $N_x$=$N_y$=$N_z$', fontsize=size_large)
axes[1].tick_params(axis='both', which='major', labelsize=tick_size)
axes[1].plot(N_bend, err_stiff_bend, marker='x', color=color2)

# Numerate subplots
axes[0].text(30, 4.2, '(a)   Torsion', fontsize=size_large)
axes[1].text(5, 4.2, '(b)   Bending', fontsize=size_large)

### ----- Finish ----- ###
name = 'results/validation_stiffnesses.pdf'
fig.savefig(name, bbox_inches='tight')
plt.show()
plt.close(fig)

##################################################################
### ----------- Plot stress in cylinder (torsion) ------------ ###
##################################################################
### ----- Parameters ----- ###
figsize = (8, 4)
cmap = mpl.colormaps['viridis']

# Fontsize
size_large = 15
tick_size = 11

# Discretization
lengths = [1, 1, 10]
radius = 0.4
Nxy = 110
Nz = 30
nb_quad_pts = 5

# z-coordinate of plots
z_pt = 0

### ----- Read data ----- ###
# File with data
folder_cyl = f'results/cylinder/method2/radius={radius}_Lz={lengths[2]}_Young={100}_'
folder_cyl += f'Poisson={0}_twist={0.15}_Nz={Nz}/'
folder = folder_cyl + f'stresses_Nxy={Nxy}/'

# Analytical stress
stress_ana = np.zeros((3, 3, nb_quad_pts, Nxy, Nxy, Nz))
for i_quad in range(nb_quad_pts):
    name = folder + f'stress_ana_quad_pt_{i_quad}_entry_02.npy'
    stress_ana[0, 2, i_quad, :, :, :] = np.load(name)
    stress_ana[2, 0, i_quad, :, :, :] = stress_ana[0, 2, i_quad, :, :, :]
    name = folder + f'stress_ana_quad_pt_{i_quad}_entry_12.npy'
    stress_ana[1, 2, i_quad, :, :, :] = np.load(name)
    stress_ana[2, 1, i_quad, :, :, :] = stress_ana[1, 2, i_quad, :, :, :]

# Stress error
stress_error = np.empty(stress_ana.shape)
for i_quad in range(nb_quad_pts):
    name = folder + f'error_stress_quad_pt_{i_quad}_entry_00.npy'
    stress_error[0, 0, i_quad, :, :, :] = np.load(name)
    name = folder + f'error_stress_quad_pt_{i_quad}_entry_01.npy'
    stress_error[0, 1, i_quad, :, :, :] = np.load(name)
    stress_error[1, 0, i_quad, :, :, :] = stress_error[0, 1, i_quad, :, :, :]
    name = folder + f'error_stress_quad_pt_{i_quad}_entry_02.npy'
    stress_error[0, 2, i_quad, :, :, :] = np.load(name)
    stress_error[2, 0, i_quad, :, :, :] = stress_error[0, 2, i_quad, :, :, :]
    name = folder + f'error_stress_quad_pt_{i_quad}_entry_11.npy'
    stress_error[1, 1, i_quad, :, :, :] = np.load(name)
    name = folder + f'error_stress_quad_pt_{i_quad}_entry_12.npy'
    stress_error[1, 2, i_quad, :, :, :] = np.load(name)
    stress_error[2, 1, i_quad, :, :, :] = stress_error[1, 2, i_quad, :, :, :]
    name = folder + f'error_stress_quad_pt_{i_quad}_entry_22.npy'
    stress_error[2, 2, i_quad, :, :, :] = np.load(name)

# Numerical stress
stress_num = stress_ana - stress_error

# Average over quad pts for all stresses
stress_ana = np.average(stress_ana, axis=2, weights=[1/3, 1/6, 1/6, 1/6, 1/6])
stress_error = np.average(stress_error, axis=2, weights=[1/3, 1/6, 1/6, 1/6, 1/6])
stress_num = np.average(stress_num, axis=2, weights=[1/3, 1/6, 1/6, 1/6, 1/6])


### ----- Plot stress_00 distribution ----- ###
# Grid coordinates
x = np.linspace(0, lengths[0], Nxy+1)
y = np.linspace(0, lengths[1], Nxy+1)

# Value range
vmin = min(np.amin(stress_ana[0, 0, :, :, z_pt]),
           np.amin(stress_num[0, 0, :, :, z_pt]))
vmax = max(np.amax(stress_ana[0, 0, :, :, z_pt]),
           np.amax(stress_num[0, 0, :, :, z_pt]))


# Prepare plot
fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'wspace':0.3})
for i in range(2):
    axes[i].set_aspect('equal')
    axes[i].set_xlabel('Position x (-)', fontsize=size_large)
    axes[i].set_ylabel('Position y (-)', fontsize=size_large)

# Mask voxels of vacuum
mask = geo.cylinder([Nxy, Nxy, Nz], lengths, radius)
mask = 1 - mask[:, :, z_pt]

# Plot
helper = np.ma.masked_array(stress_ana[0, 0, :, :, z_pt], mask)
im = axes[0].pcolormesh(x, y, helper.T, cmap=cmap,
                        vmin=vmin, vmax=vmax, rasterized=True)
helper = np.ma.masked_array(stress_num[0, 0, :, :, z_pt], mask)
im = axes[1].pcolormesh(x, y, helper.T, cmap=cmap,
                        vmin=vmin, vmax=vmax, rasterized=True)
cbar = fig.colorbar(im, ax=axes)
cbar.ax.set_ylabel(r'Stress $\sigma_{xx}$ (-)',
                   rotation=-90, va='bottom', fontsize=size_large)

# Save and show
#plt.show()
plt.close(fig)

### ----- Plot stress_02 distribution ----- ###
# Grid coordinates
x = np.linspace(0, lengths[0], Nxy+1)
y = np.linspace(0, lengths[1], Nxy+1)

# Value range
vmin = min(np.amin(stress_ana[0, 2, :, :, z_pt]),
           np.amin(stress_num[0, 2, :, :, z_pt]))
vmax = max(np.amax(stress_ana[0, 2, :, :, z_pt]),
           np.amax(stress_num[0, 2, :, :, z_pt]))


# Prepare plot
fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'wspace':0.3})
for i in range(2):
    axes[i].set_aspect('equal')
    axes[i].set_xlabel('Position x (-)', fontsize=size_large)
    axes[i].set_ylabel('Position y (-)', fontsize=size_large)

# Plot
helper = np.ma.masked_array(stress_ana[0, 2, :, :, z_pt], mask)
im = axes[0].pcolormesh(x, y, helper.T, cmap=cmap,
                        vmin=vmin, vmax=vmax, rasterized=True)
helper = np.ma.masked_array(stress_num[0, 2, :, :, z_pt], mask)
im = axes[1].pcolormesh(x, y, helper.T, cmap=cmap,
                        vmin=vmin, vmax=vmax, rasterized=True)
cbar = fig.colorbar(im, ax=axes)
cbar.ax.set_ylabel(r'Stress $\sigma_{xz}$ (-)',
                   rotation=-90, va='bottom', fontsize=size_large)

# Number subplots
axes[0].text(-0.25, 0.98, '(a)', fontsize=size_large)
axes[1].text(-0.25, 0.98, '(b)', fontsize=size_large)

# Save and show
#plt.show()
name = 'results/stress_xz_distribution_cylinder.pdf'
fig.savefig(name, bbox_inches='tight')
plt.close(fig)

### ----- Plot norm of stress distribution ----- ###
# Grid coordinates
x = np.linspace(0, lengths[0], Nxy+1)
y = np.linspace(0, lengths[1], Nxy+1)

norm_stress_ana = (stress_ana[0, 0] ** 2 + 2 * stress_ana[0, 1] ** 2 +\
                   2 * stress_ana[0, 2] ** 2 + stress_ana[1, 1] ** 2 +\
                   2 * stress_ana[1, 2] ** 2 + stress_ana[2, 2] ** 2) ** 0.5
norm_stress_num = (stress_num[0, 0] ** 2 + 2 * stress_num[0, 1] ** 2 +\
                   2 * stress_num[0, 2] ** 2 + stress_num[1, 1] ** 2 +\
                   2 * stress_num[1, 2] ** 2 + stress_num[2, 2] ** 2) ** 0.5


# Value range
vmin = min(np.amin(norm_stress_ana[:, :, z_pt]),
           np.amin(norm_stress_num[:, :, z_pt]))
vmax = max(np.amax(norm_stress_ana[:, :, z_pt]),
           np.amax(norm_stress_num[:, :, z_pt]))


# Prepare plot
fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'wspace':0.3})
for i in range(2):
    axes[i].set_aspect('equal')
    axes[i].set_xlabel('Position x (-)', fontsize=size_large)
    axes[i].set_ylabel('Position y (-)', fontsize=size_large)

# Plot
helper = np.ma.masked_array(norm_stress_ana[:, :, z_pt], mask)
im = axes[0].pcolormesh(x, y, helper.T, cmap=cmap,
                        vmin=vmin, vmax=vmax, rasterized=True)
helper = np.ma.masked_array(norm_stress_num[:, :, z_pt], mask)
im = axes[1].pcolormesh(x, y, helper.T, cmap=cmap,
                        vmin=vmin, vmax=vmax, rasterized=True)
cbar = fig.colorbar(im, ax=axes)
cbar.ax.set_ylabel(r'Norm of stress $\sigma$ (-)',
                   rotation=-90, va='bottom', fontsize=size_large)

# Number subplots
axes[0].text(-0.25, 0.98, '(a)', fontsize=size_large)
axes[1].text(-0.25, 0.98, '(b)', fontsize=size_large)

# Save and show
#plt.show()
name = 'results/stress_norm_distribution_cylinder.pdf'
fig.savefig(name, bbox_inches='tight')
plt.close(fig)

### ----- Compare displacement norm with wire ----- ###
if False:
    fontsize = 12
    # Other parameters for calculation
    nb_grid_pts = [Nxy, Nxy, Nz]
    formulation = µ.Formulation.small_strain
    gradient, weights = µ.linear_finite_elements.gradient_3d_5tet
    fft = 'fftw'
    Young = 2600
    Poisson = 0.4
    mask = geo.cylinder([Nxy, Nxy, Nz], lengths, radius)
    delta_eps = np.zeros((3, 3))
    delta_eps[2, 2] = 0.01
    F0 = np.eye(3)

    # Create + initialize muSpectre cell
    cell = µ.Cell(nb_grid_pts, lengths, formulation, gradient,
                  weights=weights, fft=fft)
    mat = µ.material.MaterialLinearElastic1_3d.make(cell, "hard", Young, Poisson)
    vac = µ.material.MaterialLinearElastic1_3d.make(cell, "vacuum", 0, 0)
    mask = mask.flatten(order='F')
    for pixel_id, pixel in cell.pixels.enumerate():
        if mask[pixel_id] == 1:
            mat.add_pixel(pixel_id)
        else:
            vac.add_pixel(pixel_id)
    cell.initialise()

    # Solve muSpectre
    solver = µ.solvers.KrylovSolverCG(cell, 5e-7, 1000)
    result = µ.solvers.newton_cg(cell, delta_eps, solver, 1e-6,
                                 1e-6)
    shape = (3, 3, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts)
    stress = result.stress.reshape(shape, order='F')
    stress = np.average(stress, axis=2, weights=[1/3, 1/6, 1/6, 1/6, 1/6])


    # For plotting
    x = np.linspace(0, lengths[0], Nxy+1)
    y = np.linspace(0, lengths[1], Nxy+1)

    norm_stress_wire = (stress[0, 0] ** 2 + 2 * stress[0, 1] ** 2 +\
                        2 * stress[0, 2] ** 2 + stress[1, 1] ** 2 +\
                        2 * stress[1, 2] ** 2 + stress[2, 2] ** 2) ** 0.5
    norm_stress_num = (stress_num[0, 0] ** 2 + 2 * stress_num[0, 1] ** 2 +\
                       2 * stress_num[0, 2] ** 2 + stress_num[1, 1] ** 2 +\
                       2 * stress_num[1, 2] ** 2 + stress_num[2, 2] ** 2) ** 0.5

    mask = 1 - mask.reshape(nb_grid_pts, order='F')[:, :, z_pt]


    # Prepare plot
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'wspace':0.45})
    for i in range(2):
        axes[i].set_aspect('equal')
        axes[i].set_xlabel('Position x (-)', fontsize=fontsize)
        axes[i].set_ylabel('Position y (-)', fontsize=fontsize)

    axes[0].set_title('Traction', fontsize=fontsize)
    axes[1].set_title('Torsion', fontsize=fontsize)

    # Plot
    helper = np.ma.masked_array(norm_stress_wire[:, :, z_pt], mask)
    im = axes[0].pcolormesh(x, y, helper.T, cmap=cmap,rasterized=True)
    cbar = fig.colorbar(im, ax=axes[0])
    cbar.ax.set_ylabel(r'Norm of stress $\sigma$ (-)',
                       rotation=-90, va='bottom', fontsize=fontsize)
    helper = np.ma.masked_array(norm_stress_num[:, :, z_pt], mask)
    im = axes[1].pcolormesh(x, y, helper.T, cmap=cmap,rasterized=True)
    cbar = fig.colorbar(im, ax=axes[1])
    cbar.ax.set_ylabel(r'Norm of stress $\sigma$ (-)',
                       rotation=-90, va='bottom', fontsize=fontsize)

    # Save and show
    #plt.show()
    name = 'results/stress_norm_traction_vs_torsion.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)




##################################################################
### ------------------ Plot result chiral 1 ------------------ ###
##################################################################
### ----- Parameters ----- ###
# Colors
color1 = 'red'
color2 = 'blue'
color3 = 'orange'

# Fontsize
size_large = 15
tick_size = 11
### ----- Read data ----- ###
# Folders
folder = f'results/chiral1/'

# 3D images of geometries
im_1 = np.asarray(Image.open((folder + 'geometry_angle=0.2.png')))
im_2 = np.asarray(Image.open((folder + 'geometry_angle=0.25.png')))

# Read force in z-direction
name_file = folder + 'Lz=1_radius=0.4_thick=0.1_Young=100_Poisson=0_Nxyz=70/'
name_file += 'data.txt'
twists = np.loadtxt(name_file, skiprows=1, max_rows=1)
force_cyl = np.loadtxt(name_file, skiprows=3, max_rows=1)
force_angle1 = np.loadtxt(name_file, skiprows=5, max_rows=1)
force_angle2 = np.loadtxt(name_file, skiprows=7, max_rows=1)


### ----- Plot data ----- ###
fig = plt.figure(figsize=(7, 5))
gs1 = fig.add_gridspec(2, 1, left=0.05, right=0.4)
gs2 = fig.add_gridspec(1, 1, left=0.5, right=0.95)

# Plot 3D images
ax = fig.add_subplot(gs1[0, 0])
ax.axis('off')
implot = ax.imshow(im_1)
ax.text(-200, 120, '(a)', color='black', fontsize=size_large)
ax = fig.add_subplot(gs1[1, 0])
ax.axis('off')
implot = ax.imshow(im_2)
ax.text(-200, 120, '(b)', color='black', fontsize=size_large)

# Plot force in z-direction
ax = fig.add_subplot(gs2[0, 0])
ax.set_xlabel(r'Twist $w$ (-)', fontsize=size_large)
ax.set_ylabel(r'Force in z-direction (-)', fontsize=size_large)
ax.tick_params(axis='both', which='major', labelsize=tick_size)

ax.plot(twists, force_cyl, marker='x', label='cylinder', color=color1)
ax.plot(twists, force_angle1, marker='x', label=f'angle=0.25', color=color2)
ax.plot(twists, force_angle2, marker='x', label=f'angle=0.2', color=color3)

ax.text(-0.3, 0.035, '(c)', color='black', fontsize=size_large)

# Legend
#ax.legend(fontsize=size_large)
ax.text(-0.175, 0.002, 'cylinder', color=color1, fontsize=size_large)
ax.text(0.12, 0.033, 'angle=0.25', color=color2, fontsize=size_large, ha='right')
ax.text(-0.21, -0.019, 'angle=0.2', color=color3, fontsize=size_large)


### ----- Finish ----- ###
name = 'results/chiral1.pdf'
fig.savefig(name, bbox_inches='tight')
#plt.show()
plt.close(fig)

##################################################################
### ------------------ Plot result chiral 2 ------------------ ###
##################################################################
### ----- Parameters ----- ###
# Colors
color1 = 'red'
color2 = 'blue'

# Fontsize
size_large = 15
tick_size = 11
### ----- Read data ----- ###
# Folders
folder = f'results/chiral2/'

# 3D image of geometries
im = np.asarray(Image.open((folder + 'geometry.png')))

# Read force in z-direction
name_file = folder + 'comp_cylinder_Nxyz=30/'
name_file += 'data.txt'
twists = np.loadtxt(name_file, skiprows=1, max_rows=1)
force_cyl = np.loadtxt(name_file, skiprows=3, max_rows=1)
force_chi = np.loadtxt(name_file, skiprows=5, max_rows=1)


### ----- Plot data ----- ###
fig = plt.figure(figsize=(7, 5))
gs = fig.add_gridspec(1, 2, wspace = 0.5)

# Plot 3D images
ax = fig.add_subplot(gs[0, 0])
ax.axis('off')
implot = ax.imshow(im)
ax.text(-180, 120, '(a)', color='black', fontsize=size_large)


# Plot force in z-direction
ax = fig.add_subplot(gs[0, 1])
ax.set_xlabel(r'Twist $w$ (-)', fontsize=size_large)
ax.set_ylabel(r'Force in z-direction (-)', fontsize=size_large)
#ax.set_xlabel(r'Drehwinkel pro Länge $w$ (-)', fontsize=size_large)
#ax.set_ylabel(r'Kraft in z-direction (-)', fontsize=size_large)
ax.tick_params(axis='both', which='major', labelsize=tick_size)

ax.plot(twists, force_cyl, marker='x', label='cylinder', color=color1)
ax.plot(twists, force_chi, marker='x', label=f'chiral', color=color2)

ax.text(-0.3, 0.007, '(b)', color='black', fontsize=size_large)

# Legend
#ax.legend(fontsize=size_large)
ax.text(-0.175, 0.001, 'cylinder', color=color1, fontsize=size_large)
ax.text(0.05, 0.005, 'chiral', color=color2, fontsize=size_large, ha='right')
#ax.text(-0.175, 0.001, 'Zylinder', color=color1, fontsize=size_large)
#ax.text(0.07, 0.005, 'Chirale \n Struktur', color=color2, fontsize=size_large, ha='right')


### ----- Finish ----- ###
name = 'results/chiral2.pdf'
fig.savefig(name, bbox_inches='tight')
# plt.show()
plt.close(fig)

##################################################################
### ------------ Plot displacement chiral 2 (2D) ------------- ###
##################################################################
### ----- Parameters ----- ###
figsize = (9, 5)

# Fontsize
fontsize = 12
tick_size = 11
fontsize_large = 14

# Geometry
Nxyz = 40
nb_quad_pts = 5

# Loading
twist = 0.05

# Scaling factor for displacement
scale = 5
scale2 = 50

# Lines for undeformed geometry
linewidth = 1.5
linecolor = 'black'

### ----- Read data ----- ###
# Folder
folder = f'results/chiral2/mult_unit_cells/Nuc_z=1_Nxyz={Nxyz}_twist={twist}/'
folder += f'N_uc=1_strains/'

# Read strain
strain = np.empty((3, 3, 5, Nxyz, Nxyz, Nxyz))
for i_quad in range(nb_quad_pts):
    name = folder + f'quad_pt_{i_quad}_entry_00.npy'
    strain[0, 0] = np.load(name)
    name = folder + f'quad_pt_{i_quad}_entry_01.npy'
    strain[0, 1] = np.load(name)
    strain[1, 0] = strain[0, 1]
    name = folder + f'quad_pt_{i_quad}_entry_02.npy'
    strain[0, 2] = np.load(name)
    strain[2, 0] = strain[0, 2]
    name = folder + f'quad_pt_{i_quad}_entry_11.npy'
    strain[1, 1] = np.load(name)
    name = folder + f'quad_pt_{i_quad}_entry_12.npy'
    strain[1, 2] = np.load(name)
    strain[2, 1] = strain[1, 2]
    name = folder + f'quad_pt_{i_quad}_entry_22.npy'
    strain[2, 2] = np.load(name)


### ----- Calculate displacement ----- ###
# Geometric parameters
a = 0.5
thickness = 0.06 * a
radius_out = 0.4 * a
radius_inn = 0.34 * a
angle_mat = np.pi * 35 / 180

# Other parameters for calculation
nb_grid_pts = [Nxyz, Nxyz, Nxyz]
formulation = µ.Formulation.small_strain
gradient, weights = µ.linear_finite_elements.gradient_3d_5tet
F0 = np.eye(3)
fft = 'fftw'
Young = 2600
Poisson = 0.4

# Mask describing geometry
mask, lengths =\
    geo.chiral_2_mult_unit_cell([1, 1, 1], nb_grid_pts, a, radius_out,
                                radius_inn, thickness, alpha=angle_mat)

# Create + initialize muSpectre cell
cell = µ.Cell(nb_grid_pts, lengths, formulation, gradient,
              weights=weights, fft=fft)
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
eigen_class = EigenStrain(cell.pixels, twist, lengths, nb_grid_pts,
                          x_rot_axis, y_rot_axis)

# Calculate displacement of periodic strain
strain_no_eigen = strain.copy()
eigen_class.remove_eigen_strain_func(strain_no_eigen)
[x_0, y_0, z_0], [x_displ, y_displ, z_displ] \
    = get_complemented_positions("0d", cell, strain_array=strain_no_eigen, F0=F0,
                                 periodically_complemented=True)
displ_fluct = np.asarray([x_displ.copy(), y_displ.copy(), z_displ.copy()])
norm_displ_fluct = np.linalg.norm(displ_fluct, axis=0)

# Add displacement of rotational strain
x = np.linspace(0, lengths[0], nb_grid_pts[0]+1, endpoint=True)
y = np.linspace(0, lengths[1], nb_grid_pts[1]+1, endpoint=True)
z = np.linspace(0, lengths[2], nb_grid_pts[2]+1, endpoint=True)
helper = - twist * np.einsum('i,j->ij', y-y_rot_axis, z)
x_displ += helper[None, :, :]
helper = twist * np.einsum('i,j->ij', x-x_rot_axis, z)
y_displ += helper[:, None, :]


pos_initial = np.asarray([x_0, y_0, z_0])
displ = np.asarray([x_displ, y_displ, z_displ])
pos_displ = pos_initial + scale * displ
norm_displ = np.linalg.norm(displ, axis=0)
norm_displ_xy = np.linalg.norm(displ[0:2], axis=0)

### ----- Plot displacement: Top ----- ###
# Prepare figure
fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'wspace':0.3})
fig.suptitle(f'Displacement at Top', fontsize=fontsize_large)
for i in range(2):
    axes[i].set_aspect('equal')
    axes[i].set_xlabel('Position x', fontsize=fontsize)
    axes[i].set_ylabel('Position y', fontsize=fontsize)

axes[0].set_title(f'Complete (scaling={scale})', fontsize=fontsize)
axes[1].set_title(f'Fluct (scaling={scale2})', fontsize=fontsize)

# Mask for neglecting voxels with void
mask = mask.reshape(nb_grid_pts, order='F')
mask_2D = mask[:, :, -1]
helper = (mask_2D != 1)
mask_points = np.full([Nxyz+1, Nxyz+1], True)
mask_points[:-1, :-1] = helper
mask_points[1:, :-1] = np.logical_and(helper, mask_points[1:, :-1])
mask_points[:-1, 1:] = np.logical_and(helper, mask_points[:-1, 1:])
mask_points[1:, 1:] = np.logical_and(helper, mask_points[1:, 1:])

# Plot deformed geometry
X = pos_displ[0, :, :, -1]
Y = pos_displ[1, :, :, -1]

Z = norm_displ[:, :, -1]
Z = np.ma.masked_array(Z, mask_points)
pm = axes[0].pcolormesh(X, Y, Z, shading='gouraud')
cbar = fig.colorbar(pm, ax=axes[0])
cbar.ax.set_ylabel(r'Norm of displacement $u$ (-)',
                   rotation=-90, va='bottom', fontsize=fontsize)

# Some voxels can not be masked directly
axes[0].pcolormesh(X, Y, np.ma.masked_array(mask_2D, mask_2D),
                   cmap=mpl.colors.ListedColormap(['white']))

# Plot deformed geometry of fluctuating disp
X = pos_initial[0, :, :, -1] + scale2 * displ_fluct[0, :, :, -1]
Y = pos_initial[1, :, :, -1] + scale2 * displ_fluct[1, :, :, -1]

Z = norm_displ_fluct[:, :, -1]
Z = np.ma.masked_array(Z, mask_points)
pm = axes[1].pcolormesh(X, Y, Z, shading='gouraud')
cbar = fig.colorbar(pm, ax=axes[1])
cbar.ax.set_ylabel(r'Norm of displacement $u$ (-)',
                   rotation=-90, va='bottom', fontsize=fontsize)

# Some voxels can not be masked directly
axes[1].pcolormesh(X, Y, np.ma.masked_array(mask_2D, mask_2D),
                   cmap=mpl.colors.ListedColormap(['white']))

# Plot initial geometry
X = pos_initial[0, :, :, -1]
Y = pos_initial[1, :, :, -1]

for i_ax in range(2):
    for i in range(Nxyz-1):
        for j in range(Nxyz-1):
            if mask[i, j, -1] != mask[i+1, j, -1]:
                axes[i_ax].plot([X[i+1, j], X[i+1, j+1]], [Y[i+1, j], Y[i+1, j+1]],
                                linewidth = linewidth, color = linecolor)
            if mask[i, j, -1] != mask[i, j+1, -1]:
                axes[i_ax].plot([X[i, j+1], X[i+1, j+1]], [Y[i, j+1], Y[i+1, j+1]],
                                linewidth = linewidth, color = linecolor)


# Save and show
# plt.show()
name = 'results/deformed_chiral_2_top.pdf'
fig.savefig(name, bbox_inches='tight')
plt.close(fig)

### ----- Plot displacement: Top with Comsol ----- ###
# Plot of Comsol results
twist_comsol = 50
name_comsol = f'../fem_tests/rot_1x1x1_unit_cells_w={twist_comsol}_displ_top.png'

# Prepare figure
fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'wspace':0.3})
fig.suptitle(f'Displacement at Top in mm (scaling={scale})', fontsize=fontsize_large)
axes[0].set_aspect('equal')
axes[0].set_xlabel('Position x', fontsize=fontsize)
axes[0].set_ylabel('Position y', fontsize=fontsize)

axes[1].axis('off')

axes[0].set_title(f'muSpectre: rot_angle={twist}/mm', fontsize=fontsize)
axes[1].set_title(f'Comsol (uz free): rot_angle={twist_comsol}/m', fontsize=fontsize)

# Mask for neglecting voxels with void
mask = mask.reshape(nb_grid_pts, order='F')
mask_2D = mask[:, :, -1]
helper = (mask_2D != 1)
mask_points = np.full([Nxyz+1, Nxyz+1], True)
mask_points[:-1, :-1] = helper
mask_points[1:, :-1] = np.logical_and(helper, mask_points[1:, :-1])
mask_points[:-1, 1:] = np.logical_and(helper, mask_points[:-1, 1:])
mask_points[1:, 1:] = np.logical_and(helper, mask_points[1:, 1:])

# Plot deformed geometry
X = pos_displ[0, :, :, -1]
Y = pos_displ[1, :, :, -1]

Z = norm_displ[:, :, -1]
Z = np.ma.masked_array(Z, mask_points)
pm = axes[0].pcolormesh(X, Y, Z, shading='gouraud')
cbar = fig.colorbar(pm, ax=axes[0])
cbar.ax.set_ylabel(r'Norm of displacement $u$ (mm)',
                   rotation=-90, va='bottom', fontsize=fontsize)

# Some voxels can not be masked directly
axes[0].pcolormesh(X, Y, np.ma.masked_array(mask_2D, mask_2D),
                   cmap=mpl.colors.ListedColormap(['white']))

# Plot initial geometry
X = pos_initial[0, :, :, -1]
Y = pos_initial[1, :, :, -1]

for i in range(Nxyz-1):
    for j in range(Nxyz-1):
        if mask[i, j, -1] != mask[i+1, j, -1]:
            axes[0].plot([X[i+1, j], X[i+1, j+1]], [Y[i+1, j], Y[i+1, j+1]],
                            linewidth = linewidth, color = linecolor)
        if mask[i, j, -1] != mask[i, j+1, -1]:
            axes[0].plot([X[i, j+1], X[i+1, j+1]], [Y[i, j+1], Y[i+1, j+1]],
                            linewidth = linewidth, color = linecolor)

# Plot displacement of Comsol
im = np.asarray(Image.open(name_comsol))
implot = axes[1].imshow(im)


# Save and show
#plt.show()
name = 'results/deformed_chiral_2_top_comsol.pdf'
fig.savefig(name, bbox_inches='tight')
plt.close(fig)

### ----- Plot displacement: Side with Comsol ----- ###
i_y = -2
# Plot of Comsol results
twist_comsol = 50
name_comsol = f'../fem_tests/rot_1x1x1_unit_cells_w={twist_comsol}_displ_side.png'

# Prepare figure
fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'wspace':0.3})
fig.suptitle(f'Displacement at Top in mm (scaling={scale})', fontsize=fontsize_large)
axes[0].set_aspect('equal')
axes[0].set_xlabel('Position x', fontsize=fontsize)
axes[0].set_ylabel('Position z', fontsize=fontsize)

axes[1].axis('off')

axes[0].set_title(f'muSpectre: rot_angle={twist}/mm', fontsize=fontsize)
axes[1].set_title(f'Comsol (uz free): rot_angle={twist_comsol}/m', fontsize=fontsize)

# Mask for neglecting voxels with void
mask = mask.reshape(nb_grid_pts, order='F')
mask_2D = mask[:, i_y, :]
helper = (mask_2D != 1)
mask_points = np.full([Nxyz+1, Nxyz+1], True)
mask_points[:-1, :-1] = helper
mask_points[1:, :-1] = np.logical_and(helper, mask_points[1:, :-1])
mask_points[:-1, 1:] = np.logical_and(helper, mask_points[:-1, 1:])
mask_points[1:, 1:] = np.logical_and(helper, mask_points[1:, 1:])

# Plot deformed geometry
X = pos_displ[0, :, i_y, :]
Z = pos_displ[2, :, i_y, :]

n = norm_displ[:, i_y, :]
n = np.ma.masked_array(n, mask_points)
pm = axes[0].pcolormesh(X, Z, n, shading='gouraud')
cbar = fig.colorbar(pm, ax=axes[0])
cbar.ax.set_ylabel(r'Norm of displacement $u$ (mm)',
                   rotation=-90, va='bottom', fontsize=fontsize)

# Some voxels can not be masked directly
axes[0].pcolormesh(X, Z, np.ma.masked_array(mask_2D, mask_2D),
                   cmap=mpl.colors.ListedColormap(['white']))

# Plot initial geometry
X = pos_initial[0, :, i_y, :]
Z = pos_initial[2, :, i_y, :]

for i in range(Nxyz-1):
    if mask[i, i_y, 0] == 1:
        axes[0].plot([X[i, 0], X[i+1, 0]], [Z[i, 0], Z[i+1, 0]],
                     linewidth = linewidth, color = linecolor)
    if mask[i, i_y, -1] == 1:
        axes[0].plot([X[i, -1], X[i+1, -1]], [Z[i, -1], Z[i+1, -1]],
                     linewidth = linewidth, color = linecolor)
    for j in range(Nxyz-1):
        if mask[i, i_y, j] != mask[i+1, i_y, j]:
            axes[0].plot([X[i+1, j], X[i+1, j+1]], [Z[i+1, j], Z[i+1, j+1]],
                            linewidth = linewidth, color = linecolor)
        if mask[i, i_y, j] != mask[i, i_y, j+1]:
            axes[0].plot([X[i, j+1], X[i+1, j+1]], [Z[i, j+1], Z[i+1, j+1]],
                            linewidth = linewidth, color = linecolor)

# Plot displacement of Comsol
im = np.asarray(Image.open(name_comsol))
implot = axes[1].imshow(im)


# Save and show
#plt.show()
name = 'results/deformed_chiral_2_side_comsol.pdf'
fig.savefig(name, bbox_inches='tight')
plt.close(fig)

"""
@file   plot_chiral_metamaterial2.py

@author Indre  Joedicke <indre.joedicke@imtek.uni-freiburg.de>

@date   28 Fev 2025

@brief  Make plots for calculations of chiral_metamaterial2.py

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
import shutil
sys.path.insert(0, os.path.join(os.getcwd(), "../../muspectre/builddir/language_bindings/libmugrid/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../../muspectre/builddir/language_bindings/libmufft/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../../muspectre/builddir/language_bindings/python"))

import numpy as np
import matplotlib.pyplot as plt
from time import time
import matplotlib as mpl
from PIL import Image

import muSpectre as µ
from NuMPI import MPI
from NuMPI.Tools import Reduction
from NuMPI.IO import save_npy

from muChirality.EigenStrainTorsion import EigenStrain
from muChirality.CalculationsTorsion import calculations
from muSpectre.gradient_integration import get_complemented_positions
import muChirality.Geometries as geo

def plots_2D_mult_unit_cells():
    ### ----- Parameters ----- ###
    # Folder
    Nz = 1
    N = 40
    twist = 0.05
    folder = f'results/chiral2/mult_unit_cells/Nuc_z={Nz}_'
    folder += f'Nxyz={N}_twist={twist}/'

    # For plotting
    figsize = (9, 5)

    fontsize = 12
    tick_size = 11
    fontsize_large = 14

    # Scaling factor for displacement
    scale = 5
    scale2 = 50

    # Lines for undeformed geometry
    linewidth = 1.5
    linecolor = 'black'

    ### ----- Definitions ----- ###
    # Discretization
    nb_quad_pts = 5

    # Geometry
    a = 0.5
    thickness = 0.06 * a
    radius_out = 0.4 * a
    radius_inn = 0.34 * a
    angle_mat = np.pi * 35 / 180

    # Other parameters for calculation
    nb_grid_pts = [N, N, N]
    formulation = µ.Formulation.small_strain
    gradient, weights = µ.linear_finite_elements.gradient_3d_5tet
    F0 = np.eye(3)
    fft = 'fftw'
    Young = 2600
    Poisson = 0.4

    ### ----- Calculate displacement for 1x1xNz unit cells----- ###
    # Read strain
    folder_strain = folder + 'N_uc=1_strains/'
    strain = np.empty((3, 3, 5, N, N, N))
    for i_quad in range(nb_quad_pts):
        name = folder_strain + f'quad_pt_{i_quad}_entry_00.npy'
        strain[0, 0] = np.load(name)
        name = folder_strain + f'quad_pt_{i_quad}_entry_01.npy'
        strain[0, 1] = np.load(name)
        strain[1, 0] = strain[0, 1]
        name = folder_strain + f'quad_pt_{i_quad}_entry_02.npy'
        strain[0, 2] = np.load(name)
        strain[2, 0] = strain[0, 2]
        name = folder_strain + f'quad_pt_{i_quad}_entry_11.npy'
        strain[1, 1] = np.load(name)
        name = folder_strain + f'quad_pt_{i_quad}_entry_12.npy'
        strain[1, 2] = np.load(name)
        strain[2, 1] = strain[1, 2]
        name = folder_strain + f'quad_pt_{i_quad}_entry_22.npy'
        strain[2, 2] = np.load(name)

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
    # norm_displ_xy = np.linalg.norm(displ[0:2], axis=0)

    ### ----- Plot displ: Top ----- ###
    # Prepare figure
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'wspace':0.4})
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
    mask_points = np.full([N+1, N+1], True)
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

    # Plot deformed geometry of fluctuating disp
    X = pos_initial[0, :, :, -1] + scale2 * displ_fluct[0, :, :, -1]
    Y = pos_initial[1, :, :, -1] + scale2 * displ_fluct[1, :, :, -1]

    Z = norm_displ_fluct[:, :, -1]
    Z = np.ma.masked_array(Z, mask_points)
    pm = axes[1].pcolormesh(X, Y, Z, shading='gouraud')
    cbar = fig.colorbar(pm, ax=axes[1])
    cbar.ax.set_ylabel(r'Norm of fluct displ $\tilde{u}$ (mm)',
                       rotation=-90, va='bottom', fontsize=fontsize)

    # Some voxels can not be masked directly
    axes[1].pcolormesh(X, Y, np.ma.masked_array(mask_2D, mask_2D),
                       cmap=mpl.colors.ListedColormap(['white']))

    # Plot initial geometry
    X = pos_initial[0, :, :, -1]
    Y = pos_initial[1, :, :, -1]

    for i_ax in range(2):
        for i in range(N-1):
            for j in range(N-1):
                if mask[i, j, -1] != mask[i+1, j, -1]:
                    axes[i_ax].plot([X[i+1, j], X[i+1, j+1]], [Y[i+1, j], Y[i+1, j+1]],
                                    linewidth = linewidth, color = linecolor)
                if mask[i, j, -1] != mask[i, j+1, -1]:
                    axes[i_ax].plot([X[i, j+1], X[i+1, j+1]], [Y[i, j+1], Y[i+1, j+1]],
                                    linewidth = linewidth, color = linecolor)


    # Save and show
    #plt.show()
    name = folder + 'displ_fluct_top.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)

    ### ----- Plot displ with Comsol: Top ----- ###
    # Comsol plot
    name_comsol = f'../fem_tests/Fixed_uz=0/1x1x1_unit_cells_displ_top_scale={scale}.png'

    # Prepare figure
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'wspace':0.3})
    fig.suptitle(f'Displacement at Top in mm (scaling={scale})', fontsize=fontsize_large)
    axes[0].set_aspect('equal')
    axes[0].set_xlabel('Position x', fontsize=fontsize)
    axes[0].set_ylabel('Position y', fontsize=fontsize)

    axes[1].axis('off')

    axes[0].set_title(f'muSpectre', fontsize=fontsize)
    axes[1].set_title(f'Comsol', fontsize=fontsize)

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

    for i in range(N-1):
        for j in range(N-1):
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
    name = folder + 'displ_comsol_top.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)

    ### ----- Plot displ: Side ----- ###
    i_y = -2

    # Prepare figure
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'wspace':0.4})
    fig.suptitle(f'Displacement at Side', fontsize=fontsize_large)
    for i_ax in range(2):
        axes[i_ax].set_aspect('equal')
        axes[i_ax].set_xlabel('Position x', fontsize=fontsize)
        axes[i_ax].set_ylabel('Position z', fontsize=fontsize)

    axes[0].set_title(f'Complete (scaling={scale})', fontsize=fontsize)
    axes[1].set_title(f'Fluct (scaling={scale})', fontsize=fontsize)

    # Mask for neglecting voxels with void
    mask = mask.reshape(nb_grid_pts, order='F')
    mask_2D = mask[:, i_y, :]
    helper = (mask_2D != 1)
    mask_points = np.full([N+1, N+1], True)
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

    # Plot deformed geometry of fluctuating disp
    X = pos_initial[0, :, i_y, :] + scale * displ_fluct[0, :, i_y, :]
    Z = pos_initial[2, :, i_y, :] + scale * displ_fluct[2, :, i_y, :]

    n = norm_displ_fluct[:, i_y, :]
    n = np.ma.masked_array(n, mask_points)
    pm = axes[1].pcolormesh(X, Z, n, shading='gouraud')
    cbar = fig.colorbar(pm, ax=axes[1])
    cbar.ax.set_ylabel(r'Norm of fluct displ $\tilde{u}$ (mm)',
                       rotation=-90, va='bottom', fontsize=fontsize)

    # Some voxels can not be masked directly
    axes[1].pcolormesh(X, Z, np.ma.masked_array(mask_2D, mask_2D),
                       cmap=mpl.colors.ListedColormap(['white']))

    # Plot initial geometry
    X = pos_initial[0, :, i_y, :]
    Z = pos_initial[2, :, i_y, :]

    for i_ax in range(2):
        for i in range(N-1):
            if mask[i, i_y, 0] == 1:
                axes[i_ax].plot([X[i, 0], X[i+1, 0]], [Z[i, 0], Z[i+1, 0]],
                                linewidth = linewidth, color = linecolor)
            if mask[i, i_y, -1] == 1:
                axes[i_ax].plot([X[i, -1], X[i+1, -1]], [Z[i, -1], Z[i+1, -1]],
                                linewidth = linewidth, color = linecolor)
            for j in range(N-1):
                if mask[i, i_y, j] != mask[i+1, i_y, j]:
                    axes[i_ax].plot([X[i+1, j], X[i+1, j+1]], [Z[i+1, j], Z[i+1, j+1]],
                                    linewidth = linewidth, color = linecolor)
                if mask[i, i_y, j] != mask[i, i_y, j+1]:
                    axes[i_ax].plot([X[i, j+1], X[i+1, j+1]], [Z[i, j+1], Z[i+1, j+1]],
                                    linewidth = linewidth, color = linecolor)

    # Save and show
    #plt.show()
    name = folder + 'displ_fluct_side.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)

    ### ----- Plot displ with comsol: Side ----- ###
    # Comsol plot
    name_comsol = f'../fem_tests/Fixed_uz=0/1x1x1_unit_cells_displ_side_scale={scale}.png'

    # Prepare figure
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'wspace':0.3})
    fig.suptitle(f'Displacement at Side in mm (scaling={scale})', fontsize=fontsize_large)
    axes[0].set_aspect('equal')
    axes[0].set_xlabel('Position x', fontsize=fontsize)
    axes[0].set_ylabel('Position z', fontsize=fontsize)

    axes[1].axis('off')

    axes[0].set_title(f'muSpectre', fontsize=fontsize)
    axes[1].set_title(f'Comsol', fontsize=fontsize)

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

    for i in range(N-1):
        if mask[i, i_y, 0] == 1:
            axes[0].plot([X[i, 0], X[i+1, 0]], [Z[i, 0], Z[i+1, 0]],
                         linewidth = linewidth, color = linecolor)
        if mask[i, i_y, -1] == 1:
            axes[0].plot([X[i, -1], X[i+1, -1]], [Z[i, -1], Z[i+1, -1]],
                         linewidth = linewidth, color = linecolor)
        for j in range(N-1):
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
    name = folder + 'displ_comsol_side.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)

    ### ----- Plot fluct displ with comsol: Side ----- ###
    # Comsol plot
    name_comsol = f'../fem_tests/Fixed_uz=0/1x1x1_unit_cells_disp_fluct_side_scale={scale}.png'

    # Prepare figure
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'wspace':0.3})
    fig.suptitle(f'Fluct Displacement at Side (scaling={scale})', fontsize=fontsize_large)
    axes[0].set_aspect('equal')
    axes[0].set_xlabel('Position x', fontsize=fontsize)
    axes[0].set_ylabel('Position z', fontsize=fontsize)

    axes[1].axis('off')

    axes[0].set_title(f'muSpectre', fontsize=fontsize)
    axes[1].set_title(f'Comsol', fontsize=fontsize)

    # Plot deformed geometry of fluctuating disp
    X = pos_initial[0, :, i_y, :] + scale * displ_fluct[0, :, i_y, :]
    Z = pos_initial[2, :, i_y, :] + scale * displ_fluct[2, :, i_y, :]

    n = norm_displ_fluct[:, i_y, :]
    n = np.ma.masked_array(n, mask_points)
    pm = axes[0].pcolormesh(X, Z, n, shading='gouraud')
    cbar = fig.colorbar(pm, ax=axes[0])
    cbar.ax.set_ylabel(r'Norm of fluct displ $\tilde{u}$ (mm)',
                       rotation=-90, va='bottom', fontsize=fontsize)

    # Some voxels can not be masked directly
    axes[0].pcolormesh(X, Z, np.ma.masked_array(mask_2D, mask_2D),
                       cmap=mpl.colors.ListedColormap(['white']))

    # Plot initial geometry
    X = pos_initial[0, :, i_y, :]
    Z = pos_initial[2, :, i_y, :]

    for i in range(N-1):
        if mask[i, i_y, 0] == 1:
            axes[0].plot([X[i, 0], X[i+1, 0]], [Z[i, 0], Z[i+1, 0]],
                         linewidth = linewidth, color = linecolor)
        if mask[i, i_y, -1] == 1:
            axes[0].plot([X[i, -1], X[i+1, -1]], [Z[i, -1], Z[i+1, -1]],
                         linewidth = linewidth, color = linecolor)
        for j in range(N-1):
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
    name = folder + 'displ_fluct_comsol_side.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)

    ### ----- Average stress (z-component) in each voxel ----- ###
    # Read stress
    folder_stress = folder + 'N_uc=1_stresses/'
    stress = np.empty((3, 3, 5, N, N, N))
    for i_quad in range(nb_quad_pts):
        name = folder_stress + f'quad_pt_{i_quad}_entry_00.npy'
        stress[0, 0] = np.load(name)
        name = folder_stress + f'quad_pt_{i_quad}_entry_01.npy'
        stress[0, 1] = np.load(name)
        stress[1, 0] = stress[0, 1]
        name = folder_stress + f'quad_pt_{i_quad}_entry_02.npy'
        stress[0, 2] = np.load(name)
        stress[2, 0] = stress[0, 2]
        name = folder_stress + f'quad_pt_{i_quad}_entry_11.npy'
        stress[1, 1] = np.load(name)
        name = folder_strain + f'quad_pt_{i_quad}_entry_12.npy'
        stress[1, 2] = np.load(name)
        stress[2, 1] = stress[1, 2]
        name = folder_stress + f'quad_pt_{i_quad}_entry_22.npy'
        stress[2, 2] = np.load(name)

    # Average in voxel
    stress = np.average(stress, axis=2, weights=(2, 1, 1, 1, 1))

    # Consider z-component
    stress_z = stress[2, 2]

    ### ----- Plot stress with Comsol: Top ----- ###
    # Comsol plot
    name_comsol = f'../fem_tests/Fixed_uz=0/1x1x1_unit_cells_stress_top_scale={scale}.png'

    # Prepare figure
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'wspace':0.3})
    fig.suptitle(f'Stress_zz at Top in MPa (scaling={scale})', fontsize=fontsize_large)
    axes[0].set_aspect('equal')
    axes[0].set_xlabel('Position x', fontsize=fontsize)
    axes[0].set_ylabel('Position y', fontsize=fontsize)

    axes[1].axis('off')

    axes[0].set_title(f'muSpectre', fontsize=fontsize)
    axes[1].set_title(f'Comsol', fontsize=fontsize)

    # Mask for neglecting voxels with void
    mask = mask.reshape(nb_grid_pts, order='F')
    mask_2D = mask[:, :, -1]
    helper = (mask_2D != 1)

    # Plot deformed geometry
    X = pos_displ[0, :, :, -1]
    Y = pos_displ[1, :, :, -1]

    Z = stress_z[:, :, -1]
    Z = np.ma.masked_array(Z, helper)
    pm = axes[0].pcolormesh(X, Y, Z, shading='flat')
    cbar = fig.colorbar(pm, ax=axes[0])
    cbar.ax.set_ylabel(r'Stress_zz (MPa)',
                       rotation=-90, va='bottom', fontsize=fontsize)

    # Plot initial geometry
    X = pos_initial[0, :, :, -1]
    Y = pos_initial[1, :, :, -1]

    for i in range(N-1):
        for j in range(N-1):
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
    name = folder + 'stress_comsol_top.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)

    ### ----- Plot stress with Comsol: Top ----- ###
    # Comsol plot
    name_comsol = f'../fem_tests/Fixed_uz=0/1x1x1_unit_cells_stress_side_scale={scale}.png'

    # Prepare figure
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'wspace':0.3})
    fig.suptitle(f'Stress_zz at Side in MPa (scaling={scale})', fontsize=fontsize_large)
    axes[0].set_aspect('equal')
    axes[0].set_xlabel('Position x', fontsize=fontsize)
    axes[0].set_ylabel('Position y', fontsize=fontsize)

    axes[1].axis('off')

    axes[0].set_title(f'muSpectre', fontsize=fontsize)
    axes[1].set_title(f'Comsol', fontsize=fontsize)

    # Mask for neglecting voxels with void
    mask = mask.reshape(nb_grid_pts, order='F')
    mask_2D = mask[:, i_y, :]
    helper = (mask_2D != 1)

    # Plot deformed geometry
    X = pos_displ[0, :, i_y, :]
    Z = pos_displ[2, :, i_y, :]

    s = stress_z[:, i_y, :]
    s = np.ma.masked_array(s, helper)
    pm = axes[0].pcolormesh(X, Z, s, shading='flat')
    cbar = fig.colorbar(pm, ax=axes[0])
    cbar.ax.set_ylabel(r'Stress_zz (MPa)',
                       rotation=-90, va='bottom', fontsize=fontsize)

    # Plot initial geometry
    X = pos_initial[0, :, i_y, :]
    Z = pos_initial[2, :, i_y, :]

    for i in range(N-1):
        if mask[i, i_y, 0] == 1:
            axes[0].plot([X[i, 0], X[i+1, 0]], [Z[i, 0], Z[i+1, 0]],
                         linewidth = linewidth, color = linecolor)
        if mask[i, i_y, -1] == 1:
            axes[0].plot([X[i, -1], X[i+1, -1]], [Z[i, -1], Z[i+1, -1]],
                         linewidth = linewidth, color = linecolor)
        for j in range(N-1):
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
    name = folder + 'stress_comsol_side.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)

def plot_on_paper_results(folder):
    imfile = 'results_paper.jpg'
    # Read data
    data_file = folder + 'data.txt'
    data = np.loadtxt(data_file, skiprows=1)
    N_uc_list = data[:, 0]
    E_eff_z = data[:, 2]
    twist_per_strain = data[:, 4]

    # Other parameters
    dpi = 100
    color = 'green'
    marker = 'x'
    markersize = 12
    linewidth = 4

    ### ----- Plot data ----- ###
    # Prepare figure
    im = image.imread(imfile)
    fig, axim = plt.subplots(figsize=(im.shape[1]/dpi, im.shape[0]/dpi), dpi=dpi)
    axim.imshow(im, aspect='equal')
    axim.axis('off')

    # Plot E_eff_z
    ax = fig.add_axes([0.25, 0.17, 0.635, 0.295], facecolor='None')
    ax.set_xlim([0.49, 5.49])
    ax.set_ylim([0, 45])
    ax.plot(nb_unit_cells, E_eff_z, marker=marker, linewidth=linewidth,
            markersize=markersize, color=color)
    ax.axis('off')

    # Plot twist per strain
    ax = fig.add_axes([0.25, 0.474, 0.635, 0.297], facecolor='None')
    ax.set_xlim([0.49, 5.49])
    ax.set_ylim([-0.21, 2.48])
    ax.plot(nb_unit_cells, twist_per_strain, marker=marker, linewidth=linewidth,
        markersize=markersize, color=color)
    ax.axis('off')

    name = folder + 'plot_on_paper.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)


def plot_convergences():
    # Parameters
    mpi_size_list = [126, 190, 630, 950]
    folder = 'results_nemo/chiral_mesh_refinement_mpi'

    # Prepare figures
    fig_force, ax_force = plt.subplots()
    title = f'Mesh refinement: 1x1x1 UC'
    fig_force.suptitle(title)
    ax_force.set_xlabel('nb_grid_pts')
    ax_force.set_ylabel('Average force (N)')

    fig_time, ax_time = plt.subplots()
    fig_time.suptitle(title)
    ax_time.set_xlabel('nb_grid_pts')
    ax_time.set_ylabel('Calculation time (min)')

    for mpi_size in mpi_size_list:
        # Read data
        name = folder + f'{mpi_size}/data.txt'
        data = np.loadtxt(name, skiprows=1)
        N = data[:, 0]

        # Plot convergence of average force
        ax_force.plot(N, data[:, 1], label=f'MPI_size={mpi_size}', marker='x')

        # Plot calculation time
        time = data[:, 3] / 60
        ax_time.plot(N, time, label=f'MPI_size={mpi_size}', marker='x')

    # Legends
    ax_force.legend()
    ax_time.legend()

    # Save and show
    name = 'results_nemo/convergence_average_force.pdf'
    fig_force.savefig(name, bbox_inches='tight')
    name = 'results_nemo/calculation_time.pdf'
    fig_time.savefig(name, bbox_inches='tight')
    plt.show()
    plt.close(fig_force)
    plt.close(fig_time)


if __name__ == "__main__":
    # plots_2D_mult_unit_cells()
    plot_convergences()


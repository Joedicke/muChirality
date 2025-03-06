"""
@file   test_bending.py

@author Indre  Joedicke <indre.joedicke@imtek.uni-freiburg.de>

@date   14 Jan 2025

@brief  Test bending of a beam

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

# Default path of the library
sys.path.insert(0, os.path.join(os.getcwd(), "../../muspectre/builddir/language_bindings/libmugrid/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../../muspectre/builddir/language_bindings/libmufft/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../../muspectre/builddir/language_bindings/python"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import muSpectre as µ

from muChirality.EigenStrainBending import EigenStrainBending3D
from muSpectre.gradient_integration import get_complemented_positions

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['font.serif'] = ['Arial']
mpl.rcParams['font.cursive'] = ['Arial']
mpl.rcParams['font.size'] = '10'
mpl.rcParams['legend.fontsize'] = '10'
mpl.rcParams['xtick.labelsize'] = '9'
mpl.rcParams['ytick.labelsize'] = '9'
mpl.rcParams['svg.fonttype'] = 'none'

def plot_functions():
    ### ----- Parameters ----- ###
    # Geometry
    Lx = 1
    Ly = 0.1

    # Discretization of space
    nb_grid_pts = [51, 31] # Note: Odd grid_pts, so that one grid pt is one the neutral axis

    # Bending
    curvature = -1
    y_neutral_axis = Ly / 2
    Poisson = 1

    # For plotting
    color_undef = 'black'
    color_def = 'red'
    color_def_old = 'magenta'
    color_circle = 'blue'
    linewidth_circle = 0.75
    plot_circle = False

    ### ----- Calculations ----- ###
    # Radius of curvature
    radius_curv = 1 / curvature
    sign = np.sign(radius_curv)

    # Center of curvature
    x_curv = Lx / 2
    y_curv = y_neutral_axis + radius_curv

    # Undeformed positions
    X = np.linspace(0, Lx, nb_grid_pts[0])
    Y = np.linspace(0, Ly, nb_grid_pts[1])
    X, Y = np.meshgrid(X, Y)
    X = X.T
    Y = Y.T

    # Slope of neutral axis
    helper = radius_curv ** 2 - (X - x_curv) ** 2
    angle = (X - x_curv) * helper ** (-0.5)
    angle = -sign *  np.arctan(angle)

    # Deformed positions
    x_old = X + sign * (Y - y_neutral_axis) * np.sin(angle)
    y_old = y_curv - sign * np.sqrt(helper) + sign * (Y - y_neutral_axis) * np.cos(angle)

    x = X + curvature * X * Y
    y = Y - Poisson / 2 * curvature * Y**2 - curvature / 2 * X**2

    # Small strain tensor
    epsilon = np.empty((2, 2, nb_grid_pts[0], nb_grid_pts[1]))
    epsilon[0, 0] = curvature * Y
    epsilon[0, 1] = 0
    epsilon[1, 0] = epsilon[0, 1]
    epsilon[1, 1] = - Poisson * curvature * Y

    # Small strain tensor: Finite differences
    dx = X[1, 0] - X[0, 0]
    dy = Y[0, 1] - Y[0, 0]
    u_x = x - X
    u_y = y - Y
    epsilon_fin_diff = np.empty(epsilon.shape)
    epsilon_fin_diff[0, 0] = (np.roll(u_x, -1, axis=0) - np.roll(u_x, 1, axis=0)) / 2 / dx
    epsilon_fin_diff[0, 1] = 0.5 * (np.roll(u_x, -1, axis=1) - np.roll(u_x, 1, axis=1)) / 2 / dy
    epsilon_fin_diff[0, 1] += 0.5 * (np.roll(u_y, -1, axis=0) - np.roll(u_y, 1, axis=0)) / 2 / dx
    epsilon_fin_diff[1, 0] = epsilon_fin_diff[0, 1]
    epsilon_fin_diff[1, 1] = (np.roll(u_y, -1, axis=1) - np.roll(u_y, 1, axis=1)) / 2 / dy

    ### ----- Plot deformed beam (Check deformed position) ----- ###
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.axis('off')

    # Plot undeformed beam
    ax.plot([0, Lx], [0, 0], color=color_undef)
    ax.plot([Lx, Lx], [0, Ly], color=color_undef)
    ax.plot([0, Lx], [Ly, Ly], color=color_undef)
    ax.plot([0, 0], [0, Ly], color=color_undef)

    # Plot neutral axis
    index_neutral = nb_grid_pts[1] // 2
    if Y[0, index_neutral] != y_neutral_axis:
        raise ValueError('The neutral axis is not included in the discretized values.')
    ax.plot([X[0, 0], X[-1, 0]], [y_neutral_axis, y_neutral_axis], linestyle='--', color = color_undef)
    ax.plot(x[:, index_neutral], y[:, index_neutral], linestyle='--', color = color_def)

    # Plot deformed beam
    ax.plot(x[:, 0], y[:, 0], color=color_def)
    ax.plot(x[:, -1], y[:, -1], color=color_def)
    ax.plot(x[0, :], y[0, :], color=color_def)
    ax.plot(x[-1, :], y[-1, :], color=color_def)

    # Plot deformed beam (old)
    #ax.plot(x_old[:, 0], y_old[:, 0], color=color_def_old)
    #ax.plot(x_old[:, -1], y_old[:, -1], color=color_def_old)
    #ax.plot(x_old[0, :], y_old[0, :], color=color_def_old)
    #ax.plot(x_old[-1, :], y_old[-1, :], color=color_def_old)

    # Circle of curvature
    if plot_circle:
        circle = mpl.patches.Circle((x_curv, y_curv), radius_curv, edgecolor=color_circle,
                                    facecolor=None, fill=False, linewidth=linewidth_circle)
        ax.plot(x_curv, y_curv, marker='x', color=color_circle)
        ax.add_artist(circle)

    # plt.show()

    ### ----- Plot small strain tensor ----- ###
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle('Small strain tensor')

    # Plot epsilon_XX
    ax = axes[0]
    ax.set_xlabel('x')
    ax.set_ylabel(r'$\varepsilon_{xx}$')
    ax.plot(X[:, index_neutral], epsilon[0, 0, :, index_neutral], label='neutral axis')
    ax.plot(X[:, -1], epsilon[0, 0, :, -1], label='top')
    ax.plot(X[:, 0], epsilon[0, 0, :, 0], label='bottom')
    ax.plot(X[1:-1, index_neutral], epsilon_fin_diff[0, 0, 1:-1, index_neutral], label='fin_diff',
            linestyle='--', color='black')
    ax.plot(X[1:-1, -1], epsilon_fin_diff[0, 0, 1:-1, -1], linestyle='--', color='black')
    ax.plot(X[1:-1, 0], epsilon_fin_diff[0, 0, 1:-1, 0], linestyle='--', color='black')

    # Plot epsilon_XY
    ax = axes[2]
    ax.set_xlabel('x')
    ax.set_ylabel(r'$\varepsilon_{xy}$')
    ax.plot(X[:, index_neutral], epsilon[0, 1, :, index_neutral], label='neutral axis')
    ax.plot(X[:, -1], epsilon[0, 1, :, -2], label='top')
    ax.plot(X[:, 0], epsilon[0, 1, :, 1], label='bottom')
    ax.plot(X[1:-1, index_neutral], epsilon_fin_diff[0, 1, 1:-1, index_neutral], label='fin_diff',
            linestyle='--', color='black')
    ax.plot(X[1:-1, -2], epsilon_fin_diff[0, 1, 1:-1, -2], linestyle='--', color='black') # Attention to borders
    ax.plot(X[1:-1, 1], epsilon_fin_diff[0, 1, 1:-1, 1], linestyle='--', color='black')

    # Plot epsilon_YY
    ax = axes[1]
    ax.set_xlabel('x')
    ax.set_ylabel(r'$\varepsilon_{yy}$')
    ax.plot(X[:, index_neutral], epsilon[1, 1, :, index_neutral], label='neutral axis')
    ax.plot(X[:, -1], epsilon[1, 1, :, -2], label='top')
    ax.plot(X[:, 0], epsilon[1, 1, :, 1], label='bottom')
    ax.plot(X[1:-1, index_neutral], epsilon_fin_diff[1, 1, 1:-1, index_neutral], label='fin_diff',
            linestyle='--', color='black')
    ax.plot(X[1:-1, -2], epsilon_fin_diff[1, 1, 1:-1, -2], linestyle='--', color='black') # Attention to borders
    ax.plot(X[1:-1, 1], epsilon_fin_diff[1, 1, 1:-1, 1], linestyle='--', color='black')

    ax.legend()

    # Show plots
    plt.show()

def stress_indices(i_ax):
    if i_ax == 0:
        return (0, 0)
    elif i_ax == 1:
        return [0, 1]
    elif i_ax == 2:
        return [0, 2]
    elif i_ax == 3:
        return [1, 1]
    elif i_ax == 4:
        return [1, 2]
    elif i_ax == 5:
        return [2, 2]
    else:
        raise ValueError('Wrong stress index.')


def test_one_case(show):
    ### ----- Parameter definitions ----- ###
    # Geometry
    L_beam = [10, 0.5, 0.8]
    dim = len(L_beam)

    # Discretization
    Nx = 10
    Ny = 10
    Nz = 10
    nb_grid_pts = [Nx, Ny, Nz]
    gradient, weights = µ.linear_finite_elements.gradient_3d_5tet

    # Material
    Young = 120
    Poisson = 0.3

    # Loading
    curvature = 0.1
    delta_F = np.zeros((dim, dim))

    # Formulation
    formulation = µ.Formulation.small_strain

    # muSpectre solver parameters
    newton_tol       = 1e-7
    cg_tol           = 1e-7 # tolerance for cg algo
    equil_tol        = 1e-7 # tolerance for equilibrium
    maxiter          = 10000
    verbose          = µ.Verbosity.Silent
    #fft = 'mpi' # Parallel fft
    fft = 'fftw' # Serial fft


    ### ----- Calculations ----- ###
    # Calculate lengths of representative volume element (rve)
    hx = L_beam[0] / nb_grid_pts[0]
    hy = L_beam[1] / (nb_grid_pts[1] - 1)
    hz = L_beam[2] / (nb_grid_pts[2] - 1)
    L_rve = [L_beam[0], nb_grid_pts[1] * hy, nb_grid_pts[2] * hz]

    # Define geometry
    mask = np.ones(nb_grid_pts)
    mask[:, :, -1] = 0
    mask[:, -1, :] = 0

    # Initialize muSpectre cell
    cell = µ.Cell(nb_grid_pts, L_rve, formulation, gradient,
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

    # Initialize Eigen class
    eigen_class = EigenStrainBending3D(cell.pixels, curvature,
                                       L_rve, nb_grid_pts, Poisson)

    # Solving
    solver = µ.solvers.KrylovSolverCG(cell, cg_tol, maxiter, verbose)
    res = µ.solvers.newton_cg(cell, delta_F, solver, newton_tol, equil_tol,
                              verbose, μ.solvers.IsStrainInitialised.No,
                              µ.StoreNativeStress.No, eigen_class.eigen_strain_func)
    shape = (dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts)
    stress_num = res.stress.reshape(shape, order='F')
    strain_num = res.grad.reshape(shape, order='F')

    ### ----- Numerical moment around y-axis ----- ###
    # Detailed
    z = np.arange(nb_grid_pts[2]) * hz
    x = np.arange(nb_grid_pts[0]) * hx
    x_local = np.arange(10*nb_grid_pts[0])
    x_local = x_local % 10
    x_local = x_local * hx / 10
    x_local = x_local.reshape((-1, 10, 1))
    A = hy * (1 - x_local / hx)
    B = hy * x_local / hx
    C = hz * (1 - x_local / hx)
    D = hz * x_local / hx

    helper = np.sum(stress_num[0, 0], axis=(2))

    moment_num = np.zeros((5, nb_grid_pts[0], 10))
    z = z.reshape((1, 1, -1))
    helper2 = (z + 0.5 * hz) * (hz * hy - B * D - A * C)
    moment_num[0] = np.sum(helper[0, :, None, :] * helper2, axis=2)
    helper2 = 1/6 * A * C**2 + 0.5 * z * A * C
    moment_num[1] = np.sum(helper[1, :, None, :] * helper2, axis=2)
    helper2 = 1/6 * B * D**2 + 0.5 * z * B * D
    moment_num[2] = np.sum(helper[2, :, None, :] * helper2, axis=2)
    helper2 = 0.5 * (z + hz) * B * D - 1/6 * B * D**2
    moment_num[3] = np.sum(helper[3, :, None, :] * helper2, axis=2)
    helper2 = 0.5 * (z + hz) * A * C - 1/6 * A * C**2
    moment_num[4] = np.sum(helper[4, :, None, :] * helper2, axis=2)
    moment_num = np.sum(moment_num, axis=0)

    moment_num = moment_num.flatten()
    x_local = x_local + x[:, None, None]
    x_local = x_local.flatten()

    # Average over voxel
    helper = np.sum(stress_num[0, 0], axis=(2))
    z = z.reshape((1, -1))
    moment_num_av = 1/3 * (z + 0.5 * hz) * helper[0]
    moment_num_av += helper[1] * 1/6 * (z + 0.25 * hz)
    moment_num_av += helper[2] * 1/6 * (z + 0.25 * hz)
    moment_num_av += helper[3] * 1/6 * (z + 0.75 * hz)
    moment_num_av += helper[4] * 1/6 * (z + 0.75 * hz)
    moment_num_av = hy * hz * np.sum(moment_num_av, axis=1)

    # Average over x
    moment_av_1 = np.average(moment_num_av)
    helper = np.sum(stress_num[0, 0, :, 0, :, :], axis=1)
    z = z.flatten()
    moment_av = 1/3 * (z + 0.5 * hz) * helper[0]
    moment_av += helper[1] * 1/6 * (z + 0.25 * hz)
    moment_av += helper[2] * 1/6 * (z + 0.25 * hz)
    moment_av += helper[3] * 1/6 * (z + 0.75 * hz)
    moment_av += helper[4] * 1/6 * (z + 0.75 * hz)
    moment_av = hy * hz * np.sum(moment_av)

    print('Diff average moment =', moment_av_1 - moment_av)
    print('Diff average moment (%) =', (moment_av_1 - moment_av) / moment_av_1 * 100)

    # Analytical moment
    Iy = L_beam[1] * L_beam[2]**3 / 3
    stiff_ana = Young * Iy
    moment_ana = curvature * stiff_ana

    ### ----- Displacement ----- ###
    # Positions
    x = np.linspace(0, L_rve[0], nb_grid_pts[0]+1, endpoint=True)
    y = np.linspace(0, L_rve[1], nb_grid_pts[1]+1, endpoint=True)
    z = np.linspace(0, L_rve[2], nb_grid_pts[2]+1, endpoint=True)
    x = x.reshape((-1, 1, 1))
    y = y.reshape((1, -1, 1))
    z = z.reshape((1, 1, -1))

    # Displacements without imposed bending
    F0 = np.eye(3)
    strain_no_eigen = strain_num.copy()
    eigen_class.remove_eigen_strain_func(strain_no_eigen)
    print('Norm of strain without eigenstrain =', np.linalg.norm(strain_no_eigen))
    [x_0, y_0, z_0], [x_displ, y_displ, z_displ] \
        = get_complemented_positions("0d", cell, strain_array=strain_no_eigen, F0=F0,
                                     periodically_complemented=True)
    print('Norm of complete strain =', np.linalg.norm(strain_num))
    displ_fluct = np.asarray([x_displ.copy(), y_displ.copy(), z_displ.copy()])

    # Complete displacements
    x_displ += curvature * x * y
    y_displ -= Poisson * curvature * y * z
    z_displ += - curvature / 2 * x**2 + Poisson / 2 * curvature * y**2 - Poisson / 2 * curvature * z**2
    pos = np.asarray([x_0, y_0, z_0])
    displ = np.asarray([x_displ, y_displ, z_displ])

    ### ----- Postprocessing ----- ###
    x = np.arange(nb_grid_pts[0]) * hx
    y = np.arange(nb_grid_pts[1]) * hy
    z = np.arange(nb_grid_pts[2]) * hz

    # Compare with analytical stiffness / moment
    fig, ax = plt.subplots()
    fig.suptitle(f'Nx={Nx}  Ny={Ny}  Nz={Nz}')
    ax.set_xlabel('x')
    ax.set_ylabel(r'$M_y$')

    ax.plot(x_local, moment_num, label='num')
    ax.plot([x_local[0], x_local[-1]], [moment_ana, moment_ana], label='ana')
    moment_av = len(x) * [moment_av]
    ax.plot(x, moment_av, marker='x', linestyle='')
    ax.plot(x, moment_num_av, label='num (average)')

    ax.legend()

    plt.show()
    name = f'results/bending/moment_N={Nx}x{Ny}x{Nz}.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)

    # Plot deformed beam
    scale = 1
    fig, axes= plt.subplots(1, 2)
    fig.suptitle(f'Displacement at y={y_0[0, 0, 0]} for: Nx={Nx} Ny={Ny} Nz={Nz}')

    axes[0].set_title(f'Complete')
    axes[0].set_aspect('equal')
    axes[0].axis('off')
    x_def = x_0[:, 0, :] + scale * x_displ[:, 0, :]
    z_def = z_0[:, 0, :] + scale * z_displ[:, 0, :]
    axes[0].plot(x_def[0, :], z_def[0, :], color='black')
    axes[0].plot(x_def[-1, :], z_def[-1, :], color='black')
    axes[0].plot(x_def[:, 0], z_def[:, 0], color='black')
    axes[0].plot(x_def[:, -1], z_def[:, -1], color='black')

    axes[1].set_title(f'Fluctuations')
    axes[1].set_aspect('equal')
    axes[1].axis('off')
    x_def = x_0[:, 0, :] + scale * displ_fluct[0, :, 0, :]
    z_def = z_0[:, 0, :] + scale * displ_fluct[2, :, 0, :]
    axes[1].plot(x_def[0, :], z_def[0, :], color='black')
    axes[1].plot(x_def[-1, :], z_def[-1, :], color='black')
    axes[1].plot(x_def[:, 0], z_def[:, 0], color='black')
    axes[1].plot(x_def[:, -1], z_def[:, -1], color='black')

    plt.show()
    name = f'results/bending/deformed_beam_N={Nx}x{Ny}x{Nz}.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)

    # Plot Strain
    x = np.linspace(0, L_rve[0], nb_grid_pts[0], endpoint=False) + hx/2
    y = np.linspace(0, L_rve[1], nb_grid_pts[1], endpoint=False) + hy/2
    z = np.linspace(0, L_rve[2], nb_grid_pts[2], endpoint=False) + hz/2

    fig = plt.figure(figsize=(16, 5))
    gs_left = fig.add_gridspec(1, 1, left=0.05, right=0.3)
    gs_right = fig.add_gridspec(2, 6, left=0.33, right=0.95, hspace=0.3, wspace=0.3)
    strain = np.average(strain_num, axis=2, weights=(1/3, 1/6, 1/6, 1/6, 1/6))
    strain_labels = [r'$\varepsilon_{xx}$', r'$\varepsilon_{xy}$', r'$\varepsilon_{xz}$',
                     r'$\varepsilon_{yy}$', r'$\varepsilon_{yz}$', r'$\varepsilon_{zz}$']

    ax = fig.add_subplot(gs_left[0, 0])
    i_x = 0
    i_y = 0
    ax.set_xlabel('Position z')
    ax.set_ylabel(f'strain at x={x[i_x]:.2} y={y[i_y]:.2}')
    for i_strain in range(6):
        ind = stress_indices(i_strain)
        ax.plot(z, strain[ind[0], ind[1], i_x, i_y, :],
                label=strain_labels[i_strain], marker='x')
    ana_xx = curvature * z
    ana_yy = -Poisson * curvature * z
    ana_zz = -Poisson * curvature * z
    #ax.plot(z[0:-1], ana_xx[0:-1], label='ana', color='black', linestyle='--')
    #ax.plot(z[0:-1], ana_yy[0:-1], color='black', linestyle='--')
    #ax.plot(z[0:-1], ana_zz[0:-1], color='black', linestyle='--')
    ax.legend()

    i_z = [1, nb_grid_pts[2] // 2]
    for i_ax in range(2):
        for i_strain in range(6):
            ax = fig.add_subplot(gs_right[i_ax, i_strain])
            ind = stress_indices(i_strain)
            helper = strain[ind[0], ind[1], :, :, i_z[i_ax]]
            im = ax.pcolormesh(x, y, helper.T, rasterized=True)
            cbar = fig.colorbar(im, ax=ax)
            title = strain_labels[i_strain] + f' at z={z[i_z[i_ax]]:.2}'
            ax.set_title(title)
            ax.axis('off')

    plt.show()
    plt.close(fig)

    # Plot stress
    fig = plt.figure(figsize=(16, 5))
    gs_left = fig.add_gridspec(1, 1, left=0.05, right=0.3)
    gs_right = fig.add_gridspec(2, 6, left=0.33, right=0.95, hspace=0.3, wspace=0.3)
    stress = np.average(stress_num, axis=2, weights=(1/3, 1/6, 1/6, 1/6, 1/6))
    stress_labels = [r'$\sigma_{xx}$', r'$\sigma_{xy}$', r'$\sigma_{xz}$',
                     r'$\sigma_{yy}$', r'$\sigma_{yz}$', r'$\sigma_{zz}$']

    ax = fig.add_subplot(gs_left[0, 0])
    i_x = 0
    i_y = 0
    ax.set_xlabel('Position z')
    ax.set_ylabel(f'Stress at x={x[i_x]:.2} y={y[i_y]:.2}')
    for i_stress in range(6):
        ind = stress_indices(i_stress)
        ax.plot(z, stress[ind[0], ind[1], i_x, i_y, :],
                label=stress_labels[i_stress], marker='x')
    ax.plot(z, stress[0, 0, 1, 2, :],
                linestyle=':')
    ana = curvature * Young * z
    ax.plot(z[0:-1], ana[0:-1], label='ana', color='black', linestyle='--')
    ax.legend()

    i_z = [1, nb_grid_pts[2] // 2]
    for i_ax in range(2):
        for i_stress in range(6):
            ax = fig.add_subplot(gs_right[i_ax, i_stress])
            ind = stress_indices(i_stress)
            helper = stress[ind[0], ind[1], :, :, i_z[i_ax]]
            im = ax.pcolormesh(x, y, helper.T, rasterized=True)
            cbar = fig.colorbar(im, ax=ax)
            title = stress_labels[i_stress] + f' at z={z[i_z[i_ax]]:.2}'
            ax.set_title(title)
            ax.axis('off')

    # Show
    plt.show()
    name = f'results/bending/stresses_N={Nx}x{Ny}x{Nz}.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)

    print()

def test_convergences():
    ### ----- Parameter definitions ----- ###
    # Geometry
    L_beam = [10, 0.5, 0.8]
    dim = len(L_beam)

    # Discretization
    N_list = [5, 10, 20, 30, 40]
    # N_list = [3, 5]
    gradient, weights = µ.linear_finite_elements.gradient_3d_5tet

    # Material
    Young = 120
    Poisson = 0.3

    # Loading
    curvature = 0.1
    delta_F = np.zeros((dim, dim))

    # Formulation
    formulation = µ.Formulation.small_strain

    # muSpectre solver parameters
    newton_tol       = 1e-7
    cg_tol           = 1e-7 # tolerance for cg algo
    equil_tol        = 1e-7 # tolerance for equilibrium
    maxiter          = 10000
    verbose          = µ.Verbosity.Silent
    #fft = 'mpi' # Parallel fft
    fft = 'fftw' # Serial fft

    ### ----- Calculations ----- ###
    # Analytical moment
    Iy = L_beam[1] * L_beam[2]**3 / 3
    stiff_ana = Young * Iy
    moment_ana = curvature * stiff_ana

    # To calculate
    error_moment = np.empty(len(N_list))
    error_stress = np.empty((6, len(N_list)))

    for ind_N, N in enumerate(N_list):
        nb_grid_pts = [N, N, N]
        print(f'Calculation {ind_N+1} of {len(N_list)}')

        # Calculate lengths of representative volume element (rve)
        hx = L_beam[0] / nb_grid_pts[0]
        hy = L_beam[1] / (nb_grid_pts[1] - 1)
        hz = L_beam[2] / (nb_grid_pts[2] - 1)
        L_rve = [L_beam[0], nb_grid_pts[1] * hy, nb_grid_pts[2] * hz]

        # Define geometry
        mask = np.ones(nb_grid_pts)
        mask[:, :, -1] = 0
        mask[:, -1, :] = 0

        # Initialize muSpectre cell
        cell = µ.Cell(nb_grid_pts, L_rve, formulation, gradient,
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

        # Initialize Eigen class
        eigen_class = EigenStrainBending3D(cell.pixels, curvature,
                                           L_rve, nb_grid_pts, Poisson)

        # Solving
        solver = µ.solvers.KrylovSolverCG(cell, cg_tol, maxiter, verbose)
        res = µ.solvers.newton_cg(cell, delta_F, solver, newton_tol, equil_tol,
                                  verbose, μ.solvers.IsStrainInitialised.No,
                                  µ.StoreNativeStress.No, eigen_class.eigen_strain_func)
        shape = (dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts)
        stress_num = res.stress.reshape(shape, order='F')
        strain_num = res.grad.reshape(shape, order='F')

        # Calculate error of moment
        helper = np.sum(stress_num[0, 0, :, 0, :, :], axis=1)
        z = np.arange(nb_grid_pts[2]) * hz
        moment_num = 1/3 * (z + 0.5 * hz) * helper[0] # Assumption: moment independent of x
        moment_num += helper[1] * 1/6 * (z + 0.25 * hz)
        moment_num += helper[2] * 1/6 * (z + 0.25 * hz)
        moment_num += helper[3] * 1/6 * (z + 0.75 * hz)
        moment_num += helper[4] * 1/6 * (z + 0.75 * hz)
        moment_num = hy * hz * np.sum(moment_num)

        error = abs(moment_num - moment_ana) / moment_ana * 100
        error_moment[ind_N] = error

        # Calculate error of stress
        z = np.linspace(0, L_rve[2], nb_grid_pts[2], endpoint=False) + hz/2
        stress = np.average(stress_num, axis=2, weights=(1/3, 1/6, 1/6, 1/6, 1/6))
        stress = np.average(stress[:, :, :, 0:-1, :], axis=(2, 3))
        stress_ana = curvature * z * Young
        stress_ana[-1] = 0
        error = np.linalg.norm(stress[0, 0] - stress_ana) / np.linalg.norm(stress_ana) * 100
        error_stress[0, ind_N] = error
        for i_stress in range(1, 6):
            ind = stress_indices(i_stress)
            error_stress[i_stress, ind_N] = np.linalg.norm(stress[ind[0], ind[1]])

    ### ----- Plotting ----- ###
    # Moment
    fig, ax = plt.subplots()
    ax.set_xlabel('Nx=Ny=Nz')
    ax.set_ylabel('Error of moment (%)')
    ax.plot(N_list, error_moment)

    plt.show()
    name = 'results/bending/moment.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)

    # Stress
    fig, axes = plt.subplots(1, 2, figsize=(9, 5), gridspec_kw={'wspace':0.35})
    axes[0].set_xlabel('Nx=Ny=Nz')
    axes[0].set_ylabel('Error of stress (%)')
    axes[0].plot(N_list, error_stress[0], label=r'$\sigma_{xx}$')
    axes[0].legend()
    axes[1].set_xlabel('Nx=Ny=Nz')
    axes[1].set_ylabel('Error of stress (-)')
    axes[1].plot(N_list, error_stress[1], label=r'$\sigma_{yy}$')
    axes[1].plot(N_list, error_stress[2], label=r'$\sigma_{zz}$')
    axes[1].plot(N_list, error_stress[3], label=r'$\sigma_{xy}$')
    axes[1].plot(N_list, error_stress[4], label=r'$\sigma_{xz}$')
    axes[1].plot(N_list, error_stress[5], label=r'$\sigma_{yz}$')
    axes[1].legend()

    plt.show()
    name = 'results/bending/stresses.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)
    print()

def test_non_periodic_y():
    ### ----- Parameter definitions ----- ###
    # Geometry
    L_beam = [10, 0.5, 0.8]
    dim = len(L_beam)

    # Discretization
    N_list = [5, 10, 20, 30, 40]
    #N_list = [3, 5]
    gradient, weights = µ.linear_finite_elements.gradient_3d_5tet

    # Material
    Young = 120
    Poisson = 0.3

    # Loading
    curvature = 0.1
    delta_F = np.zeros((dim, dim))

    # Formulation
    formulation = µ.Formulation.small_strain

    # muSpectre solver parameters
    newton_tol       = 1e-7
    cg_tol           = 1e-7 # tolerance for cg algo
    equil_tol        = 1e-7 # tolerance for equilibrium
    maxiter          = 10000
    verbose          = µ.Verbosity.Silent
    #fft = 'mpi' # Parallel fft
    fft = 'fftw' # Serial fft

    ### ----- Calculations ----- ###
    # Analytical moment
    Iy = L_beam[1] * L_beam[2]**3 / 3
    stiff_ana = Young * Iy
    moment_ana = curvature * stiff_ana

    # To calculate
    error_moment = np.empty(len(N_list))
    error_stress = np.empty((6, len(N_list)))

    for ind_N, N in enumerate(N_list):
        nb_grid_pts = [N, N, N]
        print(f'Calculation {ind_N+1} of {len(N_list)}')

        # Calculate lengths of representative volume element (rve)
        hx = L_beam[0] / nb_grid_pts[0]
        #hy = L_beam[1] / (nb_grid_pts[1] - 1)
        hy = L_beam[1] / nb_grid_pts[1]
        hz = L_beam[2] / (nb_grid_pts[2] - 1)
        #L_rve = [L_beam[0], nb_grid_pts[1] * hy, nb_grid_pts[2] * hz]
        L_rve = [L_beam[0], L_beam[1], nb_grid_pts[2] * hz]

        # Define geometry
        mask = np.ones(nb_grid_pts)
        mask[:, :, -1] = 0
        #mask[:, -1, :] = 0

        # Initialize muSpectre cell
        cell = µ.Cell(nb_grid_pts, L_rve, formulation, gradient,
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

        # Initialize Eigen class
        eigen_class = EigenStrainBending3D(cell.pixels, curvature,
                                           L_rve, nb_grid_pts, Poisson)

        # Solving
        solver = µ.solvers.KrylovSolverCG(cell, cg_tol, maxiter, verbose)
        res = µ.solvers.newton_cg(cell, delta_F, solver, newton_tol, equil_tol,
                                  verbose, μ.solvers.IsStrainInitialised.No,
                                  µ.StoreNativeStress.No, eigen_class.eigen_strain_func)
        shape = (dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts)
        stress_num = res.stress.reshape(shape, order='F')
        strain_num = res.grad.reshape(shape, order='F')

        # Calculate error of moment
        helper = np.sum(stress_num[0, 0, :, 0, :, :], axis=1)
        z = np.arange(nb_grid_pts[2]) * hz
        moment_num = 1/3 * (z + 0.5 * hz) * helper[0] # Assumption: moment independent of x
        moment_num += helper[1] * 1/6 * (z + 0.25 * hz)
        moment_num += helper[2] * 1/6 * (z + 0.25 * hz)
        moment_num += helper[3] * 1/6 * (z + 0.75 * hz)
        moment_num += helper[4] * 1/6 * (z + 0.75 * hz)
        moment_num = hy * hz * np.sum(moment_num)

        error = abs(moment_num - moment_ana) / moment_ana * 100
        error_moment[ind_N] = error

        # Calculate error of stress
        z = np.linspace(0, L_rve[2], nb_grid_pts[2], endpoint=False) + hz/2
        stress = np.average(stress_num, axis=2, weights=(1/3, 1/6, 1/6, 1/6, 1/6))
        stress = np.average(stress[:, :, :, 0:-1], axis=(2, 3))
        stress_ana = curvature * z * Young
        stress_ana[-1] = 0
        error = np.linalg.norm(stress[0, 0] - stress_ana) / np.linalg.norm(stress_ana) * 100
        error_stress[0, ind_N] = error
        for i_stress in range(1, 6):
            ind = stress_indices(i_stress)
            error_stress[i_stress, ind_N] = np.linalg.norm(stress[ind[0], ind[1]])

    ### ----- Plotting ----- ###
    # Moment
    fig, ax = plt.subplots()
    ax.set_xlabel('Nx=Ny=Nz')
    ax.set_ylabel('Error of moment (%)')
    ax.plot(N_list, error_moment)

    # Stress
    fig, axes = plt.subplots(1, 2, figsize=(9, 5), gridspec_kw={'wspace':0.35})
    axes[0].set_xlabel('Nx=Ny=Nz')
    axes[0].set_ylabel('Error of stress (%)')
    axes[0].plot(N_list, error_stress[0], label=r'$\sigma_{xx}$')
    axes[0].legend()
    axes[1].set_xlabel('Nx=Ny=Nz')
    axes[1].set_ylabel('Error of stress (-)')
    axes[1].plot(N_list, error_stress[1], label=r'$\sigma_{yy}$')
    axes[1].plot(N_list, error_stress[2], label=r'$\sigma_{zz}$')
    axes[1].plot(N_list, error_stress[3], label=r'$\sigma_{xy}$')
    axes[1].plot(N_list, error_stress[4], label=r'$\sigma_{xz}$')
    axes[1].plot(N_list, error_stress[5], label=r'$\sigma_{yz}$')
    axes[1].legend()

    # Show
    plt.show()
    print()

def test_problem():
    """ Test implementation of bending EigenStrain by solving a simple test problem:
        Bending of a beam with rectangular cross-section
    """
    test_one_case(show=True)
    #test_convergences()
    #test_non_periodic_y()

def calculation_paper():
    """ Save data for publishing.
    """
    ### ----- Parameter definitions ----- ###
    # Geometry
    L_beam = [10, 0.5, 0.8]
    dim = len(L_beam)

    # Discretization
    N_list = [5, 10, 20, 30, 40]
    # N_list = [3, 5]
    #N_list = [30, 50, 70]
    gradient, weights = µ.linear_finite_elements.gradient_3d_5tet

    # Material
    Young = 120
    Poisson = 0.3

    # Loading
    moment = 4
    curvature = moment / Young / Poisson
    delta_F = np.zeros((dim, dim))

    # Formulation
    formulation = µ.Formulation.small_strain

    # muSpectre solver parameters
    newton_tol       = 1e-7
    cg_tol           = 1e-7 # tolerance for cg algo
    equil_tol        = 1e-7 # tolerance for equilibrium
    maxiter          = 10000
    verbose          = µ.Verbosity.Silent
    #fft = 'mpi' # Parallel fft
    fft = 'fftw' # Serial fft

    # Parameters for plotting
    folder = '/mnt/c/Users/indre/Documents/Ubuntu/optimization/chirality/'
    folder += 'muChirality/examples/results/bending/'
    name = folder + f'data_L_beam={L_beam[0]}x{L_beam[1]}x{L_beam[2]}_Young={Young}_Poisson={Poisson}_moment={moment}.txt'

    font_large = 12

    ### ----- Calculations ----- ###
    # Analytical moment
    Iy = L_beam[1] * L_beam[2]**3 / 3
    stiff_ana = Young * Iy

    # To calculate
    error_stiff = np.empty(len(N_list))

    for ind_N, N in enumerate(N_list):
        nb_grid_pts = [N, N, N]
        print(f'Calculation {ind_N+1} of {len(N_list)}')

        # Calculate lengths of representative volume element (rve)
        hx = L_beam[0] / nb_grid_pts[0]
        #hy = L_beam[1] / (nb_grid_pts[1] - 1)
        hy = L_beam[1] / nb_grid_pts[1]
        hz = L_beam[2] / (nb_grid_pts[2] - 1)
        #L_rve = [L_beam[0], nb_grid_pts[1] * hy, nb_grid_pts[2] * hz]
        L_rve = [L_beam[0], L_beam[1], nb_grid_pts[2] * hz]

        # Define geometry
        mask = np.ones(nb_grid_pts)
        mask[:, :, -1] = 0
        #mask[:, -1, :] = 0

        # Initialize muSpectre cell
        cell = µ.Cell(nb_grid_pts, L_rve, formulation, gradient,
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

        # Initialize Eigen class
        eigen_class = EigenStrainBending3D(cell.pixels, curvature,
                                           L_rve, nb_grid_pts, Poisson)

        # Solving
        solver = µ.solvers.KrylovSolverCG(cell, cg_tol, maxiter, verbose)
        res = µ.solvers.newton_cg(cell, delta_F, solver, newton_tol, equil_tol,
                                  verbose, μ.solvers.IsStrainInitialised.No,
                                  µ.StoreNativeStress.No, eigen_class.eigen_strain_func)
        shape = (dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts)
        stress_num = res.stress.reshape(shape, order='F')
        strain_num = res.grad.reshape(shape, order='F')

        # Calculate error of stiffness
        helper = np.sum(stress_num[0, 0, :, 0, :, :], axis=1)
        z = np.arange(nb_grid_pts[2]) * hz
        moment_num = 1/3 * (z + 0.5 * hz) * helper[0] # Assumption: moment independent of x
        moment_num += helper[1] * 1/6 * (z + 0.25 * hz)
        moment_num += helper[2] * 1/6 * (z + 0.25 * hz)
        moment_num += helper[3] * 1/6 * (z + 0.75 * hz)
        moment_num += helper[4] * 1/6 * (z + 0.75 * hz)
        moment_num = hy * hz * np.sum(moment_num)

        stiff_num = moment_num / curvature

        error = abs(stiff_num - stiff_ana) / stiff_ana * 100
        error_stiff[ind_N] = error

    ### ----- Save data ----- ###
    with open(name, 'w') as f:
        title = 'nb_grid_pts (in each direction)'
        print(title, file=f)
        np.savetxt(f, N_list, newline=' ')
        title = '\nrel_error_of_stiffness(%)'
        print(title, file=f)
        np.savetxt(f, error_stiff, newline=' ')

    ### ----- Plotting ----- ###
    # Prepare figure
    fig, ax = plt.subplots()
    ax.set_xlabel('Number of voxels in every direction (-)', fontsize=font_large)
    ax.set_ylabel('Differences between stiffnesses (%)', fontsize=font_large)

    # Plot
    ax.plot(N_list, error_stiff, marker='x')

    # Show
    plt.show()
    name = folder + 'stiffnesses.pdf'
    fig.savefig(name, bbox_inches='tight')

if __name__ == "__main__":
    # plot_functions()
    # test_problem()
    calculation_paper()

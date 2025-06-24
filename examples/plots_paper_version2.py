"""
@file   plots_paper.py

@author Indre  Joedicke <indre.joedicke@imtek.uni-freiburg.de>

@date   28 Fev 2025

@brief  Make plots for paper to chiral metamaterials

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
import matplotlib as mpl

import muSpectre as µ
import netCDF4

from muChirality.EigenStrainTorsion import EigenStrain
from muSpectre.gradient_integration import get_complemented_positions
import muChirality.Geometries as geo

# Change automatic font in matplotlib
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['font.serif'] = ['Arial']
mpl.rcParams['font.cursive'] = ['Arial']
mpl.rcParams['font.size'] = '8' #'10'
mpl.rcParams['legend.fontsize'] = '8' #'10'
mpl.rcParams['xtick.labelsize'] = '9'
mpl.rcParams['ytick.labelsize'] = '9'
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['axes.linewidth'] = 0.5

mpl.rc('text', usetex=False)
mpl.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"

##################################################################
### ----------- Plot models: Torsion and bending ------------- ###
##################################################################
def plot_models():
    print('Plot definition small strain rotation and bending.')
    ### ----- Parameter definition ----- ###
    # Geometry
    Lz_torsion = 0.5
    Lxy_torsion = 0.3
    Lx_bend = 8
    Lz_bend = 0.8

    # Loading
    twist = 0.09
    curvature_max = 0.15
    Poisson = 0.3

    # Discretization
    Nx = 10
    Nz = 5

    # Coordinate system
    Lc = 0.12
    angle_c = np.pi / 10

    L_coord = 0.75
    head_width_coord = 0.2
    head_length_coord = 1.5 * head_width_coord
    width_coord = 0.001
    x_coord = 0
    z_coord = 0

    # Colors
    color_def = 'black'
    color_undef = 'grey'
    color_twist = 'blue'
    color_c = 'black'
    color_coord = 'blue'
    color_rot_axis = 'red'
    color_bend = 'red'

    # Lines
    width_d = 1.5
    width_u = 1.5
    style_u = '-'
    style_u_inv = ''
    style_inv = '--'
    style_rot_axis = ':'
    style_d = '-'

    # Arrows
    head_width = 0.02
    head_length = 1.5 * head_width
    width_t = 0.001
    head_width_c = 0.01
    head_length_c = 1.5 * head_width_c
    width_c = 0.001

    # Fontsize
    fontsize = 12

    ### ----- Coordinates positions: Rotation ----- ###
    # Coordinates of bottom points
    B1_x = 1.8 * Lc
    B1_y = - 0.5 * Lc
    B2_x = B1_x - Lxy_torsion * np.sin(angle_c)
    B2_y = B1_y + Lxy_torsion * np.cos(angle_c)
    B3_x = B2_x + Lxy_torsion * np.cos(angle_c)
    B3_y = B2_y + Lxy_torsion * np.sin(angle_c)
    B4_x = B1_x + Lxy_torsion * np.cos(angle_c)
    B4_y = B1_y + Lxy_torsion * np.sin(angle_c)

    # Coordinates of top points (undeformed geometry)
    U1_x = B1_x
    U1_y = B1_y + Lz_torsion
    U2_x = B2_x
    U2_y = B2_y + Lz_torsion
    U3_x = B3_x
    U3_y = B3_y + Lz_torsion
    U4_x = B4_x
    U4_y = B4_y + Lz_torsion

    # Coordinates of top points (deformed geometry)
    # Note: magnitude of rotational displacement in both directions is
    # twist * sin(pi/4) = 1 / sqrt(2) * twist
    # or twist * cos(pi/4) = 1 / sqrt(2) * twist
    disp = twist / np.sqrt(2)
    D1_x = U1_x + disp * np.cos(angle_c) + disp * np.sin(angle_c)
    D1_y = U1_y + disp * np.sin(angle_c) - disp * np.cos(angle_c)
    D2_x = U2_x - disp * np.cos(angle_c) + disp * np.sin(angle_c)
    D2_y = U2_y - disp * np.sin(angle_c) - disp * np.cos(angle_c)
    D3_x = U3_x - disp * np.cos(angle_c) - disp * np.sin(angle_c)
    D3_y = U3_y - disp * np.sin(angle_c) + disp * np.cos(angle_c)
    D4_x = U4_x + disp * np.cos(angle_c) - disp * np.sin(angle_c)
    D4_y = U4_y + disp * np.sin(angle_c) + disp * np.cos(angle_c)

    # Coordinates of rotation axis
    A_x = B1_x + Lxy_torsion / 2 * np.cos(angle_c) - Lxy_torsion / 2 * np.sin(angle_c)
    A2_y = B1_y + (A_x - B1_x) * np.tan(angle_c)
    A1_y = A2_y - 1.2 * Lc
    A4_y = U1_y + Lxy_torsion / 2 * np.sin(angle_c) + Lxy_torsion / 2 * np.cos(angle_c)
    A5_y = D3_y + 0.2 * Lxy_torsion
    A3_y = B1_y + Lxy_torsion / 2 * np.cos(angle_c) + Lxy_torsion / 2 * np.sin(angle_c)

    ### ----- Calculate positions: Bending ----- ###
    # Undeformed beam
    x = np.linspace(0, Lx_bend, Nx)
    z = np.linspace(0, Lz_bend, Nz)
    X_bend, Z_bend = np.meshgrid(x, z)

    # Displacements (for y=0)
    disp_x = curvature_max * X_bend * Z_bend
    disp_z = - curvature_max / 2 * X_bend**2 - Poisson * curvature_max / 2 * Z_bend**2

    # Deformed beam
    pos_x_bend = X_bend + disp_x
    pos_z_bend = Z_bend + disp_z

    ### ---- Prepare figure ----- ###
    fig = plt.figure(figsize=(7, 4))
    grid_left = fig.add_gridspec(1, 1, right=0.5)
    grid_right = fig.add_gridspec(1, 1, left=0.5)

    ax_torsion = fig.add_subplot(grid_left[0, 0])
    ax_torsion.set_aspect('equal')
    ax_torsion.axis('off')
    ax_bend = fig.add_subplot(grid_right[0, 0])
    ax_bend.set_aspect('equal')
    ax_bend.axis('off')

    ### ----- Plotting small strain rotation ----- ###
    # Plot coordinate system
    ax_torsion.arrow(0, 0, Lc * np.cos(angle_c), Lc * np.sin(angle_c),
             color=color_c, width=width_c,
             length_includes_head=True, head_width = head_width_c,
             head_length=head_length_c)
    ax_torsion.arrow(0, 0, 0, Lc, color=color_c, width=width_c,
             length_includes_head=True, head_width = head_width_c,
             head_length=head_length_c)
    ax_torsion.arrow(0, 0, -Lc * np.sin(angle_c), Lc * np.cos(angle_c),
             color=color_c, width=width_c,
             length_includes_head=True, head_width = head_width_c,
             head_length=head_length_c)
    ax_torsion.text(Lc * 0.8, - Lc * 0.25, 'x', color=color_c, fontsize=fontsize)
    ax_torsion.text(-Lc * 0.75, Lc * 0.7, 'y', color=color_c, fontsize=fontsize)
    ax_torsion.text(0, Lc * 1.05, 'z', color=color_c, fontsize=fontsize)

    # Plot undeformed beam
    ax_torsion.plot([B1_x, B2_x], [B1_y, B2_y], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax_torsion.plot([B1_x, B4_x], [B1_y, B4_y], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax_torsion.plot([U1_x, U2_x], [U1_y, U2_y], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax_torsion.plot([U2_x, U3_x], [U2_y, U3_y], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax_torsion.plot([U3_x, U4_x], [U3_y, U4_y], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax_torsion.plot([U4_x, U1_x], [U4_y, U1_y], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax_torsion.plot([B1_x, U1_x], [B1_y, U1_y], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax_torsion.plot([B2_x, U2_x], [B2_y, U2_y], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax_torsion.plot([B4_x, U4_x], [B4_y, U4_y], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax_torsion.plot([B2_x, B3_x], [B2_y, B3_y], color=color_undef, linewidth=width_u, linestyle=style_u_inv)
    ax_torsion.plot([B3_x, B4_x], [B3_y, B4_y], color=color_undef, linewidth=width_u, linestyle=style_u_inv)
    ax_torsion.plot([B3_x, U3_x], [B3_y, U3_y], color=color_undef, linewidth=width_u, linestyle=style_u_inv)

    # Plot rotation axis
    ax_torsion.plot([A_x, A_x], [A1_y, A2_y], color=color_rot_axis)
    ax_torsion.plot([A_x, A_x], [A2_y, A4_y], color=color_rot_axis, linestyle=style_inv)
    ax_torsion.plot([A_x, A_x], [A4_y, A5_y], color=color_rot_axis)

    ax_torsion.arrow(A_x, A1_y, 0, Lc, color=color_rot_axis, length_includes_head=True,
             head_width=head_width, head_length=head_length, width=width_t)
    ax_torsion.text(A_x + 0.2 * Lc, A1_y + 0.25 * Lc, '$\hat{n}$', color=color_rot_axis,
            fontsize=fontsize)

    # Plot reference point
    ax_torsion.plot([A_x, A_x], [A3_y, A3_y], color=color_rot_axis, marker='o', markersize=5)
    ax_torsion.text(A_x + 0.2 * Lc, A3_y - 0.3 * Lc, r'$\vec X_0$', color=color_rot_axis, fontsize=fontsize)

    # Plot deformed beam
    ax_torsion.plot([D1_x, D2_x], [D1_y, D2_y], color=color_def, linewidth=width_d)
    ax_torsion.plot([D1_x, D4_x], [D1_y, D4_y], color=color_def, linewidth=width_d)
    ax_torsion.plot([D2_x, D3_x], [D2_y, D3_y], color=color_def, linewidth=width_d)
    ax_torsion.plot([D3_x, D4_x], [D3_y, D4_y], color=color_def, linewidth=width_d)

    ax_torsion.plot([B1_x, D1_x], [B1_y, D1_y], color=color_def, linewidth=width_d)
    ax_torsion.plot([B2_x, D2_x], [B2_y, D2_y], color=color_def, linewidth=width_d)
    ax_torsion.plot([B3_x, D3_x], [B3_y, D3_y], color=color_def, linewidth=width_d, linestyle=style_inv)
    ax_torsion.plot([B4_x, D4_x], [B4_y, D4_y], color=color_def, linewidth=width_d)

    ax_torsion.plot([B1_x, B2_x], [B1_y, B2_y], color=color_def, linewidth=width_d)
    ax_torsion.plot([B1_x, B4_x], [B1_y, B4_y], color=color_def, linewidth=width_d)
    ax_torsion.plot([B2_x, B3_x], [B2_y, B3_y], color=color_def, linewidth=width_d, linestyle=style_inv)
    ax_torsion.plot([B4_x, B3_x], [B4_y, B3_y], color=color_def, linewidth=width_d, linestyle=style_inv)

    # Plot twist
    ax_torsion.arrow(U1_x, U1_y, D1_x - U1_x, D1_y - U1_y, color=color_twist,
             length_includes_head=True, head_width=head_width,
             head_length=head_length, width=width_t)
    ax_torsion.arrow(U2_x, U2_y, D2_x - U2_x, D2_y - U2_y, color=color_twist,
             length_includes_head=True, head_width=head_width,
             head_length=head_length, width=width_t)
    ax_torsion.arrow(U3_x, U3_y, D3_x - U3_x, D3_y - U3_y, color=color_twist,
             length_includes_head=True, head_width=head_width,
             head_length=head_length, width=width_t)
    ax_torsion.arrow(U4_x, U4_y, D4_x - U4_x, D4_y - U4_y, color=color_twist,
             length_includes_head=True, head_width=head_width,
             head_length=head_length, width=width_t)
    #ax_torsion.text(U2_x + (D2_x - U2_x) / 2, U2_y + (D2_y - U2_y) / 4, '$u_{\mathsf{max}}$',
    #        color=color_twist, fontstyle='italic', ha='right', fontsize=fontsize)

    ax_torsion.text(-Lc * np.sin(angle_c), 1.02 * A5_y, '(a) Torsion',
                    fontsize=fontsize, ha='left', weight='bold')

    ### ----- Plot Small strain bending ----- ###
    # Plot coordinate system
    ax_bend.arrow(x_coord, z_coord, L_coord, 0, color=color_coord, width=width_coord,
             length_includes_head=True, head_width=head_width_coord,
             head_length=head_length_coord)
    ax_bend.arrow(x_coord, z_coord, 0, L_coord, color=color_coord, width=width_coord, length_includes_head=True,
             head_width=head_width_coord, head_length=head_length_coord)
    ax_bend.text(x_coord + 0.65 * L_coord, z_coord - 0.9 * L_coord, 'x', color=color_coord, fontsize=fontsize)
    ax_bend.text(x_coord - 0.8 * L_coord, z_coord + 0.75 * L_coord, 'z', color=color_coord, fontsize=fontsize)

    # Plot undeformed beam
    ax_bend.plot(X_bend[:, 0], Z_bend[:, 0], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax_bend.plot(X_bend[:, -1], Z_bend[:, -1], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax_bend.plot(X_bend[0, :], Z_bend[0, :], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax_bend.plot(X_bend[-1, :], Z_bend[-1, :], color=color_undef, linewidth=width_u, linestyle=style_u)

    # Plot deformed beam
    ax_bend.plot(pos_x_bend[:, 0], pos_z_bend[:, 0], color=color_def, linewidth=width_d, linestyle=style_d)
    ax_bend.plot(pos_x_bend[:, -1], pos_z_bend[:, -1], color=color_def, linewidth=width_d, linestyle=style_d)
    ax_bend.plot(pos_x_bend[0, :], pos_z_bend[0, :], color=color_def, linewidth=width_d, linestyle=style_d)
    ax_bend.plot(pos_x_bend[-1, :], pos_z_bend[-1, :], color=color_def, linewidth=width_d, linestyle=style_d)

    # Plot loading
    curv = curvature_max
    radius = 1 / curv
    Kx = 0
    Kz = pos_z_bend[0, 0] - 1 / curv

    artist = mpl.patches.Wedge((Kx, Kz), radius, 90-5, 90+5, fc=color_bend, ec='None', width=radius-6.6)
    ax_bend.add_artist(artist)
    ax_bend.plot([0, Kx], [0, Kz], color=color_bend, linewidth=1)
    ax_bend.plot([Kx], [Kz], color=color_bend, marker='x')
    ax_bend.text(Kx+0.15, Kz/2, r'$R$', color=color_bend, fontsize=fontsize)

    ax_bend.text(-0.5, 2.9 * Lz_bend, '(b) Bending', fontsize=fontsize, ha='left', weight='bold')

    ### ----- Save figure ----- ###
    name = 'results_paper/loadings.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.show()
    plt.close(fig)


##################################################################
### ------------------- Plot discretization ------------------ ###
##################################################################
def plot_discretization_3D_5_tetraedras():
    print('Plot discretization.')
    ### ----- Parameter definition ----- ###
    # Geometry
    Lxz = 1
    Ly = 0.7

    # Coordinate system
    Lc = 0.6
    angle_c = np.pi / 4

    # Distance between cubes
    d = 0.25

    # Colors
    color_cube = 'black'
    color_tet = 'red'
    color_cs = 'black'

    # Lines
    width_cube = 1.5
    width_tet = 1.5
    style = '-'
    style_inv = '--'

    # Arrows
    head_width = 0.1
    head_length = 1.5 * head_width
    width_c = 0.001

    ### ----- Coordinates calculations ----- ###
    sin = np.sin(angle_c)
    cos = np.cos(angle_c)

    # Coordinates of cubes: Cube - Bottom or top - Corner - x or y component
    co = np.empty((5, 2, 4, 2))

    # First cube
    co[0, 0, 0, :] = 0 # Origin of coordinate system
    co[0, 0, 1, 0] = Ly * cos
    co[0, 0, 1, 1] = Ly * sin
    co[0, 0, 2, 0] = Ly * cos + Lxz
    co[0, 0, 2, 1] = Ly * sin
    co[0, 0, 3, 0] = Lxz
    co[0, 0, 3, 1] = 0
    co[0, 1, 0, 0] = 0
    co[0, 1, 0, 1] = Lxz
    co[0, 1, 1, 0] = Ly * cos
    co[0, 1, 1, 1] = Lxz + Ly * sin
    co[0, 1, 2, 0] = Lxz + Ly * cos
    co[0, 1, 2, 1] = Lxz + Ly * sin
    co[0, 1, 3, 0] = Lxz
    co[0, 1, 3, 1] = Lxz

    # Other cubes
    co[:, 0, 0, 1] = co[0, 0, 0, 1]
    for i in range(4):
        co[i+1, 0, 0, 0] = co[i, 0, 0, 0] + co[0, 0, 2, 0] + d
    for i in range(1, 5):
        for j in range(4):
            co[i, 0, j, :] = co[i, 0, 0, :] + co[0, 0, j, :]
            co[i, 1, j, :] = co[i, 0, 0, :] + co[0, 1, j, :]

    # Coordinate system
    coord_x = - Lc / 2
    coord_y = - Lc * 1.1

    ### ----- Plotting discretization ----- ###
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.axis('off')

    # Plot coordinate system
    ax.arrow(coord_x, coord_y, Lc, 0,
             color=color_cs, width=width_c,
             length_includes_head=True, head_width = head_width,
             head_length=head_length)
    ax.arrow(coord_x, coord_y, Lc * cos, Lc * sin, color=color_cs, width=width_c,
             length_includes_head=True, head_width = head_width,
             head_length=head_length)
    ax.arrow(coord_x, coord_y, 0, Lc,
             color=color_cs, width=width_c,
             length_includes_head=True, head_width = head_width,
             head_length=head_length)
    ax.text(coord_x + Lc * 1.01, coord_y + Lc * 0.1, 'x', color=color_cs)
    ax.text(coord_x + Lc * 0.9, coord_y + Lc * sin * 0.9, 'y', color=color_cs)
    ax.text(coord_x, coord_y + Lc * 1.05, 'z', color=color_cs)

    # Plot cubes
    for i in range(5):
        ax.plot([co[i, 0, 0, 0], co[i, 0, 1, 0]], [co[i, 0, 0, 1], co[i, 0, 1, 1]],
                linestyle=style_inv, color=color_cube, linewidth=width_cube)
        ax.plot([co[i, 0, 1, 0], co[i, 0, 2, 0]], [co[i, 0, 1, 1], co[i, 0, 2, 1]],
                linestyle=style_inv, color=color_cube, linewidth=width_cube)
        ax.plot([co[i, 0, 2, 0], co[i, 0, 3, 0]], [co[i, 0, 2, 1], co[i, 0, 3, 1]],
                linestyle=style, color=color_cube, linewidth=width_cube)
        ax.plot([co[i, 0, 0, 0], co[i, 0, 3, 0]], [co[i, 0, 0, 1], co[i, 0, 3, 1]],
                linestyle=style, color=color_cube, linewidth=width_cube)

        ax.plot([co[i, 1, 0, 0], co[i, 1, 1, 0]], [co[i, 1, 0, 1], co[i, 1, 1, 1]],
                linestyle=style, color=color_cube, linewidth=width_cube)
        ax.plot([co[i, 1, 1, 0], co[i, 1, 2, 0]], [co[i, 1, 1, 1], co[i, 1, 2, 1]],
                linestyle=style, color=color_cube, linewidth=width_cube)
        ax.plot([co[i, 1, 2, 0], co[i, 1, 3, 0]], [co[i, 1, 2, 1], co[i, 1, 3, 1]],
                linestyle=style, color=color_cube, linewidth=width_cube)
        ax.plot([co[i, 1, 0, 0], co[i, 1, 3, 0]], [co[i, 1, 0, 1], co[i, 1, 3, 1]],
                linestyle=style, color=color_cube, linewidth=width_cube)

        ax.plot([co[i, 0, 0, 0], co[i, 1, 0, 0]], [co[i, 0, 0, 1], co[i, 1, 0, 1]],
                linestyle=style, color=color_cube, linewidth=width_cube)
        ax.plot([co[i, 0, 1, 0], co[i, 1, 1, 0]], [co[i, 0, 1, 1], co[i, 1, 1, 1]],
                linestyle=style_inv, color=color_cube, linewidth=width_cube)
        ax.plot([co[i, 0, 2, 0], co[i, 1, 2, 0]], [co[i, 0, 2, 1], co[i, 1, 2, 1]],
                linestyle=style, color=color_cube, linewidth=width_cube)
        ax.plot([co[i, 0, 3, 0], co[i, 1, 3, 0]], [co[i, 0, 3, 1], co[i, 1, 3, 1]],
                linestyle=style, color=color_cube, linewidth=width_cube)


    # Plot first tetrahedra
    ax.plot([co[0, 0, 3, 0], co[0, 1, 0, 0]], [co[0, 0, 3, 1], co[0, 1, 0, 1]],
            linestyle=style, color=color_tet, linewidth=width_tet)
    ax.plot([co[0, 0, 3, 0], co[0, 1, 2, 0]], [co[0, 0, 3, 1], co[0, 1, 2, 1]],
            linestyle=style, color=color_tet, linewidth=width_tet)
    ax.plot([co[0, 1, 2, 0], co[0, 1, 0, 0]], [co[0, 1, 2, 1], co[0, 1, 0, 1]],
            linestyle=style, color=color_tet, linewidth=width_tet)
    ax.plot([co[0, 0, 3, 0], co[0, 0, 1, 0]], [co[0, 0, 3, 1], co[0, 0, 1, 1]],
            linestyle=style_inv, color=color_tet, linewidth=width_tet)
    ax.plot([co[0, 1, 0, 0], co[0, 0, 1, 0]], [co[0, 1, 0, 1], co[0, 0, 1, 1]],
            linestyle=style_inv, color=color_tet, linewidth=width_tet)
    ax.plot([co[0, 1, 2, 0], co[0, 0, 1, 0]], [co[0, 1, 2, 1], co[0, 0, 1, 1]],
            linestyle=style_inv, color=color_tet, linewidth=width_tet)

    # Plot second tetrahedra
    ax.plot([co[1, 0, 0, 0], co[1, 0, 3, 0]], [co[1, 0, 0, 1], co[1, 0, 3, 1]],
            linestyle=style, color=color_tet, linewidth=width_tet)
    ax.plot([co[1, 0, 0, 0], co[1, 1, 0, 0]], [co[1, 0, 0, 1], co[1, 1, 0, 1]],
            linestyle=style, color=color_tet, linewidth=width_tet)
    ax.plot([co[1, 0, 3, 0], co[1, 1, 0, 0]], [co[1, 0, 3, 1], co[1, 1, 0, 1]],
            linestyle=style, color=color_tet, linewidth=width_tet)
    ax.plot([co[1, 0, 0, 0], co[1, 0, 1, 0]], [co[1, 0, 0, 1], co[1, 0, 1, 1]],
            linestyle=style_inv, color=color_tet, linewidth=width_tet)
    ax.plot([co[1, 0, 3, 0], co[1, 0, 1, 0]], [co[1, 0, 3, 1], co[1, 0, 1, 1]],
            linestyle=style_inv, color=color_tet, linewidth=width_tet)
    ax.plot([co[1, 1, 0, 0], co[1, 0, 1, 0]], [co[1, 1, 0, 1], co[1, 0, 1, 1]],
            linestyle=style_inv, color=color_tet, linewidth=width_tet)

    # Plot third tetrahedra
    ax.plot([co[2, 0, 2, 0], co[2, 0, 3, 0]], [co[2, 0, 2, 1], co[2, 0, 3, 1]],
            linestyle=style, color=color_tet, linewidth=width_tet)
    ax.plot([co[2, 0, 2, 0], co[2, 1, 2, 0]], [co[2, 0, 2, 1], co[2, 1, 2, 1]],
            linestyle=style, color=color_tet, linewidth=width_tet)
    ax.plot([co[2, 1, 2, 0], co[2, 0, 3, 0]], [co[2, 1, 2, 1], co[2, 0, 3, 1]],
            linestyle=style, color=color_tet, linewidth=width_tet)
    ax.plot([co[2, 0, 2, 0], co[2, 0, 1, 0]], [co[2, 0, 2, 1], co[2, 0, 1, 1]],
            linestyle=style_inv, color=color_tet, linewidth=width_tet)
    ax.plot([co[2, 0, 3, 0], co[2, 0, 1, 0]], [co[2, 0, 3, 1], co[2, 0, 1, 1]],
            linestyle=style_inv, color=color_tet, linewidth=width_tet)
    ax.plot([co[2, 1, 2, 0], co[2, 0, 1, 0]], [co[2, 1, 2, 1], co[2, 0, 1, 1]],
            linestyle=style_inv, color=color_tet, linewidth=width_tet)

    # Plot forth tetrahedra
    ax.plot([co[3, 1, 3, 0], co[3, 0, 3, 0]], [co[3, 1, 3, 1], co[3, 0, 3, 1]],
            linestyle=style, color=color_tet, linewidth=width_tet)
    ax.plot([co[3, 1, 3, 0], co[3, 1, 0, 0]], [co[3, 1, 3, 1], co[3, 1, 0, 1]],
            linestyle=style, color=color_tet, linewidth=width_tet)
    ax.plot([co[3, 1, 3, 0], co[3, 1, 2, 0]], [co[3, 1, 3, 1], co[3, 1, 2, 1]],
            linestyle=style, color=color_tet, linewidth=width_tet)
    ax.plot([co[3, 1, 0, 0], co[3, 0, 3, 0]], [co[3, 1, 0, 1], co[3, 0, 3, 1]],
            linestyle=style, color=color_tet, linewidth=width_tet)
    ax.plot([co[3, 1, 2, 0], co[3, 0, 3, 0]], [co[3, 1, 2, 1], co[3, 0, 3, 1]],
            linestyle=style, color=color_tet, linewidth=width_tet)
    ax.plot([co[3, 1, 2, 0], co[3, 1, 0, 0]], [co[3, 1, 2, 1], co[3, 1, 0, 1]],
            linestyle=style, color=color_tet, linewidth=width_tet)

    # Plot fifth tetrahedra
    ax.plot([co[4, 1, 1, 0], co[4, 1, 2, 0]], [co[4, 1, 1, 1], co[4, 1, 2, 1]],
            linestyle=style, color=color_tet, linewidth=width_tet)
    ax.plot([co[4, 1, 0, 0], co[4, 1, 2, 0]], [co[4, 1, 0, 1], co[4, 1, 2, 1]],
            linestyle=style, color=color_tet, linewidth=width_tet)
    ax.plot([co[4, 1, 1, 0], co[4, 1, 0, 0]], [co[4, 1, 1, 1], co[4, 1, 0, 1]],
            linestyle=style, color=color_tet, linewidth=width_tet)
    ax.plot([co[4, 0, 1, 0], co[4, 1, 2, 0]], [co[4, 0, 1, 1], co[4, 1, 2, 1]],
            linestyle=style_inv, color=color_tet, linewidth=width_tet)
    ax.plot([co[4, 1, 1, 0], co[4, 0, 1, 0]], [co[4, 1, 1, 1], co[4, 0, 1, 1]],
            linestyle=style_inv, color=color_tet, linewidth=width_tet)
    ax.plot([co[4, 1, 0, 0], co[4, 0, 1, 0]], [co[4, 1, 0, 1], co[4, 0, 1, 1]],
            linestyle=style_inv, color=color_tet, linewidth=width_tet)


    ### ----- Save figure ----- ###
    name = 'results_paper/discretization.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.show()
    plt.close(fig)

##################################################################
### ----------------- Plot 2D-stress cylinder ---------------- ###
##################################################################
def plot_2D_stress_cylinder():
    print('Plot 2D stress cylinder.')
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

    # Lengths are expressed in terms of lengths[2]
    Lz = lengths[2]
    lengths = [lengths[0] / Lz, lengths[1] / Lz, 1.]
    radius = radius / Lz

    # Stresses are expressed in terms of Youngs modulus
    Young = 100

    # z-coordinate of plots
    z_pt = 0

    ### ----- Read data ----- ###
    # File with data
    folder_cyl = f'results_paper/torsion_cylinder/'
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

    # Stress expressed in Young's modulus
    stress_ana = stress_ana / Young
    stress_num = stress_num / Young

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
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'wspace':0.4})
    for i in range(2):
        axes[i].set_aspect('equal')
        axes[i].set_xlabel('Position x / $L_z$ (-)', fontsize=size_large)
        axes[i].set_ylabel('Position y / $L_z$ (-)', fontsize=size_large)
        axes[i].tick_params(labelsize=tick_size)
        axes[i].xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05))
        axes[i].yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05))

    # Mask voxels of vacuum
    mask = geo.cylinder([Nxy, Nxy, Nz], lengths, radius)
    mask = 1 - mask[:, :, z_pt]

    # Plot
    helper = np.ma.masked_array(stress_ana[0, 2, :, :, z_pt], mask)
    im = axes[0].pcolormesh(x, y, helper.T, cmap=cmap,
                            vmin=vmin, vmax=vmax, rasterized=True)
    helper = np.ma.masked_array(stress_num[0, 2, :, :, z_pt], mask)
    im = axes[1].pcolormesh(x, y, helper.T, cmap=cmap,
                            vmin=vmin, vmax=vmax, rasterized=True)
    cbar = fig.colorbar(im, ax=axes, shrink=0.75)
    cbar.ax.set_ylabel('Stress $\sigma_{xz}$ \n per Youngs modulus $E$ (-)',
                       rotation=90, va='bottom', fontsize=size_large, labelpad=35.0)
    cbar.ax.tick_params(labelsize=tick_size)

    # Number subplots
    axes[0].text(-0.01, 0.105, '(a)  Analytical', fontsize=size_large, weight='bold', ha='left')
    axes[1].text(-0.01, 0.105, '(b)  Numerical', fontsize=size_large, weight='bold', ha='left')

    # Save and show
    plt.show()
    name = 'results_paper/stress_xz_cylinder.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)

##################################################################
### ---------------- Plot torsional stiffness ---------------- ###
##################################################################
def plot_error_torsional_stiff():
    print('Plot error stiffnesses.')
    ### ----- Parameters ----- ###
    # Colors
    color1 = 'red'
    color2 = 'blue'

    # Fontsize
    size_large = 16
    tick_size = 11

    ### ----- Read data ----- ###
    # Read error torsion stiffness cylinder
    folder_cyl = f'results_paper/torsion_cylinder/'
    data = np.loadtxt((folder_cyl + 'data.txt'), skiprows=1)
    Nx_cyl = data[:, 0].astype(int)
    err_stiff_cyl = data[:, 7]

    # Read error torsion stiffness square beam
    folder_sq = f'results_paper/torsion_square_beam/'
    data = np.loadtxt((folder_sq + 'data.txt'), skiprows=1)
    Nx_sq = data[:, 0].astype(int)
    err_stiff_sq = data[:, 5]

    # Read error bending stiffness rectangular beam
    name = 'results_paper/bending_square_beam/data.txt'
    N_bend = np.loadtxt(name, skiprows=1, max_rows=1).astype(int)
    err_stiff_bend = np.loadtxt(name, skiprows=3)

    ### ----- Plot data ----- ###
    fig, axes = plt.subplots(1, 2, figsize=(7, 4), sharey=True, layout='constrained')

    # Plot torsion stiffnesses
    axes[0].set_xlabel('Number of voxels $n_x$=$n_y$', fontsize=size_large)
    axes[0].set_ylabel('Difference between stiffnesses (%)', fontsize=size_large)
    axes[0].tick_params(axis='both', which='major', labelsize=tick_size)

    axes[0].plot(Nx_cyl, err_stiff_cyl, marker='.', color=color1)
    axes[0].plot(Nx_sq, err_stiff_sq, marker='x', color=color2)

    x = 1.2 * Nx_cyl[0]
    y = 0.8 * err_stiff_cyl[0]
    axes[0].text(x, y, 'circular cross-section', color=color1, fontsize=size_large)

    x = 1.02 * Nx_sq[-1]
    y = -0.18 * err_stiff_sq[0]
    axes[0].text(x, y, 'square cross-section', color=color2, fontsize=size_large, ha='right')

    # Plot bending stiffness
    axes[1].set_xlabel('Number of voxels $n_x$=$n_y$=$n_z$', fontsize=size_large)
    axes[1].tick_params(axis='both', which='major', labelsize=tick_size)
    axes[1].plot(N_bend, err_stiff_bend, marker='x', color=color2)

    # Numerate subplots
    axes[0].text(20, 4.2, '(a)  Torsion', fontsize=size_large, weight='bold', ha='left')
    axes[1].text(1, 4.2, '(b)  Bending', fontsize=size_large, weight='bold', ha='left')

    ### ----- Finish ----- ###
    name = 'results_paper/error_stiffnesses.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.show()
    plt.close(fig)

##################################################################
### ------------------ Plot geometry metamat ----------------- ###
##################################################################
def plot_geometry_chiral_metamat():
    print('Plot geometry chiral metamat.')
    ### ----- Parameters ----- ###
    # Where is the data saved?
    name_muspectre_geometry = 'results_paper/chiral_muspectre/chiral_geometry/geometry.png'
    name_comsol_1x1x1 = 'results_paper/chiral_comsol/chiral_comsol_1x1x1_unit_cells/geometry.png'
    name_comsol_3x3x2 = 'results_paper/chiral_comsol/chiral_comsol_3x3x2_unit_cells/geometry.png'

    # Plotting
    figsize= (10, 4)
    fontsize = 16
    scale_color = 'black'
    scale_width = 2

    color_para = 'red'
    linewidth_para = 1

    ### ----- Read data ----- ###
    # 3D image of geometries
    im_1x1x1 = np.asarray(Image.open(name_comsol_1x1x1))
    im_3x3x2 = np.asarray(Image.open(name_comsol_3x3x2))
    im_muspectre = np.asarray(Image.open(name_muspectre_geometry))

    ### ----- Plot geometries ----- ###
    fig = plt.figure(figsize=figsize)
    gs_left = fig.add_gridspec(1, 2, wspace = 0.1, right=0.55, left=0.05)
    gs_right = fig.add_gridspec(1, 1, left=0.5, right=0.95)

    # Plot 3D images
    ax = fig.add_subplot(gs_left[0, 0])
    ax.axis('off')
    implot = ax.imshow(im_1x1x1)
    ax.text(ax.get_xlim()[0] + 15,
            -15, '(a) 1 unit cell', color='black', fontsize=fontsize, weight='bold', ha='left')
    ax.plot([122 + 15, 290 + 15], [280 + 20, 238 + 20], linewidth=scale_width, color=scale_color)
    ax.text(61 + 145 + 15 + 20, 140 + 119 + 20 + 20, '0.5mm', fontsize=fontsize, color=scale_color,
            ha='center', va='center', rotation=15)
    ax_left = ax

    ax = fig.add_subplot(gs_left[0, 1])
    ax.axis('off')
    implot = ax.imshow(im_muspectre)
    ax.text(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) / 2,
            0, '(b) Discretized unit cell', color='black', fontsize=fontsize, weight='bold', ha='center')
    ax.plot([360 + 30, 780 + 30], [875 + 30, 640 + 30], linewidth=scale_width, color=scale_color)
    ax.text(180 + 380 + 30 + 30, 432.5 + 320 + 30 - 25, '0.5mm', fontsize=fontsize,
            color=scale_color, ha='center', va='top', rotation=30)

    ax = fig.add_subplot(gs_right[0, 0])
    ax.axis('off')
    implot = ax.imshow(im_3x3x2)
    ax.text(ax.get_xlim()[0] + 25,
            25, '(c) 3x3x2 unit cells', color='black', fontsize=fontsize, weight='bold', ha='left')
    ax.plot([130 + 10, 298 + 10], [255 + 10, 215 + 10], linewidth=scale_width, color=scale_color)
    ax.text(65 + 149 + 10, 127.5 + 107.5 + 10, '0.5mm', fontsize=fontsize,
            color=scale_color, ha='center', va='top', rotation=15)

    ### ----- Plot geometric parameters ----- ###
    ax = ax_left
    # Plot angle
    ax.plot([124, 280], [261, 222], linewidth=linewidth_para, color=color_para)
    ax.plot([124, 150], [261, 115], linewidth=linewidth_para, color=color_para)
    ax.add_artist(mpl.patches.Wedge((124, 261), 40, -80, -15, width=2*linewidth_para, fc=color_para))
    ax.text(130, 250, r'$\alpha$', fontsize=fontsize, color=color_para)

    # Plot radius
    Rx = 208
    Ry = 163
    R = 76
    ax.arrow(Rx, Ry, R / 2**0.5, -R / 2**0.5 - 3, length_includes_head=True, color=color_para,
             linewidth=linewidth_para, head_length=10, head_width=5)
    ax.text(Rx + R / 2 / 2**0.5, Ry - R / 2 / 2**0.5, '$r$', color=color_para, fontsize=fontsize, va='top')

    # Plot beam thickness
    ax.arrow(240, 37, 0, 20, length_includes_head=True, color=color_para,
             linewidth=linewidth_para, head_length=10, head_width=5)
    ax.arrow(240, 88, 0, -20, length_includes_head=True, color=color_para,
             linewidth=linewidth_para, head_length=10, head_width=5)
    ax.text(243, 43, r'$t$', fontsize=fontsize, color=color_para)

    ax.arrow(235, 185, 20/2**0.5, 20/2**0.5, length_includes_head=True, color=color_para,
             linewidth=linewidth_para, head_length=10, head_width=5)
    ax.arrow(270, 220, -20/2**0.5, -20/2**0.5, length_includes_head=True, color=color_para,
             linewidth=linewidth_para, head_length=10, head_width=5)
    ax.text(240, 185, r'$t$', fontsize=fontsize, color=color_para)

    # Plot side length
    ax.arrow(120, 279, 170, -42, length_includes_head=True, color=color_para,
             linewidth=linewidth_para, head_length=10, head_width=5)
    ax.text(200, 276, r'$a$', fontsize=fontsize, color=color_para)
    ax.arrow(120, 279, 0, -197, length_includes_head=True, color=color_para,
             linewidth=linewidth_para, head_length=10, head_width=5)
    ax.text(95, 180, r'$a$', fontsize=fontsize, color=color_para)
    ax.arrow(120, 279, -95, -70, length_includes_head=True, color=color_para,
             linewidth=linewidth_para, head_length=10, head_width=5)
    ax.text(60, 265, r'$a$', fontsize=fontsize, color=color_para)


    ### ----- Finish ----- ###
    name = 'results_paper/chiral_geometry.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.show()
    plt.close(fig)

##################################################################
### -------------- Plot mesh refinement metamat -------------- ###
##################################################################
def plot_results_mesh_refinement():
    print('Plot mesh refinement.')
    ### ----- Parameters ----- ###
    # Data
    name_muspectre = 'results_paper/chiral_muspectre/chiral_mesh_refinement/data.txt'
    folder_comsol = 'results_paper/chiral_comsol/chiral_comsol_1x1x1_unit_cells/'

    # Plotting
    figsize = (9, 3)
    fontsize = 13
    fontsize_small = 12
    color_com = 'red'
    color_mu = 'blue'
    color_slopes = 'black'
    tick_size = 10

    case = 1

    ### ----- Read data ----- ###
    # muSpectre
    data = np.loadtxt(name_muspectre, skiprows=1)
    nb_grid_pts = data[:, 0].astype(int)
    force_z_muspectre = data[:, 1]
    time_muspectre = data[:, 3]

    # Degrees of freedom muSpectre:
    nb_dof_muspectre_all = 6 * 2 * nb_grid_pts ** 3

    # Comsol
    data = np.loadtxt((folder_comsol + 'mesh_refinement_overview.txt'),
                      skiprows = 1)
    nb_dof_comsol = data[:, 1].astype(int)
    time_comsol = data[:, 2]
    force_z_comsol = data[:, 3]

    ### ----- Plot data ----- ###
    fig, axes = plt.subplots(1, 3, figsize=figsize, gridspec_kw={'bottom':0.2, 'wspace':0.5, 'top':0.8})

    # Plot Comsol convergence
    def format_ticks_a(x, pos):
        x = x / 1e6
        return f'${x:.0f}^6$'

    axes[0].set_xlabel('Degrees of freedom', fontsize=fontsize)
    axes[0].xaxis.set_major_formatter(mpl.ticker.FuncFormatter(format_ticks_a))
    axes[0].set_ylabel('Force in z-direction (N)', fontsize=fontsize)
    axes[0].plot(nb_dof_comsol, force_z_comsol, color=color_com,
                 marker='x')
    axes[0].tick_params(labelsize=tick_size)

    # Plot muSpectre convergence
    axes[1].set_xlabel('Number of voxels', fontsize=fontsize)
    axes[1].xaxis.set_major_formatter(r'${x:.0f}^3$')
    axes[1].yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.0005))
    axes[1].set_ylabel('Force in z-direction (N)', fontsize=fontsize, labelpad=0.95)
    axes[1].plot(nb_grid_pts, force_z_muspectre, color=color_mu,
                 marker='x')
    axes[1].tick_params(labelsize=tick_size)

    # Plot calculation time
    axes[2].set_xlabel('Degrees of freedom', fontsize=fontsize, labelpad=0.0)
    axes[2].set_ylabel('Time (s)', fontsize=fontsize)
    axes[2].set_yscale('log')
    axes[2].set_xscale('log')
    axes[2].tick_params(labelsize=tick_size)
    axes[2].plot(nb_dof_comsol, time_comsol, color=color_com,
                 marker='x', label='Comsol')
    axes[2].plot(nb_dof_muspectre_all, time_muspectre, color=color_mu,
                 marker='x', label='muSpectre')

    axes[2].text(2e5, 5e2, 'Comsol', color=color_com, fontsize=fontsize_small)
    axes[2].text(8e6, 1e1, 'muSpectre', color=color_mu, fontsize=fontsize_small)

    # Plot slopes for comparison
    if False: # Test that I plot the right functions
        a = 1
        b = 1e-5
        axes[2].plot([1e6, 1e8], [1e6**a, 1e8**a], color=color_slopes)
        axes[2].plot([1e6, 1e8], [1e6**a * b, 1e8**a * b], color=color_slopes)
        a = 2
        b = 1e-10
        axes[2].plot([1e6, 1e8], [1e6**a, 1e8**a], color=color_slopes)
        axes[2].plot([1e6, 1e8], [1e6**a * b, 1e8**a * b], color=color_slopes)
    if case == 1:
        axes[2].plot([6e7, 2e8], [6e7 * 8e-7, 2e8 * 8e-7], color=color_slopes)
        axes[2].text(4.5e7, 6.5e1, r'$\propto x$', color=color_slopes, fontsize=fontsize_small,
                     rotation=63)
        axes[2].plot([8e5, 1.5e6], [8e5**2 * 1e-10, 1.5e6**2 * 1e-10], color=color_slopes)
        axes[2].text(4e5, 9e1, r'$\propto x^2$', color=color_slopes, fontsize=fontsize_small,
                     rotation=80)

    # Number subplots
    axes[0].text(2.8e6, 0.01184, '(a) Convergence Comsol', fontsize=fontsize, va='bottom', ha='center', weight='bold')
    axes[1].text(180, 0.00776, '(b) Convergence muSpectre', fontsize=fontsize, ha='center', weight='bold')
    axes[2].text(4e4, 5.5e3, '(c) Calculation time', fontsize=fontsize, weight='bold', ha='left')

    ### ----- Finish ----- ###
    name = 'results_paper/chiral_mesh_refinement.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.show()
    plt.close(fig)

##################################################################
### ------------- Plot force vs. nb_unit_cells_z ------------- ###
##################################################################
def plot_force_vs_unit_cells():
    print('Plot force vs. nb_unit_cells_z.')
    ### ----- Parameters ------ ###
    # Data
    folder = 'results_paper/'
    nb_unit_cell_xy_list = [1, 3]
    nb_unit_cell_z_list = [1, 2, 3, 4]
    max_el_size = ['0.5e-5', '0.2e-5']

    # Plotting
    figsize = (7, 4)
    fontsize = 14
    tick_size = 12
    color_1x1 = 'blue'
    color_3x3 = 'red'
    linestyle_mu = '-'
    linestyle_com = '--'
    marker_1x1 = 'x'
    marker_3x3 = '.'

    ### ----- Read data ----- ###
    # Comsol
    forces_comsol = np.empty((len(nb_unit_cell_xy_list), len(nb_unit_cell_z_list)))
    for index_xy in range(len(nb_unit_cell_xy_list)):
        N_xy = nb_unit_cell_xy_list[index_xy]
        name = folder + f'chiral_comsol/chiral_comsol_{N_xy}x{N_xy}x{1}_unit_cells/'
        name += f'average_force_per_rot_max_el=' + max_el_size[index_xy] + '.txt'
        data = np.loadtxt(name, skiprows=5)
        #forces_comsol[index_xy, 0] = data[1]
        for index_z in range(0, len(nb_unit_cell_z_list)):
            N_z = nb_unit_cell_z_list[index_z]
            name = folder + f'chiral_comsol/chiral_comsol_{N_xy}x{N_xy}x{N_z}_unit_cells/'
            name += 'average_force_per_rot.txt'
            data = np.loadtxt(name, skiprows=5)
            forces_comsol[index_xy, index_z] = data[1]

    # muSpectre
    N_xy = nb_unit_cell_xy_list[0]
    name = folder + f'chiral_muspectre/chiral_Nuc={N_xy}x{N_xy}xNucz/data.txt'
    data = np.loadtxt(name, skiprows=1)
    N_z_1 = data[:, 0].astype(int)
    force_muspectre_1 = data[:, 4]

    #force_muspectre_1
    N_xy = nb_unit_cell_xy_list[1]
    name = folder + f'chiral_muspectre/chiral_Nuc={N_xy}x{N_xy}xNucz/data.txt'
    data = np.loadtxt(name, skiprows=1)
    N_z_2 = data[:, 0].astype(int)
    force_muspectre_2 = data[:, 4]

    ### ----- Plotting ----- ###
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel('Number of unit cells in z-direction $N_z$', fontsize=fontsize)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
    ax.tick_params(labelsize=tick_size)
    ax.set_ylabel('Force in z-direction (N)', fontsize=fontsize)
    ax.plot(nb_unit_cell_z_list, forces_comsol[0], marker=marker_1x1, color=color_1x1,
            linestyle=linestyle_com, label=f'Comsol N_xy={1}')
    ax.plot(nb_unit_cell_z_list, forces_comsol[1], marker=marker_3x3, color=color_3x3,
            linestyle=linestyle_com, label=f'Comsol N_xy={3}')
    ax.plot(N_z_1, force_muspectre_1, marker=marker_1x1, color=color_1x1,
            linestyle=linestyle_mu, label=f'muSpectre N_xy={nb_unit_cell_xy_list[0]}')
    ax.plot(N_z_2, force_muspectre_2, marker=marker_3x3, color=color_3x3,
            linestyle=linestyle_mu, label=f'muSpectre N_xy={nb_unit_cell_xy_list[1]}')
    ax.set_ylim(0, None)

    # Labels
    ax.text(1.82, 1e-2, 'Comsol: $N_{x}$=$N_{y}$=1', fontsize=fontsize, color=color_1x1)
    ax.text(2.5, 3.8e-3, 'Comsol: $N_{x}$=$N_{y}$=3', fontsize=fontsize, color=color_3x3)
    ax.text(1.0, 7.6e-3, 'muSpectre:$N_{x}$=$N_{y}$=1', fontsize=fontsize, color=color_1x1)
    ax.text(1.0, 2.5e-3, 'muSpectre:$N_{x}$=$N_{y}$=3', fontsize=fontsize, color=color_3x3)

    ### ----- Finish ----- ###
    name = 'results_paper/chiral_force_vs_unit_cells.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.show()
    plt.close(fig)

##################################################################
### ----------------- Plot 2D-stress metamat ----------------- ###
##################################################################
def plot_border_of_geometry(ax, nb_grid_pts_xy, X_xy, Y_xy, mask_xy,
                            linewidth=1., linecolor='black'):
    """
    Plot the border of the geometry described by mask.

    Input
    -----
    ax: matplotlib.pyplot.ax object
        Axis on which the border should be plotted.
    nb_grid_pts_xy: list with two ints
                    Number of pixels in each direction
    X_xy: np.array([nb_grid_pts[0]+1, nb_grid_pts[1]+1]) of floats
          x-coordinate for each node.
    Y_xy: np.array([nb_grid_pts[0]+1, nb_grid_pts[1]+1]) of floats
          y-coordinate for each node.
    mask_xy: np.array(nb_grid_pts) of ints
             Describes the geometry. 1 corresponds to material, 0 to void.
    linewidth: float
               Linewidth of the plotted lines. Default is 1
    linecolor: string
               Color of the plotted lines. Must be color recognized by
               matplotlib. Default is 'black'.
    """
    # Plot geometry in the interior of the plot
    for ix in range(nb_grid_pts_xy[0]-1):
        for iy in range(nb_grid_pts_xy[1]-1):
            if mask_xy[ix, iy] != mask_xy[ix+1, iy]:
                ax.plot([X_xy[ix+1, iy], X_xy[ix+1, iy+1]],
                        [Y_xy[ix+1, iy], Y_xy[ix+1, iy+1]],
                        linewidth=linewidth, color=linecolor)
            if mask_xy[ix, iy] != mask_xy[ix, iy+1]:
                ax.plot([X_xy[ix, iy+1], X_xy[ix+1, iy+1]],
                        [Y_xy[ix, iy+1], Y_xy[ix+1, iy+1]],
                        linewidth=linewidth, color=linecolor)

    # Plot geometry at the borders of the plot
    for ix in range(nb_grid_pts_xy[0]-1):
        if mask_xy[ix, -1] != mask_xy[ix+1, -1]:
            ax.plot([X_xy[ix+1, -2], X_xy[ix+1, -1]],
                    [Y_xy[ix+1, -2], Y_xy[ix+1, -1]],
                    linewidth=linewidth, color=linecolor)
        if mask_xy[ix, 0] == 1:
            ax.plot([X_xy[ix, 0], X_xy[ix+1, 0]], [Y_xy[ix, 0], Y_xy[ix+1, 0]],
                    linewidth=linewidth, color=linecolor)
        if mask_xy[ix, -1] == 1:
            ax.plot([X_xy[ix, -1], X_xy[ix+1, -1]], [Y_xy[ix, -1], Y_xy[ix+1, -1]],
                    linewidth=linewidth, color=linecolor)
    for iy in range(nb_grid_pts_xy[1]-1):
        if mask_xy[-1, iy] != mask_xy[-1, iy+1]:
            ax.plot([X_xy[-2, iy+1], X_xy[-1, iy+1]],
                    [Y_xy[-2, iy+1], Y_xy[-1, iy+1]],
                    linewidth=linewidth, color=linecolor)
        if mask_xy[0, iy] == 1:
            ax.plot([X_xy[0, iy], X_xy[0, iy+1]], [Y_xy[0, iy], Y_xy[0, iy+1]],
                    linewidth=linewidth, color=linecolor)
        if mask_xy[-1, iy] == 1:
            ax.plot([X_xy[-1, iy], X_xy[-1, iy+1]], [Y_xy[-1, iy], Y_xy[-1, iy+1]],
                    linewidth=linewidth, color=linecolor)

    # Plot geometry at the corners of the plot
    if mask_xy[-1, 0] == 1:
        ax.plot([X_xy[-2, 0], X_xy[-1, 0]], [Y_xy[-2, 0], Y_xy[-1, 0]],
                linewidth=linewidth, color=linecolor)
    if mask_xy[0, -1] == 1:
        ax.plot([X_xy[0, -2], X_xy[0, -1]], [Y_xy[0, -2], Y_xy[0, -1]],
                linewidth=linewidth, color=linecolor)
    if mask_xy[-1, -1] == 1:
        ax.plot([X_xy[-2, -1], X_xy[-1, -1]], [Y_xy[-2, -1], Y_xy[-1, -1]],
                linewidth=linewidth, color=linecolor)
        ax.plot([X_xy[-1, -2], X_xy[-1, -1]], [Y_xy[-1, -2], Y_xy[-1, -1]],
                linewidth=linewidth, color=linecolor)


def plot_2D_stress_chiral():
    print('Plot 2D stress chiral metamat.')

    ### ----- Parameters ----- ###
    # Data
    name_mu = 'results_paper/chiral_muspectre/chiral_Nuc=1x1xNucz/data_3D_Nucz=1.nc'
    name_comsol = 'results_paper/chiral_comsol/chiral_comsol_1x1x1_unit_cells/stress_zz_in_xz_plane.png'

    # Limits of colorbar (TODO: same as Comsol)
    vmin = -9 # Comsol: -9  muSpectre: -7.6
    vmax = 15 # Comsol: 15  muSpectre: 8.9

    # Scale for displacement (Same as Comsol)
    scale = 5

    # For plotting
    figsize = (7, 4)
    fontsize = 12
    linewidth_undef = 0.5
    linecolor_undef = 'black'
    iy = 1

    # Discretization muSpectre
    nb_grid_pts = [100, 100, 100]
    nb_quad_pts = 5

    # Loading
    twist = 0.05

    # Geometry
    a = 0.5
    thickness = 0.06 * a
    radius_out = 0.4 * a
    radius_inn = 0.34 * a
    angle_mat = np.pi * 35 / 180

    # Plot scale
    scale_length = 0.1
    scale_width = 1.5
    scale_color = 'black'
    scale_fontsize = 10

    ### ----- Geometry ----- ###
    mask, lengths =\
        geo.chiral_2_mult_unit_cell([1, 1, 1], nb_grid_pts, a, radius_out,
                                    radius_inn, thickness, alpha=angle_mat)

    x_rot_axis = lengths[0] / 2
    y_rot_axis = lengths[1] / 2

    ### ----- Read data ----- ###
    # muSpectre
    nc = netCDF4.Dataset(name_mu, 'r')
    displ_fluct = nc.variables['fluctuating_displacement'][3, :, 0, :, :, :]
    stress_zz = nc.variables['stress'][0, 2, 2]
    nc.close()

    # Average stress over quadrature points
    stress_zz = np.average(stress_zz, axis=0, weights=(2, 1, 1, 1, 1))

    ### ----- Calculate complete displacement ----- ###
    # Only interested in 1 plane
    displ = np.empty((3, nb_grid_pts[0]+1, nb_grid_pts[2]+1))
    displ[:, 0:-1, 0:-1] = displ_fluct[:, :, iy, :]

    # Complement periodically
    displ[:, -1, 0:-1] = displ_fluct[:, 0, iy, :]
    displ[:, 0:-1, -1] = displ_fluct[:, :, iy, 0]
    displ[:, -1, -1] = displ_fluct[:, 0, iy, 0]

    # Add displacement due to non-periodic torsion strain
    x = np.linspace(0, lengths[0], nb_grid_pts[0]+1, endpoint=True)
    y = np.linspace(0, lengths[1], nb_grid_pts[1]+1, endpoint=True)
    z = np.linspace(0, lengths[2], nb_grid_pts[2]+1, endpoint=True)
    helper = - twist * np.einsum('i,j->ij', y-y_rot_axis, z)
    displ[0, :, :] += helper[None, iy, :]
    helper = twist * np.einsum('i,j->ij', x-x_rot_axis, z)
    displ[1, :, :] += helper[:, :]

    # Initial and final node positions
    pos_initial = np.empty(displ.shape)
    pos_initial[0] = x[:, None]
    pos_initial[1] = y[iy]
    pos_initial[2] = z[None, :]
    pos_displ = pos_initial + scale * displ

    ### ----- Plot data ----- ###
    # Load Comsol
    im_com = np.asarray(Image.open(name_comsol)) # Note: Units in MPa

    # Prepare figure
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, wspace = 0.2)
    mask = mask.reshape(nb_grid_pts, order='F')

    # Plot Comsol
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.axis('off')
    implot = ax1.imshow(im_com)

    # Plot deformed geometry muSpectre
    ax = fig.add_subplot(gs[0, 0])
    ax.axis('off')
    ax.set_aspect('equal')

    X = pos_displ[0, :, :]
    Z = pos_displ[2, :, :]

    #print(np.amin(stress_zz[:, iy, :]), np.amax(stress_zz[:, iy, :]))

    mask_elements = mask[:, iy, :]
    mask_elements = (mask_elements == 0)
    n = np.ma.masked_array(stress_zz[:, iy, :], mask_elements)
    pm = ax.pcolormesh(X, Z, n, shading='flat', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(pm, ax = (ax, ax1), shrink = 0.7)
    cbar.ax.set_ylabel(r'Stress $\sigma_{zz}$ (MPa)',
                       rotation=-90, va='bottom', fontsize=fontsize)
    cbar.ax.tick_params(labelsize=scale_fontsize)

    # Plot initial geometry
    X = pos_initial[0, :, :]
    Z = pos_initial[2, :, :]

    plot_border_of_geometry(ax, nb_grid_pts, X, Z, mask[:, iy, :],
                            linewidth=linewidth_undef, linecolor=linecolor_undef)

    # Plot scales
    ax.plot([0.1, 0.1 + scale_length], [0, 0], color=scale_color, linewidth=scale_width)
    ax.text(0.1 + scale_length/2, 0.005, f'{scale_length}mm', color=scale_color, fontsize=scale_fontsize,
            ha='center', va='bottom')

    # Label subplots
    ax.text(ax.get_xlim()[0],
            ax.get_ylim()[1] + 0.03 * ax.get_ylim()[1],
            '(a) muSpectre', fontsize=fontsize, weight='bold', ha='left')
    ax1.text(ax1.get_xlim()[0],
             ax1.get_ylim()[1],
             '(b) Comsol', fontsize=fontsize, weight='bold', ha='left')


    ### ----- Finish ----- ###
    name = 'results_paper/stress_zz_chiral.png'
    fig.savefig(name, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close(fig)

##################################################################
### --------------------- Force vs twist --------------------- ###
##################################################################
def plot_force_vs_twist():
    print('Plot force vs. twist.')
    ### ----- Parameters ----- ###
    # Where is the data saved?
    name_muspectre = 'results_paper/chiral_muspectre/chiral_comp_cylinder/data.txt'
    name_comsol_1x1x4 = 'results_paper/chiral_comsol/chiral_comsol_1x1x4_unit_cells/influence_twist_overview.txt'
    name_comsol_3x3x4 = 'results_paper/chiral_comsol/chiral_comsol_3x3x4_unit_cells/influence_twist_overview.txt'

    # Plotting
    figsize= (10, 5)
    color_cyl = 'red'
    color_chi_1 = 'blue'
    color_chi_2 = 'orange'
    style_com = '--'
    fontsize = 14
    tick_size = 12
    marker_cyl = 's'
    marker_chi_1 = 'x'
    marker_chi_2 = '.'
    tick_size = 12


    ### ----- Read data ----- ###
    # Read muSpectre data
    twists = np.loadtxt(name_muspectre, skiprows=1, max_rows=1)
    force_cyl = np.loadtxt(name_muspectre, skiprows=5, max_rows=1)
    rot_angles_1x1x1 = np.loadtxt(name_muspectre, skiprows=7, max_rows=1)
    force_chi_1x1x1 = np.loadtxt(name_muspectre, skiprows=9, max_rows=1)
    rot_angles_3x3x1 = np.loadtxt(name_muspectre, skiprows=11, max_rows=1)
    force_chi_3x3x1 = np.loadtxt(name_muspectre, skiprows=13, max_rows=1)

    twists = twists * 1000 # twists in 1/m

    # Read comsol data
    data_com_1x1x4 = np.loadtxt(name_comsol_1x1x4, skiprows=1)
    data_com_3x3x4 = np.loadtxt(name_comsol_3x3x4, skiprows=1)


    ### ----- Plotting ----- ###
    # Prepare figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel('Twist angle per length $w$ (1/m)', fontsize=fontsize)
    ax.set_ylabel('Force in z-direction (N)', fontsize=fontsize)
    ax.tick_params(labelsize=tick_size)

    # Plot muSpectre results
    ax.plot(twists, force_cyl, marker=marker_cyl, color=color_cyl, label=r'muSpectre: Cylinder')
    ax.plot(twists, force_chi_1x1x1, marker=marker_chi_1, color=color_chi_1, label='muSpectre: $N_x=N_y=1$')
    ax.plot(twists, force_chi_3x3x1, marker=marker_chi_2, color=color_chi_2, label='muSpectre: $N_x=N_y=3$')

    # Plot comsol results
    ax.plot(data_com_1x1x4[:, 0], data_com_1x1x4[:, 2], linestyle=style_com, marker=marker_chi_1,
            color=color_chi_1, label='Comsol: $N_x=N_y=1$')
    ax.plot(data_com_3x3x4[:, 0], data_com_3x3x4[:, 2], linestyle=style_com, marker=marker_chi_2,
            color=color_chi_2, label='Comsol: $N_x=N_y=3$')

    # Legends
    ax.legend(fontsize=fontsize)

    # Print out stiffnesses
    print('Stiffness = Average force / twist angle')
    print(f'muSpectre 1x1x1uc: {force_chi_1x1x1[-1] / twists[-1]}')
    print(f'muSpectre 3x3x1uc: {force_chi_3x3x1[-1] / twists[-1]}')
    print(f'Comsol 1x1x4uc: {data_com_1x1x4[-1, 2] / data_com_1x1x4[-1, 0]}')
    print(f'Comsol 3x3x4uc: {data_com_3x3x4[-1, 2] / data_com_3x3x4[-1, 0]}')

    ### ----- Finish ----- ###
    name = 'results_paper/chiral_force_vs_twist.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.show()
    plt.close(fig)

##################################################################
### ------------------ Make plots for paper ------------------ ###
##################################################################
def plots_for_paper():
    # Plots
    #plot_models()
    #plot_discretization_3D_5_tetraedras()
    #plot_error_torsional_stiff()
    #plot_2D_stress_cylinder()
    #plot_geometry_chiral_metamat()
    #plot_results_mesh_refinement()
    #plot_force_vs_unit_cells()
    plot_2D_stress_chiral()
    plot_force_vs_twist()


if __name__ == "__main__":
    plots_for_paper()

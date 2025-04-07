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

import muSpectre as µ
from NuMPI import MPI
from NuMPI.Tools import Reduction
from NuMPI.IO import save_npy

from muChirality.EigenStrainTorsion import EigenStrain
from muSpectre.gradient_integration import get_complemented_positions
import muChirality.Geometries as geo

##################################################################
### --------------- Plot small strain rotation --------------- ###
##################################################################
def plot_small_strain_rotation_model():
    ### ----- Parameter definition ----- ###
    # Geometry
    Lz = 0.5
    Lxy = 0.3

    # Rotation
    twist = 0.09

    # Coordinate system
    Lc = 0.12
    angle_c = np.pi / 10

    # Colors
    color_def = 'black'
    color_undef = 'grey'
    color_twist = 'blue'
    color_c = 'black'
    color_rot_axis = 'red'

    # Lines
    width_d = 1.5
    width_u = 1.5
    style_u = '-'
    style_u_inv = ''
    style_inv = '--'
    style_rot_axis = ':'

    # Arrows
    head_width = 0.02
    head_length = 1.5 * head_width
    width_t = 0.001
    head_width_c = 0.01
    head_length_c = 1.5 * head_width_c
    width_c = 0.001

    # Fontsize
    fontsize = 20

    ### ----- Coordinates calculations ----- ###
    # Coordinates of bottom points
    B1_x = 1.8 * Lc
    B1_y = - 0.5 * Lc
    B2_x = B1_x - Lxy * np.sin(angle_c)
    B2_y = B1_y + Lxy * np.cos(angle_c)
    B3_x = B2_x + Lxy * np.cos(angle_c)
    B3_y = B2_y + Lxy * np.sin(angle_c)
    B4_x = B1_x + Lxy * np.cos(angle_c)
    B4_y = B1_y + Lxy * np.sin(angle_c)

    # Coordinates of top points (undeformed geometry)
    U1_x = B1_x
    U1_y = B1_y + Lz
    U2_x = B2_x
    U2_y = B2_y + Lz
    U3_x = B3_x
    U3_y = B3_y + Lz
    U4_x = B4_x
    U4_y = B4_y + Lz

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
    A_x = B1_x + Lxy / 2 * np.cos(angle_c) - Lxy / 2 * np.sin(angle_c)
    A2_y = B1_y + (A_x - B1_x) * np.tan(angle_c)
    A1_y = A2_y - 1.2 * Lc
    A4_y = U1_y + Lxy / 2 * np.sin(angle_c) + Lxy / 2 * np.cos(angle_c)
    A5_y = D3_y + 0.2 * Lxy
    A3_y = B1_y + Lxy / 2 * np.cos(angle_c) + Lxy / 2 * np.sin(angle_c)

    ### ----- Plotting small strain rotation ----- ###
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.axis('off')

    # Plot coordinate system
    ax.arrow(0, 0, Lc * np.cos(angle_c), Lc * np.sin(angle_c),
             color=color_c, width=width_c,
             length_includes_head=True, head_width = head_width_c,
             head_length=head_length_c)
    ax.arrow(0, 0, 0, Lc, color=color_c, width=width_c,
             length_includes_head=True, head_width = head_width_c,
             head_length=head_length_c)
    ax.arrow(0, 0, -Lc * np.sin(angle_c), Lc * np.cos(angle_c),
             color=color_c, width=width_c,
             length_includes_head=True, head_width = head_width_c,
             head_length=head_length_c)
    ax.text(Lc * 0.8, - Lc * 0.25, 'x', color=color_c, fontsize=fontsize)
    ax.text(-Lc * 0.75, Lc * 0.7, 'y', color=color_c, fontsize=fontsize)
    ax.text(0, Lc * 1.05, 'z', color=color_c, fontsize=fontsize)

    # Plot undeformed beam
    ax.plot([B1_x, B2_x], [B1_y, B2_y], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax.plot([B1_x, B4_x], [B1_y, B4_y], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax.plot([U1_x, U2_x], [U1_y, U2_y], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax.plot([U2_x, U3_x], [U2_y, U3_y], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax.plot([U3_x, U4_x], [U3_y, U4_y], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax.plot([U4_x, U1_x], [U4_y, U1_y], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax.plot([B1_x, U1_x], [B1_y, U1_y], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax.plot([B2_x, U2_x], [B2_y, U2_y], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax.plot([B4_x, U4_x], [B4_y, U4_y], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax.plot([B2_x, B3_x], [B2_y, B3_y], color=color_undef, linewidth=width_u, linestyle=style_u_inv)
    ax.plot([B3_x, B4_x], [B3_y, B4_y], color=color_undef, linewidth=width_u, linestyle=style_u_inv)
    ax.plot([B3_x, U3_x], [B3_y, U3_y], color=color_undef, linewidth=width_u, linestyle=style_u_inv)

    # Plot rotation axis
    ax.plot([A_x, A_x], [A1_y, A2_y], color=color_rot_axis)
    ax.plot([A_x, A_x], [A2_y, A4_y], color=color_rot_axis, linestyle=style_inv)
    ax.plot([A_x, A_x], [A4_y, A5_y], color=color_rot_axis)

    ax.arrow(A_x, A1_y, 0, Lc, color=color_rot_axis, length_includes_head=True,
             head_width=head_width, head_length=head_length, width=width_t)
    ax.text(A_x + 0.2 * Lc, A1_y + 0.25 * Lc, '$\hat{n}$', color=color_rot_axis,
            fontsize=fontsize)

    # Plot reference point
    ax.plot([A_x, A_x], [A3_y, A3_y], color=color_rot_axis, marker='o', markersize=5)
    ax.text(A_x + 0.2 * Lc, A3_y - 0.2 * Lc, r'$\vec X_0$', color=color_rot_axis, fontsize=fontsize)

    # Plot deformed beam
    ax.plot([D1_x, D2_x], [D1_y, D2_y], color=color_def, linewidth=width_d)
    ax.plot([D1_x, D4_x], [D1_y, D4_y], color=color_def, linewidth=width_d)
    ax.plot([D2_x, D3_x], [D2_y, D3_y], color=color_def, linewidth=width_d)
    ax.plot([D3_x, D4_x], [D3_y, D4_y], color=color_def, linewidth=width_d)

    ax.plot([B1_x, D1_x], [B1_y, D1_y], color=color_def, linewidth=width_d)
    ax.plot([B2_x, D2_x], [B2_y, D2_y], color=color_def, linewidth=width_d)
    ax.plot([B3_x, D3_x], [B3_y, D3_y], color=color_def, linewidth=width_d, linestyle=style_inv)
    ax.plot([B4_x, D4_x], [B4_y, D4_y], color=color_def, linewidth=width_d)

    ax.plot([B1_x, B2_x], [B1_y, B2_y], color=color_def, linewidth=width_d)
    ax.plot([B1_x, B4_x], [B1_y, B4_y], color=color_def, linewidth=width_d)
    ax.plot([B2_x, B3_x], [B2_y, B3_y], color=color_def, linewidth=width_d, linestyle=style_inv)
    ax.plot([B4_x, B3_x], [B4_y, B3_y], color=color_def, linewidth=width_d, linestyle=style_inv)

    # Plot twist
    ax.arrow(U1_x, U1_y, D1_x - U1_x, D1_y - U1_y, color=color_twist,
             length_includes_head=True, head_width=head_width,
             head_length=head_length, width=width_t)
    ax.arrow(U2_x, U2_y, D2_x - U2_x, D2_y - U2_y, color=color_twist,
             length_includes_head=True, head_width=head_width,
             head_length=head_length, width=width_t)
    ax.arrow(U3_x, U3_y, D3_x - U3_x, D3_y - U3_y, color=color_twist,
             length_includes_head=True, head_width=head_width,
             head_length=head_length, width=width_t)
    ax.arrow(U4_x, U4_y, D4_x - U4_x, D4_y - U4_y, color=color_twist,
             length_includes_head=True, head_width=head_width,
             head_length=head_length, width=width_t)
    ax.text(U2_x + (D2_x - U2_x) / 2, U2_y + (D2_y - U2_y) / 4, '$u_{max}$',
            color=color_twist, fontstyle='italic', ha='right', fontsize=fontsize)

    ### ----- Save figure ----- ###
    name = 'results_paper/rotation.pdf'
    fig.savefig(name, bbox_inches='tight')
    #plt.show()
    plt.close(fig)

##################################################################
### ------------------- Plot discretization ------------------ ###
##################################################################
def plot_discretization_3D_5_tetraedras():
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
    #plt.show()
    plt.close(fig)

##################################################################
### --------------- Plot small strain bending ---------------- ###
##################################################################
def plot_small_strain_bending_idea():
    ### ----- Parameter definition ----- ###
    # Geometry
    Lx = 8
    Lz = 0.8

    # Discretization
    Nx = 10
    Nz = 5

    # Loading
    curvature_max = 0.1

    # Material
    Poisson = 0.3

    # Coordinate system
    L_coord = 0.75
    head_width_coord = 0.1
    head_length_coord = 1.5 * head_width_coord
    width_coord = 0.001
    x_coord = 0
    z_coord = 0

    # Colors
    color_def = 'black'
    color_undef = 'grey'
    color_disp = 'blue'
    color_coord = 'blue'
    color_bend_axis = 'red'

    # Lines
    width_d = 1.5
    width_u = 1.5
    style_u = '-'
    style_d = '-'

    # Arrows
    head_width = 0.02
    head_length = 1.5 * head_width
    width_t = 0.001

    # Fontsize
    fontsize = 18

    ### ----- Calculate positions ----- ###
    # Undeformed beam
    x = np.linspace(0, Lx, Nx)
    z = np.linspace(0, Lz, Nz)
    X, Z = np.meshgrid(x, z)


    # Displacements (for y=0)
    disp_x = curvature_max * X * Z
    disp_z = - curvature_max / 2 * X**2 - Poisson * curvature_max / 2 * Z**2

    # Deformed beam
    pos_x = X + disp_x
    pos_z = Z + disp_z

    ### ----- Plot deformed beam ----- ###
    fig, ax= plt.subplots(1)
    ax.set_aspect('equal')
    ax.axis('off')

    # Plot coordinate system
    ax.arrow(x_coord, z_coord, L_coord, 0, color=color_coord, width=width_coord,
             length_includes_head=True, head_width=head_width_coord,
             head_length=head_length_coord)
    ax.arrow(x_coord, z_coord, 0, L_coord, color=color_coord, width=width_coord, length_includes_head=True,
             head_width=head_width_coord, head_length=head_length_coord)
    ax.text(x_coord + 0.75 * L_coord, z_coord - 0.5 * L_coord, 'x', color=color_coord, fontsize=fontsize)
    ax.text(x_coord - 0.5 * L_coord, z_coord + 0.75 * L_coord, 'z', color=color_coord, fontsize=fontsize)

    # Plot undeformed beam
    ax.plot(X[:, 0], Z[:, 0], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax.plot(X[:, -1], Z[:, -1], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax.plot(X[0, :], Z[0, :], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax.plot(X[-1, :], Z[-1, :], color=color_undef, linewidth=width_u, linestyle=style_u)

    # Plot deformed beam
    ax.plot(pos_x[:, 0], pos_z[:, 0], color=color_def, linewidth=width_d, linestyle=style_d)
    ax.plot(pos_x[:, -1], pos_z[:, -1], color=color_def, linewidth=width_d, linestyle=style_d)
    ax.plot(pos_x[0, :], pos_z[0, :], color=color_def, linewidth=width_d, linestyle=style_d)
    ax.plot(pos_x[-1, :], pos_z[-1, :], color=color_def, linewidth=width_d, linestyle=style_d)

    # Plot moment
    radius = 0.4
    theta1 = -35
    width = 0.03
    h = 0.15
    w = 0.15
    ax.add_artist(mpl.patches.Wedge((0, 0), radius, theta1, -90, width=width, fc='red'))
    theta1 = theta1 * np.pi / 180
    Bx = (radius - width/2) * np.cos(theta1)
    By = (radius - width/2) * np.sin(theta1)
    helper = (Bx**2 + By**2 + (radius-width/2)**2 - h**2) / 2
    Ax = helper**2 * Bx**2 - (Bx**2 + By**2) * (helper**2 - (radius - width/2)**2 * By**2)
    Ax = (helper * Bx - Ax**0.5) / (Bx**2 + By**2)
    Ay = (helper - Bx * Ax) / By
    Cx = Bx - w/2 * (By - Ay) / h
    Cy = By + w/2 * (Bx - Ax) / h
    Dx = Bx + w/2 * (By - Ay) / h
    Dy = By - w/2 * (Bx - Ax) / h
    ax.add_patch(mpl.patches.Polygon([(Ax, Ay), (Cx, Cy), (Dx, Dy)], closed=True,
                                     fc='red', ec='None'))
    ax.text(0.4, 0.3, 'M', color='red', fontsize=fontsize)

    ### ----- Plot curvature ? ----- ###
    if False:
        curv = curvature_max
        radius = 1 / curv
        Kx = 0
        Kz = pos_z[0, 0] - 1 / curv

        artist = mpl.patches.Circle((Kx, Kz), radius, ec='red', fc='None')
        ax.add_artist(artist)


    ### ----- Save figure ----- ###
    #plt.show()
    name = f'results_paper/bending.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)

##################################################################
### ----------------- Plot 2D-stress cylinder ---------------- ###
##################################################################
def plot_2D_stress_cylinder():
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
        axes[i].set_xlabel('Position x (-)', fontsize=size_large)
        axes[i].set_ylabel('Position y (-)', fontsize=size_large)

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
    cbar = fig.colorbar(im, ax=axes)
    cbar.ax.set_ylabel(r'Stress $\sigma_{xz}$ (-)',
                       rotation=-90, va='bottom', fontsize=size_large)

    # Number subplots
    axes[0].text(-0.3, 0.98, '(a) Analytical', fontsize=size_large)
    axes[1].text(-0.3, 0.98, '(b) Numerical', fontsize=size_large)

    # Save and show
    #plt.show()
    name = 'results_paper/stress_xz_cylinder.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)

##################################################################
### ---------------- Plot torsional stiffness ---------------- ###
##################################################################
def plot_error_torsional_stiff():
    ### ----- Parameters ----- ###
    # Colors
    color1 = 'red'
    color2 = 'blue'

    # Fontsize
    size_large = 15
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
    name = 'results_paper/error_stiffnesses.pdf'
    fig.savefig(name, bbox_inches='tight')
    #plt.show()
    plt.close(fig)

##################################################################
### ------------------ Plot geometry metamat ----------------- ###
##################################################################
def plot_geometry_chiral_metamat():
    ### ----- Parameters ----- ###
    # Where is the data saved?
    name_muspectre_geometry = 'results_paper/chiral_geometry_muspectre/geometry.png'
    name_muspectre_data = 'results_paper/chiral_comp_cylinder/data.txt'
    name_comsol_1x1x1 = 'results_paper/chiral_comsol_1x1x1_unit_cells/geometry.png'
    name_comsol_3x3x2 = 'results_paper/chiral_comsol_3x3x2_unit_cells/geometry.png'

    # Plotting
    figsize= (8, 6)
    color_cyl = 'red'
    color_chi_1 = 'blue'
    color_chi_2 = 'orange'
    fontsize = 15
    tick_size = 11

    ### ----- Read data ----- ###
    # 3D image of geometries
    im_1x1x1 = np.asarray(Image.open(name_comsol_1x1x1))
    im_3x3x2 = np.asarray(Image.open(name_comsol_3x3x2))
    im_muspectre = np.asarray(Image.open(name_muspectre_geometry))

    # Read force in z-direction
    twists = np.loadtxt(name_muspectre_data, skiprows=1, max_rows=1)
    force_cyl = np.loadtxt(name_muspectre_data, skiprows=3, max_rows=1)
    force_chi_1x1x1 = np.loadtxt(name_muspectre_data, skiprows=5, max_rows=1)
    force_chi_3x3x1 = np.loadtxt(name_muspectre_data, skiprows=5, max_rows=1)

    ### ----- Plot data ----- ###
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, wspace = 0.5)

    # Plot 3D images
    ax = fig.add_subplot(gs[0, 0])
    ax.axis('off')
    implot = ax.imshow(im_1x1x1)
    #ax.text(-180, 120, '(a)', color='black', fontsize=fontsize)

    ax = fig.add_subplot(gs[1, 0])
    ax.axis('off')
    implot = ax.imshow(im_3x3x2)

    ax = fig.add_subplot(gs[0, 1])
    ax.axis('off')
    implot = ax.imshow(im_muspectre)


    # Plot force in z-direction
    ax = fig.add_subplot(gs[1, 1])
    ax.set_xlabel(r'Twist $w$ (-)', fontsize=fontsize)
    ax.set_ylabel(r'Force in z-direction (-)', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)

    ax.plot(twists, force_cyl, marker='x', color=color_cyl)
    ax.plot(twists, force_chi_1x1x1, marker='x', color=color_chi_1)
    #ax.plot(twists, force_chi_3x3x1, marker='x', color=color_chi_2)

    #ax.text(-0.3, 0.007, '(b)', color='black', fontsize=size_large)

    # Legend
    #ax.text(-0.175, 0.001, 'cylinder', color=color_cyl, fontsize=fontsize)
    #ax.text(0.05, 0.005, 'chiral', color=color_chi_1, fontsize=fontsize, ha='right')


    ### ----- Finish ----- ###
    name = 'results_paper/chiral_geometry.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.show()
    plt.close(fig)

##################################################################
### -------------- Plot mesh refinement metamat -------------- ###
##################################################################
def plot_results_mesh_refinement():
    ### ----- Parameters ----- ###
    # Data
    name_muspectre = 'results_paper/chiral_mesh_refinement/data.txt'
    folder_comsol = 'results_paper/chiral_comsol_1x1x1_unit_cells/'

    # Plotting
    figsize = (7, 4)
    fontsize = 12
    color_com = 'red'
    color_mu = 'blue'

    ### ----- Read data ----- ###
    # muSpectre
    data = np.loadtxt(name_muspectre, skiprows=1)
    nb_grid_pts = data[:, 0].astype(int)
    force_z_muspectre = data[:, 1]
    nb_mat_voxels = data[:, 2].astype(int)
    time_muspectre = data[:, 3]

    # Degrees of freedom muSpectre:
    # nb_dof = nb_dof_per_element * nb_element_per_voxel * nb_voxel
    nb_dof_muspectre_all = 6 * 2 * nb_grid_pts ** 3
    nb_dof_muspectre_mat = 6 * 2 * nb_mat_voxels

    # Comsol
    data = np.loadtxt((folder_comsol + 'mesh_refinement_overview.txt'),
                      skiprows = 1)
    nb_dof_comsol = data[:, 1].astype(int)
    time_comsol = data[:, 2]
    force_z_comsol = data[:, 3]

    ### ----- Plot data ----- ###
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Plot Comsol convergence
    axes[0].set_xlabel('Degrees of freedom', fontsize=fontsize)
    axes[0].set_ylabel('Average force_z (N)', fontsize=fontsize)
    axes[0].plot(nb_dof_comsol, force_z_comsol, color=color_com,
                 marker='x')

    # Plot muSpectre convergence
    axes[1].set_xlabel('Degrees of freedom', fontsize=fontsize)
    axes[1].set_ylabel('Average force_z (N)', fontsize=fontsize)
    axes[1].plot(nb_dof_muspectre_all, force_z_muspectre, color=color_mu,
                 marker='x')

    # Plot calculation time
    axes[2].set_xlabel('Degrees of freedom', fontsize=fontsize)
    axes[2].set_ylabel('Time (s)', fontsize=fontsize)
    axes[2].plot(nb_dof_comsol, time_comsol, color=color_com,
                 marker='x', label='comsol')
    axes[2].plot(nb_dof_muspectre_all, time_muspectre, color=color_mu,
                 marker='x', label='muspectre')
    axes[2].plot(nb_dof_muspectre_mat, time_muspectre, color=color_mu,
                 marker='x', linestyle='--', label='muspectre (dof in material)')

    axes[2].legend()

    # Number subplots
    # TODO

    ### ----- Finish ----- ###
    name = 'results_paper/chiral_mesh_refinement.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.show()
    plt.close(fig)

##################################################################
### ------------- Plot force vs. nb_unit_cells_z ------------- ###
##################################################################
def plot_force_vs_unit_cells():
    ### ----- Parameters ------ ###
    # Data
    folder = 'results_paper/'
    nb_unit_cell_xy_list = [1, 3]
    nb_unit_cell_z_list = [1, 2, 3, 4]
    max_el_size = ['0.5e-5', '0.2e-5']

    # Plotting
    figsize = (7, 4)
    fontsize = 12
    color_com = 'blue'
    color_mu = 'red'
    linestyles = ['-', '--']

    ### ----- Read data ----- ###
    # Comsol
    forces_comsol = np.empty((len(nb_unit_cell_xy_list), len(nb_unit_cell_z_list)))
    for index_xy in range(len(nb_unit_cell_xy_list)):
        N_xy = nb_unit_cell_xy_list[index_xy]
        name = folder + f'chiral_comsol_{N_xy}x{N_xy}x{1}_unit_cells/'
        name += f'average_force_per_rot_max_el=' + max_el_size[index_xy] + '.txt'
        data = np.loadtxt(name, skiprows=5)
        forces_comsol[index_xy, 0] = data[2]
        for index_z in range(1, len(nb_unit_cell_z_list)):
            N_z = nb_unit_cell_z_list[index_z]
            name = folder + f'chiral_comsol_{N_xy}x{N_xy}x{N_z}_unit_cells/'
            name += 'average_force_per_rot.txt'
            data = np.loadtxt(name, skiprows=5)
            forces_comsol[index_xy, index_z] = data[2]

    # muSpectre
    N_xy = nb_unit_cell_xy_list[0]
    name = folder + f'chiral__Nuc={N_xy}x{N_xy}xN_uc_z/data.txt'
    data = np.loadtxt(name, skiprows=1)
    N_z_1 = data[:, 0].astype(int)
    force_muspectre_1 = data[:, 4]

    force_muspectre_1
    N_xy = nb_unit_cell_xy_list[1]
    name = folder + f'chiral__Nuc={N_xy}x{N_xy}xN_uc_z/data.txt'
    data = np.loadtxt(name, skiprows=1)
    N_z_2 = data[:, 0].astype(int)
    force_muspectre_2 = data[:, 4]

    ### ----- Plotting ----- ###
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel('Number of unit cells in z-direction $N_z$', fontsize=fontsize)
    ax.set_ylabel('Average force per unit cell (N/rad)', fontsize=fontsize)
    for i, N_xy in enumerate(nb_unit_cell_xy_list):
        ax.plot(nb_unit_cell_z_list, forces_comsol[i], marker='x',
                color=color_com, linestyle=linestyles[i],
                label=f'Comsol N_xy={N_xy}')
    ax.plot(N_z_1, force_muspectre_1, marker='x', color=color_mu,
            linestyle=linestyles[0], label=f'muSpectre N_xy={nb_unit_cell_xy_list[0]}')
    ax.plot(N_z_2, force_muspectre_2, marker='x', color=color_mu,
            linestyle=linestyles[1], label=f'muSpectre N_xy={nb_unit_cell_xy_list[1]}')

    ax.legend()

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

def plot_2D_stress_chiral_metamat():
    ### ----- Parameters ----- ###
    # Data
    folder_mu = 'results_paper/chiral__Nuc=1x1xN_uc_z/'
    name_comsol = 'results_paper/chiral_comsol_1x1x1_unit_cells/stress_zz_in_xz_plane.png'

    # Limits of colorbar (TODO: same as Comsol)
    vmin = -3
    vmax = 6

    # Scale for displacement (TODO: same as Comsol)
    scale = 5

    # For plotting
    figsize = (7, 4)
    fontsize = 12
    linewidth_undef = 1.5
    linecolor_undef = 'black'

    # Discretization muSpectre
    nb_grid_pts_uc = [30, 30, 30]
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
    fft = 'fftw'
    Young = 2600
    Poisson = 0.4

    ### ----- Read data ----- ###
    # Comsol
    # TODO: UNITS OF COMSOL
    im_com = np.asarray(Image.open(name_comsol))


    # muSpectre strain
    folder = folder_mu + 'N_uc_z=1__strains/'
    strain = np.empty((3, 3, 5, *nb_grid_pts_uc))
    for i_quad in range(nb_quad_pts):
        name = folder + f'quad_pt_{i_quad}_entry_00.npy'
        strain[0, 0, i_quad] = np.load(name)
        name = folder + f'quad_pt_{i_quad}_entry_01.npy'
        strain[0, 1, i_quad] = np.load(name)
        strain[1, 0, i_quad] = strain[0, 1, i_quad]
        name = folder + f'quad_pt_{i_quad}_entry_02.npy'
        strain[0, 2, i_quad] = np.load(name)
        strain[2, 0, i_quad] = strain[0, 2, i_quad]
        name = folder + f'quad_pt_{i_quad}_entry_11.npy'
        strain[1, 1, i_quad] = np.load(name)
        name = folder + f'quad_pt_{i_quad}_entry_12.npy'
        strain[1, 2, i_quad] = np.load(name)
        strain[2, 1, i_quad] = strain[1, 2, i_quad]
        name = folder + f'quad_pt_{i_quad}_entry_22.npy'
        strain[2, 2, i_quad] = np.load(name)

    # muSpectre stress (zz-component)
    folder = folder_mu + 'N_uc_z=1__stress/'
    stress_zz = np.empty((5, *nb_grid_pts_uc))
    for i_quad in range(nb_quad_pts):
        name = folder + f'quad_pt_{i_quad}_entry_22.npy'
        stress_zz[i_quad] = np.load(name)

    # Average stress over quadrature points
    stress_zz = np.average(stress_zz, axis=0, weights=(2, 1, 1, 1, 1))

    ### ----- Calculate muSpectre displacement ----- ###
    # Mask describing geometry
    mask, lengths =\
        geo.chiral_2_mult_unit_cell([1, 1, 1], nb_grid_pts_uc, a, radius_out,
                                    radius_inn, thickness, alpha=angle_mat)

    # Create + initialize muSpectre cell
    cell = µ.Cell(nb_grid_pts_uc, lengths, formulation, gradient,
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
    eigen_class = EigenStrain(cell.pixels, twist, lengths, nb_grid_pts_uc,
                              x_rot_axis, y_rot_axis)

    # Calculate displacement of periodic strain
    strain_no_eigen = strain.copy()
    eigen_class.remove_eigen_strain_func(strain_no_eigen)
    [x_0, y_0, z_0], [x_displ, y_displ, z_displ] \
        = get_complemented_positions("0d", cell, strain_array=strain_no_eigen, F0=F0,
                                     periodically_complemented=True)

    # Add displacement of rotational strain
    x = np.linspace(0, lengths[0], nb_grid_pts_uc[0]+1, endpoint=True)
    y = np.linspace(0, lengths[1], nb_grid_pts_uc[1]+1, endpoint=True)
    z = np.linspace(0, lengths[2], nb_grid_pts_uc[2]+1, endpoint=True)
    helper = - twist * np.einsum('i,j->ij', y-y_rot_axis, z)
    x_displ += helper[None, :, :]
    helper = twist * np.einsum('i,j->ij', x-x_rot_axis, z)
    y_displ += helper[:, None, :]

    # Initial and final node positions
    pos_initial = np.asarray([x_0, y_0, z_0])
    displ = np.asarray([x_displ, y_displ, z_displ])
    pos_displ = pos_initial + scale * displ

    ### ----- Plot data ----- ###
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, wspace = 0.5)
    mask = mask.reshape(nb_grid_pts_uc, order='F')

    # Plot Comsol
    # TODO: Add directions? Add scale?
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.axis('off')
    implot = ax1.imshow(im_com)

    # Plot deformed geometry muSpectre
    ax = fig.add_subplot(gs[0, 0])
    ax.set_aspect('equal')
    iy = 1

    X = pos_displ[0, :, iy, :]
    Z = pos_displ[2, :, iy, :]

    mask_elements = mask[:, iy, :]
    mask_elements = (mask_elements == 0)
    n = np.ma.masked_array(stress_zz[:, iy, :], mask_elements)
    pm = ax.pcolormesh(X, Z, n, shading='flat', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(pm, ax=(ax, ax1))
    cbar.ax.set_ylabel(r'Stress $\sigma_{zz}$ (MPa)',
                       rotation=-90, va='bottom', fontsize=fontsize)

    # Plot initial geometry
    X = pos_initial[0, :, iy, :]
    Z = pos_initial[2, :, iy, :]

    plot_border_of_geometry(ax, nb_grid_pts_uc, X, Z, mask[:, iy, :],
                            linewidth=linewidth_undef, linecolor=linecolor_undef)


    ### ----- Finish ----- ###
    name = 'results_paper/stress_zz_chiral.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.show()
    plt.close(fig)

##################################################################
### ------------------ Make plots for paper ------------------ ###
##################################################################
def plots_for_paper():
    ### ----- Small strain rotation ----- ###
    print('Plot definition small strain rotation.')
    plot_small_strain_rotation_model()

    ### ----- Small strain bending ----- ###
    print('Plot definition small strain bending.')
    plot_small_strain_bending_idea()

    ### ----- Discretization of cube ----- ###
    print('Plot discretization.')
    plot_discretization_3D_5_tetraedras()

    ### ----- 2D-cut stress cylinder ------ ###
    # TODO: Check transpose of pcolormesh
    print('Plot 2D stress cylinder.')
    plot_2D_stress_cylinder()

    ### ----- Torsional stiffnesses ----- ###
    print('Plot error stiffnesses.')
    plot_error_torsional_stiff()

    ### ----- Geometry chiral metamaterial ----- ###
    print('Plot geometry chiral metamat.')
    # TODO
    plot_geometry_chiral_metamat()

    ### ----- Mesh refinement ----- ###
    print('Plot mesh refinement.')
    plot_results_mesh_refinement()
    # TODO: muSpectre
    # TODO: Comsol
    # TODO: Calculation time (both)

    ### ----- Force vs. nb_unit_cells_z ----- ###
    print('Plot force vs. nb_unit_cells_z.')
    # TODO
    plot_force_vs_unit_cells()

    ### ----- 2D-cut stress chiral metamat ----- ###
    print('Plot 2D stress chiral metamat.')
    plot_2D_stress_chiral_metamat()
    # TODO: muSpectre
    # TODO: Comsol


if __name__ == "__main__":
    plots_for_paper()

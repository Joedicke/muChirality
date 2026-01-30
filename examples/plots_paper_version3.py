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
def plot_torsion_model():
    print('Plot definition small strain rotation.')
    ### ----- Parameter definition ----- ###
    # Geometry
    Lz = 0.5
    Lxy = 0.3
    L_void = 0.05

    # Loading
    twist = 0.09

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
    color_coord = 'black'
    color_rot_axis = 'red'
    color_geo = 'dimgray'
    color_void = 'lightgray'

    # Lines
    width_d = 1.5
    width_u = 1.5
    style_u = '-'
    style_u_inv = ''
    style_inv = '--'
    style_rot_axis = '--'
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

    ### ----- Coordinates (3D plot) ----- ###
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

    ### ---- Prepare figure ----- ###
    fig = plt.figure(figsize=(7, 4))
    grid_left = fig.add_gridspec(1, 1, right=0.4)
    grid_right = fig.add_gridspec(2, 2, left=0.4)

    ax_3D = fig.add_subplot(grid_left[0, 0])
    ax_3D.set_aspect('equal')
    ax_3D.axis('off')

    ax_top_undef = fig.add_subplot(grid_right[0, 0])
    ax_top_undef.set_aspect('equal')
    ax_top_undef.axis('off')

    ax_top_def = fig.add_subplot(grid_right[1, 0])
    ax_top_def.set_aspect('equal')
    ax_top_def.axis('off')

    ax_side_undef = fig.add_subplot(grid_right[0, 1])
    ax_side_undef.set_aspect('equal')
    ax_side_undef.axis('off')

    ax_side_def = fig.add_subplot(grid_right[1, 1])
    ax_side_def.set_aspect('equal')
    ax_side_def.axis('off')

    ### ----- Plotting 3D small strain rotation ----- ###
    # Plot coordinate system
    ax_3D.arrow(0, 0, Lc * np.cos(angle_c), Lc * np.sin(angle_c),
             color=color_coord, width=width_c,
             length_includes_head=True, head_width = head_width_c,
             head_length=head_length_c)
    ax_3D.arrow(0, 0, 0, Lc, color=color_coord, width=width_c,
             length_includes_head=True, head_width = head_width_c,
             head_length=head_length_c)
    ax_3D.arrow(0, 0, -Lc * np.sin(angle_c), Lc * np.cos(angle_c),
             color=color_coord, width=width_c,
             length_includes_head=True, head_width = head_width_c,
             head_length=head_length_c)
    ax_3D.text(Lc * 0.8, - Lc * 0.25, 'x', color=color_coord, fontsize=fontsize)
    ax_3D.text(-Lc * 0.75, Lc * 0.7, 'y', color=color_coord, fontsize=fontsize)
    ax_3D.text(0, Lc * 1.05, 'z', color=color_coord, fontsize=fontsize)

    # Plot undeformed beam
    ax_3D.plot([B1_x, B2_x], [B1_y, B2_y], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax_3D.plot([B1_x, B4_x], [B1_y, B4_y], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax_3D.plot([U1_x, U2_x], [U1_y, U2_y], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax_3D.plot([U2_x, U3_x], [U2_y, U3_y], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax_3D.plot([U3_x, U4_x], [U3_y, U4_y], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax_3D.plot([U4_x, U1_x], [U4_y, U1_y], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax_3D.plot([B1_x, U1_x], [B1_y, U1_y], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax_3D.plot([B2_x, U2_x], [B2_y, U2_y], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax_3D.plot([B4_x, U4_x], [B4_y, U4_y], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax_3D.plot([B2_x, B3_x], [B2_y, B3_y], color=color_undef, linewidth=width_u, linestyle=style_u_inv)
    ax_3D.plot([B3_x, B4_x], [B3_y, B4_y], color=color_undef, linewidth=width_u, linestyle=style_u_inv)
    ax_3D.plot([B3_x, U3_x], [B3_y, U3_y], color=color_undef, linewidth=width_u, linestyle=style_u_inv)

    # Plot rotation axis
    ax_3D.plot([A_x, A_x], [A1_y, A2_y], color=color_rot_axis)
    ax_3D.plot([A_x, A_x], [A2_y, A4_y], color=color_rot_axis, linestyle=style_inv)
    ax_3D.plot([A_x, A_x], [A4_y, A5_y], color=color_rot_axis)

    ax_3D.arrow(A_x, A1_y, 0, Lc, color=color_rot_axis, length_includes_head=True,
             head_width=head_width, head_length=head_length, width=width_t)
    ax_3D.text(A_x + 0.2 * Lc, A1_y + 0.25 * Lc, '$\hat{n}$', color=color_rot_axis,
            fontsize=fontsize)

    # Plot reference point
    ax_3D.plot([A_x, A_x], [A3_y, A3_y], color=color_rot_axis, marker='o', markersize=5)
    ax_3D.text(A_x + 0.2 * Lc, A3_y - 0.3 * Lc, r'$\vec X_0$', color=color_rot_axis, fontsize=fontsize)

    # Plot deformed beam
    ax_3D.plot([D1_x, D2_x], [D1_y, D2_y], color=color_def, linewidth=width_d)
    ax_3D.plot([D1_x, D4_x], [D1_y, D4_y], color=color_def, linewidth=width_d)
    ax_3D.plot([D2_x, D3_x], [D2_y, D3_y], color=color_def, linewidth=width_d)
    ax_3D.plot([D3_x, D4_x], [D3_y, D4_y], color=color_def, linewidth=width_d)

    ax_3D.plot([B1_x, D1_x], [B1_y, D1_y], color=color_def, linewidth=width_d)
    ax_3D.plot([B2_x, D2_x], [B2_y, D2_y], color=color_def, linewidth=width_d)
    ax_3D.plot([B3_x, D3_x], [B3_y, D3_y], color=color_def, linewidth=width_d, linestyle=style_inv)
    ax_3D.plot([B4_x, D4_x], [B4_y, D4_y], color=color_def, linewidth=width_d)

    ax_3D.plot([B1_x, B2_x], [B1_y, B2_y], color=color_def, linewidth=width_d)
    ax_3D.plot([B1_x, B4_x], [B1_y, B4_y], color=color_def, linewidth=width_d)
    ax_3D.plot([B2_x, B3_x], [B2_y, B3_y], color=color_def, linewidth=width_d, linestyle=style_inv)
    ax_3D.plot([B4_x, B3_x], [B4_y, B3_y], color=color_def, linewidth=width_d, linestyle=style_inv)

    # Plot twist
    ax_3D.arrow(U1_x, U1_y, D1_x - U1_x, D1_y - U1_y, color=color_twist,
             length_includes_head=True, head_width=head_width,
             head_length=head_length, width=width_t)
    ax_3D.arrow(U2_x, U2_y, D2_x - U2_x, D2_y - U2_y, color=color_twist,
             length_includes_head=True, head_width=head_width,
             head_length=head_length, width=width_t)
    ax_3D.arrow(U3_x, U3_y, D3_x - U3_x, D3_y - U3_y, color=color_twist,
             length_includes_head=True, head_width=head_width,
             head_length=head_length, width=width_t)
    ax_3D.arrow(U4_x, U4_y, D4_x - U4_x, D4_y - U4_y, color=color_twist,
             length_includes_head=True, head_width=head_width,
             head_length=head_length, width=width_t)

    ### ----- Plot top view ----- ###
    # General
    xlim = (-3*L_void, Lxy + 3*L_void)
    ylim = (-5*L_void, Lxy + 3*L_void)

    coord_sys_x = -1.5 * disp
    coord_sys_y = -2.5 * disp

    disp_void = disp / (Lxy / 2) * (Lxy / 2 + L_void)

    # Undeformed
    void = mpl.patches.Rectangle((-L_void, -L_void), Lxy+2*L_void, Lxy+2*L_void, ec='none', fc=color_void)
    ax_top_undef.add_artist(void)
    geo = mpl.patches.Rectangle((0, 0), Lxy, Lxy, ec='none', fc=color_geo)
    ax_top_undef.add_artist(geo)

    ax_top_undef.arrow(coord_sys_x, coord_sys_y, Lc, 0, color=color_coord, length_includes_head=True,
                    head_width=head_width_c, head_length=head_length_c, width=width_c)
    ax_top_undef.arrow(coord_sys_x, coord_sys_y, 0, Lc, color=color_coord, length_includes_head=True,
                    head_width=head_width_c, head_length=head_length_c, width=width_c)
    ax_top_undef.text(coord_sys_x + Lc/2, coord_sys_y + 0.2 * disp, 'x', fontsize=fontsize, color=color_coord)
    ax_top_undef.text(coord_sys_x + 0.2 * disp, coord_sys_y + Lc/2, 'y', fontsize=fontsize, color=color_coord)

    ax_top_undef.set_xlim(xlim[0], xlim[1])
    ax_top_undef.set_ylim(ylim[0], ylim[1])

    # Labels
    ax_top_undef.annotate('Geometry', (0.75 * Lxy, 0.75 * Lxy), xytext=(Lxy + 2 * L_void, 0.9 * Lxy),
                          arrowprops=dict(facecolor='black', width=1, headwidth=5, headlength=7),
                          annotation_clip=False, fontsize=fontsize, va='top')
    ax_top_undef.annotate('Void', (Lxy + L_void / 2, 0.25 * Lxy), xytext=(Lxy + 4 * L_void, 0.2 * Lxy),
                          arrowprops=dict(facecolor='black', width=1, headwidth=5, headlength=7),
                          annotation_clip=False, fontsize=fontsize, va='center')

    # Deformed
    void = mpl.patches.Polygon(((-L_void+disp_void, -L_void-disp_void), (Lxy+L_void+disp_void, -L_void+disp_void),
                                (Lxy+L_void-disp_void, Lxy+L_void+disp_void), (-L_void-disp_void, Lxy+L_void-disp_void)),
                               closed=True, ec='none', fc=color_void)
    ax_top_def.add_artist(void)
    #geo2 = mpl.patches.Rectangle((0, 0), Lxy, Lxy, ec='none', fc='red')
    #ax_top_def.add_artist(geo2)
    geo = mpl.patches.Polygon(((disp, -disp), (Lxy+disp, disp), (Lxy-disp, Lxy+disp), (-disp, Lxy-disp)), closed=True,
                               ec='none', fc=color_geo)
    ax_top_def.add_artist(geo)

    ax_top_def.arrow(coord_sys_x, coord_sys_y, Lc, 0, color=color_coord, length_includes_head=True,
                    head_width=head_width_c, head_length=head_length_c, width=width_c)
    ax_top_def.arrow(coord_sys_x, coord_sys_y, 0, Lc, color=color_coord, length_includes_head=True,
                    head_width=head_width_c, head_length=head_length_c, width=width_c)
    ax_top_def.text(coord_sys_x + Lc/2, coord_sys_y + 0.2 * disp, 'x', fontsize=fontsize, color=color_coord)
    ax_top_def.text(coord_sys_x + 0.2 * disp, coord_sys_y + Lc/2, 'y', fontsize=fontsize, color=color_coord)

    #ax_top_def.arrow(0, 0, disp, -disp, color=color_twist,
    #                length_includes_head=True, head_width=head_width,
    #                head_length=head_length, width=width_t)

    ax_top_def.set_xlim(xlim[0], xlim[1])
    ax_top_def.set_ylim(ylim[0], ylim[1])

    ### ----- Plot side view ----- ###
    # General
    xlim = (-5 * L_void, Lxy + 3 * L_void)
    ylim = (-2.5 * L_void, Lz + L_void)

    coord_sys_x = -2.5 * disp
    coord_sys_y = -1.5 * disp

    # Undeformed
    void = mpl.patches.Rectangle((-L_void, 0), Lxy+2*L_void, Lz, ec='none', fc=color_void)
    ax_side_undef.add_artist(void)
    geo = mpl.patches.Rectangle((0, 0), Lxy, Lz, ec='none', fc=color_geo)
    ax_side_undef.add_artist(geo)

    ax_side_undef.arrow(coord_sys_x, coord_sys_y, Lc, 0, color=color_coord, length_includes_head=True,
                    head_width=head_width_c, head_length=head_length_c, width=width_c)
    ax_side_undef.arrow(coord_sys_x, coord_sys_y, 0, Lc, color=color_coord, length_includes_head=True,
                    head_width=head_width_c, head_length=head_length_c, width=width_c)
    ax_side_undef.text(coord_sys_x + Lc/2, coord_sys_y + 0.2 * disp, 'x', fontsize=fontsize, color=color_coord)
    ax_side_undef.text(coord_sys_x + 0.2 * disp, coord_sys_y + Lc/2, 'y', fontsize=fontsize, color=color_coord)

    ax_side_undef.set_xlim(xlim[0], xlim[1])
    ax_side_undef.set_ylim(ylim[0], ylim[1])

    # Deformed
    void = mpl.patches.Polygon(((-L_void, 0), (Lxy+L_void, 0),
                                (Lxy+L_void+disp_void, Lz), (-L_void+disp_void, Lz)),
                               closed=True, ec='none', fc=color_void)
    ax_side_def.add_artist(void)
    geo = mpl.patches.Polygon(((0, 0), (Lxy, 0), (Lxy+disp, Lz), (disp, Lz)), closed=True,
                               ec='none', fc=color_geo)
    ax_side_def.add_artist(geo)

    ax_side_def.arrow(coord_sys_x, coord_sys_y, Lc, 0, color=color_coord, length_includes_head=True,
                    head_width=head_width_c, head_length=head_length_c, width=width_c)
    ax_side_def.arrow(coord_sys_x, coord_sys_y, 0, Lc, color=color_coord, length_includes_head=True,
                    head_width=head_width_c, head_length=head_length_c, width=width_c)
    ax_side_def.text(coord_sys_x + Lc/2, coord_sys_y + 0.2 * disp, 'x', fontsize=fontsize, color=color_coord)
    ax_side_def.text(coord_sys_x + 0.2 * disp, coord_sys_y + Lc/2, 'y', fontsize=fontsize, color=color_coord)

    ax_side_def.set_xlim(xlim[0], xlim[1])
    ax_side_def.set_ylim(ylim[0], ylim[1])

    ### -----  Number subplots ------ ###
    ax_3D.text(-Lc * np.sin(angle_c), 1.02 * A5_y, '(a) 3D view',
                    fontsize=fontsize, ha='left', weight='bold')
    ax_top_undef.text(-3 * disp, Lxy + L_void + 1.5 * disp_void, '(b.1) Top: Undeformed',
                      fontsize=fontsize, ha='left', weight='bold')
    ax_top_def.text(-3 * disp, Lxy + L_void + 1.5 * disp_void, '(b.2) Top: Deformed',
                      fontsize=fontsize, ha='left', weight='bold')
    ax_side_undef.text(-1.5 * disp, Lz + L_void, '(c.1) Side: Undeformed',
                       fontsize=fontsize, ha='left', weight='bold')
    ax_side_def.text(-1.5 * disp, Lz + L_void, '(c.1) Side: Deformed',
                       fontsize=fontsize, ha='left', weight='bold')

    ### ----- Save figure ----- ###
    name = 'results_paper/torsion.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def plot_torsion_model_case_1():
    print('Plot definition small strain rotation.')
    ### ----- Parameter definition ----- ###
    # Geometry
    Lz = 0.5
    Lxy = 0.3
    L_void = 0.05

    # Loading
    twist = 0.09

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
    color_coord = 'black'
    color_rot_axis = 'red'

    # Lines
    width_d = 1.5
    width_u = 1.5
    style_u = '-'
    style_u_inv = ''
    style_inv = '--'
    style_rot_axis = '--'
    style_d = '-'
    style_void = ':'

    # Arrows
    head_width = 0.02
    head_length = 1.5 * head_width
    width_t = 0.001
    head_width_c = 0.01
    head_length_c = 1.5 * head_width_c
    width_c = 0.001

    # Fontsize
    fontsize = 12

    ### ----- Coordinates (3D plot) ----- ###
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

    ### ---- Prepare figure ----- ###
    fig = plt.figure(figsize=(7, 4))
    grid = fig.add_gridspec(1, 3)

    ax_3D = fig.add_subplot(grid[0, 0])
    ax_3D.set_aspect('equal')
    ax_3D.axis('off')

    ax_2D_top = fig.add_subplot(grid[0, 1])
    ax_2D_top.set_aspect('equal')
    ax_2D_top.axis('off')

    ax_2D_side = fig.add_subplot(grid[0, 2])
    ax_2D_side.set_aspect('equal')
    ax_2D_side.axis('off')

    ### ----- Plotting 3D small strain rotation ----- ###
    # Plot coordinate system
    ax_3D.arrow(0, 0, Lc * np.cos(angle_c), Lc * np.sin(angle_c),
             color=color_coord, width=width_c,
             length_includes_head=True, head_width = head_width_c,
             head_length=head_length_c)
    ax_3D.arrow(0, 0, 0, Lc, color=color_coord, width=width_c,
             length_includes_head=True, head_width = head_width_c,
             head_length=head_length_c)
    ax_3D.arrow(0, 0, -Lc * np.sin(angle_c), Lc * np.cos(angle_c),
             color=color_coord, width=width_c,
             length_includes_head=True, head_width = head_width_c,
             head_length=head_length_c)
    ax_3D.text(Lc * 0.8, - Lc * 0.25, 'x', color=color_coord, fontsize=fontsize)
    ax_3D.text(-Lc * 0.75, Lc * 0.7, 'y', color=color_coord, fontsize=fontsize)
    ax_3D.text(0, Lc * 1.05, 'z', color=color_coord, fontsize=fontsize)

    # Plot undeformed beam
    ax_3D.plot([B1_x, B2_x], [B1_y, B2_y], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax_3D.plot([B1_x, B4_x], [B1_y, B4_y], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax_3D.plot([U1_x, U2_x], [U1_y, U2_y], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax_3D.plot([U2_x, U3_x], [U2_y, U3_y], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax_3D.plot([U3_x, U4_x], [U3_y, U4_y], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax_3D.plot([U4_x, U1_x], [U4_y, U1_y], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax_3D.plot([B1_x, U1_x], [B1_y, U1_y], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax_3D.plot([B2_x, U2_x], [B2_y, U2_y], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax_3D.plot([B4_x, U4_x], [B4_y, U4_y], color=color_undef, linewidth=width_u, linestyle=style_u)
    ax_3D.plot([B2_x, B3_x], [B2_y, B3_y], color=color_undef, linewidth=width_u, linestyle=style_u_inv)
    ax_3D.plot([B3_x, B4_x], [B3_y, B4_y], color=color_undef, linewidth=width_u, linestyle=style_u_inv)
    ax_3D.plot([B3_x, U3_x], [B3_y, U3_y], color=color_undef, linewidth=width_u, linestyle=style_u_inv)

    # Plot rotation axis
    ax_3D.plot([A_x, A_x], [A1_y, A2_y], color=color_rot_axis)
    ax_3D.plot([A_x, A_x], [A2_y, A4_y], color=color_rot_axis, linestyle=style_inv)
    ax_3D.plot([A_x, A_x], [A4_y, A5_y], color=color_rot_axis)

    ax_3D.arrow(A_x, A1_y, 0, Lc, color=color_rot_axis, length_includes_head=True,
             head_width=head_width, head_length=head_length, width=width_t)
    ax_3D.text(A_x + 0.2 * Lc, A1_y + 0.25 * Lc, '$\hat{n}$', color=color_rot_axis,
            fontsize=fontsize)

    # Plot reference point
    ax_3D.plot([A_x, A_x], [A3_y, A3_y], color=color_rot_axis, marker='o', markersize=5)
    ax_3D.text(A_x + 0.2 * Lc, A3_y - 0.3 * Lc, r'$\vec X_0$', color=color_rot_axis, fontsize=fontsize)

    # Plot deformed beam
    ax_3D.plot([D1_x, D2_x], [D1_y, D2_y], color=color_def, linewidth=width_d)
    ax_3D.plot([D1_x, D4_x], [D1_y, D4_y], color=color_def, linewidth=width_d)
    ax_3D.plot([D2_x, D3_x], [D2_y, D3_y], color=color_def, linewidth=width_d)
    ax_3D.plot([D3_x, D4_x], [D3_y, D4_y], color=color_def, linewidth=width_d)

    ax_3D.plot([B1_x, D1_x], [B1_y, D1_y], color=color_def, linewidth=width_d)
    ax_3D.plot([B2_x, D2_x], [B2_y, D2_y], color=color_def, linewidth=width_d)
    ax_3D.plot([B3_x, D3_x], [B3_y, D3_y], color=color_def, linewidth=width_d, linestyle=style_inv)
    ax_3D.plot([B4_x, D4_x], [B4_y, D4_y], color=color_def, linewidth=width_d)

    ax_3D.plot([B1_x, B2_x], [B1_y, B2_y], color=color_def, linewidth=width_d)
    ax_3D.plot([B1_x, B4_x], [B1_y, B4_y], color=color_def, linewidth=width_d)
    ax_3D.plot([B2_x, B3_x], [B2_y, B3_y], color=color_def, linewidth=width_d, linestyle=style_inv)
    ax_3D.plot([B4_x, B3_x], [B4_y, B3_y], color=color_def, linewidth=width_d, linestyle=style_inv)

    # Plot twist
    ax_3D.arrow(U1_x, U1_y, D1_x - U1_x, D1_y - U1_y, color=color_twist,
             length_includes_head=True, head_width=head_width,
             head_length=head_length, width=width_t)
    ax_3D.arrow(U2_x, U2_y, D2_x - U2_x, D2_y - U2_y, color=color_twist,
             length_includes_head=True, head_width=head_width,
             head_length=head_length, width=width_t)
    ax_3D.arrow(U3_x, U3_y, D3_x - U3_x, D3_y - U3_y, color=color_twist,
             length_includes_head=True, head_width=head_width,
             head_length=head_length, width=width_t)
    ax_3D.arrow(U4_x, U4_y, D4_x - U4_x, D4_y - U4_y, color=color_twist,
             length_includes_head=True, head_width=head_width,
             head_length=head_length, width=width_t)

    ### ----- Plotting 2D small strain rotation (top) ----- ###
    # Coordinate system
    ax_2D_top.arrow(-1.5*disp, -2.5*disp, Lc, 0, color=color_coord, length_includes_head=True,
                    head_width=head_width_c, head_length=head_length_c, width=width_c)
    ax_2D_top.arrow(-1.5*disp, -2.5*disp, 0, Lc, color=color_coord, length_includes_head=True,
                    head_width=head_width_c, head_length=head_length_c, width=width_c)

    ax_2D_top.text(-1.5*disp + Lc/2, -2.3*disp, 'x', fontsize=fontsize, color=color_coord)
    ax_2D_top.text(-1.3*disp, -2.5*disp + Lc/2, 'y', fontsize=fontsize, color=color_coord)

    # Structure
    ax_2D_top.plot((0, Lxy), (0, 0), color=color_undef, linewidth=width_u)
    ax_2D_top.plot((Lxy, Lxy), (0, Lxy), color=color_undef, linewidth=width_u)
    ax_2D_top.plot((0, Lxy), (Lxy, Lxy), color=color_undef, linewidth=width_u)
    ax_2D_top.plot((0, 0), (0, Lxy), color=color_undef, linewidth=width_u)

    ax_2D_top.plot((disp, Lxy+disp), (-disp, disp), color=color_def, linewidth=width_d)
    ax_2D_top.plot((Lxy+disp, Lxy-disp), (disp, Lxy+disp), color=color_def, linewidth=width_d)
    ax_2D_top.plot((-disp, Lxy-disp), (Lxy-disp, Lxy+disp), color=color_def, linewidth=width_d)
    ax_2D_top.plot((disp, -disp), (-disp, Lxy-disp), color=color_def, linewidth=width_d)

    # Displacements
    ax_2D_top.arrow(0, 0, disp, -disp, color=color_twist,
                    length_includes_head=True, head_width=head_width,
                    head_length=head_length, width=width_t)
    ax_2D_top.arrow(Lxy, 0, disp, disp, color=color_twist,
                    length_includes_head=True, head_width=head_width,
                    head_length=head_length, width=width_t)
    ax_2D_top.arrow(Lxy, Lxy, -disp, disp, color=color_twist,
                    length_includes_head=True, head_width=head_width,
                    head_length=head_length, width=width_t)
    ax_2D_top.arrow(0, Lxy, -disp, -disp, color=color_twist,
                    length_includes_head=True, head_width=head_width,
                    head_length=head_length, width=width_t)

    # Void
    ax_2D_top.plot((-L_void, Lxy+L_void), (-L_void, -L_void), linestyle=style_void,
                   color=color_undef, linewidth=width_u)
    ax_2D_top.plot((Lxy+L_void, Lxy+L_void), (-L_void, Lxy+L_void), linestyle=style_void,
                   color=color_undef, linewidth=width_u)
    ax_2D_top.plot((Lxy+L_void, -L_void), (Lxy+L_void, Lxy+L_void), linestyle=style_void,
                   color=color_undef, linewidth=width_u)
    ax_2D_top.plot((-L_void, -L_void), (Lxy+L_void, -L_void), linestyle=style_void,
                   color=color_undef, linewidth=width_u)

    disp_void = disp / (Lxy / 2) * (Lxy / 2 + L_void)
    ax_2D_top.plot((disp_void - L_void, Lxy + L_void + disp_void), (-L_void - disp_void, disp_void - L_void),
                   linestyle=style_void, color=color_def, linewidth=width_d)
    ax_2D_top.plot((Lxy + L_void + disp_void, Lxy + L_void - disp_void), (disp_void - L_void, Lxy + L_void + disp_void),
                   linestyle=style_void, color=color_def, linewidth=width_d)
    ax_2D_top.plot((Lxy + L_void - disp_void, -L_void - disp_void), (Lxy + L_void + disp_void, Lxy + L_void - disp_void),
                   linestyle=style_void, color=color_def, linewidth=width_d)
    ax_2D_top.plot((-L_void - disp_void, disp_void - L_void), (Lxy + L_void - disp_void, -disp_void - L_void),
                   linestyle=style_void, color=color_def, linewidth=width_d)

    ### ----- Plotting 2D small strain rotation (side) ----- ###
    # Structure
    ax_2D_side.plot((0, Lxy), (0, 0), color=color_undef, linewidth=width_u)
    ax_2D_side.plot((Lxy, Lxy), (0, Lz), color=color_undef, linewidth=width_u)
    ax_2D_side.plot((Lxy, 0), (Lz, Lz), color=color_undef, linewidth=width_u)
    ax_2D_side.plot((0, 0), (Lz, 0), color=color_undef, linewidth=width_u)

    ax_2D_side.plot((0, Lxy), (0, 0), color=color_def, linewidth=width_d)
    ax_2D_side.plot((Lxy, Lxy+disp), (0, Lz), color=color_def, linewidth=width_d)
    ax_2D_side.plot((Lxy+disp, disp), (Lz, Lz), color=color_def, linewidth=width_d)
    ax_2D_side.plot((disp, 0), (Lz, 0), color=color_def, linewidth=width_d)

    # Displacements
    ax_2D_side.arrow(0, Lz, disp, 0, color=color_twist,
                    length_includes_head=True, head_width=head_width,
                    head_length=head_length, width=width_t)
    ax_2D_side.arrow(Lxy, Lz, disp, 0, color=color_twist,
                    length_includes_head=True, head_width=head_width,
                    head_length=head_length, width=width_t)

    # Void
    ax_2D_side.plot((-L_void, Lxy+L_void), (0, 0), linestyle=style_void,
                    color=color_undef, linewidth=width_u)
    ax_2D_side.plot((Lxy+L_void, Lxy+L_void), (0, Lz), linestyle=style_void,
                    color=color_undef, linewidth=width_u)
    ax_2D_side.plot((Lxy+L_void, -L_void), (Lz, Lz), linestyle=style_void,
                    color=color_undef, linewidth=width_u)
    ax_2D_side.plot((-L_void, -L_void), (Lz, 0), linestyle=style_void,
                    color=color_undef, linewidth=width_u)

    ax_2D_side.plot((-L_void, Lxy+L_void), (0, 0), linestyle=style_void,
                    color=color_def, linewidth=width_d)
    ax_2D_side.plot((Lxy+L_void, Lxy+L_void+disp_void), (0, Lz), linestyle=style_void,
                    color=color_def, linewidth=width_d)
    ax_2D_side.plot((Lxy+L_void+disp_void, -L_void+disp_void), (Lz, Lz), linestyle=style_void,
                    color=color_def, linewidth=width_d)
    ax_2D_side.plot((-L_void+disp_void, -L_void), (Lz, 0), linestyle=style_void,
                    color=color_def, linewidth=width_d)

    # Coordinate system
    ax_2D_side.arrow(-1.5*disp, -2.5*disp, Lc, 0, color=color_coord, length_includes_head=True,
                    head_width=head_width_c, head_length=head_length_c, width=width_c)
    ax_2D_side.arrow(-1.5*disp, -2.5*disp, 0, Lc, color=color_coord, length_includes_head=True,
                    head_width=head_width_c, head_length=head_length_c, width=width_c)

    ax_2D_side.text(-1.5*disp + Lc/2, -2.3*disp, 'x', fontsize=fontsize, color=color_coord)
    ax_2D_side.text(-1.3*disp, -2.5*disp + Lc/2, 'y', fontsize=fontsize, color=color_coord)


    ### -----  Number subplots ------ ###
    ax_3D.text(-Lc * np.sin(angle_c), 1.02 * A5_y, '(a) 3D view',
                    fontsize=fontsize, ha='left', weight='bold')
    ax_2D_top.text(-1.5 * disp, Lxy + L_void + 1.5 * disp_void, '(b) Top view',
                   fontsize=fontsize, ha='left', weight='bold')
    ax_2D_side.text(-1.5 * disp, Lz + L_void, '(c) Side view',
                   fontsize=fontsize, ha='left', weight='bold')

    ### ----- Save figure ----- ###
    name = 'results_paper/torsion.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def plot_bending_model():
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
    name = f'results_paper/torsion_cylinder/data_3D_Nxy=110.nc'

    nc = netCDF4.Dataset(name, 'r')
    stress_ana = nc.variables['analytical_stress'][0]
    stress_num = nc.variables['stress'][1]
    nc.close()

    # Average over quad pts for all stresses
    stress_ana = np.average(stress_ana, axis=2, weights=[1/3, 1/6, 1/6, 1/6, 1/6])
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
        axes[i].set_xlabel('Position x / $L_z$', fontsize=size_large)
        axes[i].set_ylabel('Position y / $L_z$', fontsize=size_large)
        axes[i].tick_params(labelsize=tick_size)
        axes[i].xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05))
        axes[i].yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05))

    # Mask voxels of vacuum
    mask = geo.cylinder([Nxy, Nxy, Nz], lengths, radius)
    mask = 1 - mask[:, :, z_pt]
    mask[Nxy//2:, :] = 1 # Only plot one half, since the stress is symmetric

    # Plot analytical stress
    helper = np.ma.masked_array(stress_ana[0, 2, :, :, z_pt], mask)
    im = axes[0].pcolormesh(x, y, helper.T, cmap=cmap,
                            vmin=vmin, vmax=vmax, rasterized=True)
    xlim = (x[Nxy//2-10], x[Nxy//2])
    ylim = (y[10], y[15])
    ratio = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    axins_ana = axes[0].inset_axes([0.55, 0.2, 0.4, 0.4 * ratio], xlim=xlim,
                                   ylim=ylim, xticklabels=[], yticklabels=[])
    axins_ana.pcolormesh(x, y, helper.T, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    axes[0].indicate_inset_zoom(axins_ana, edgecolor='black', linewidth=1)
    xlim2 = (x[Nxy//2-10], x[Nxy//2])
    ylim2 = (y[Nxy-15], y[Nxy-10])
    ratio2 = (ylim2[1] - ylim2[0]) / (xlim2[1] - xlim2[0])
    axins_ana2 = axes[0].inset_axes([0.55, 0.8 - 0.4 * ratio2, 0.4, 0.4 * ratio2], xlim=xlim2,
                                   ylim=ylim2, xticklabels=[], yticklabels=[])
    axins_ana2.pcolormesh(x, y, helper.T, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    axes[0].indicate_inset_zoom(axins_ana2, edgecolor='black', linewidth=1)

    # Plot numerical stress
    helper = np.ma.masked_array(stress_num[0, 2, :, :, z_pt], mask)
    im = axes[1].pcolormesh(x, y, helper.T, cmap=cmap,
                            vmin=vmin, vmax=vmax, rasterized=True)
    axins_num = axes[1].inset_axes([0.55, 0.2, 0.4, 0.4 * ratio], xlim=xlim,
                                   ylim=ylim, xticklabels=[], yticklabels=[])
    axins_num.pcolormesh(x, y, helper.T, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    axes[1].indicate_inset_zoom(axins_num, edgecolor='black', linewidth=1)
    axins_num2 = axes[1].inset_axes([0.55, 0.8 - 0.4 * ratio2, 0.4, 0.4 * ratio2], xlim=xlim2,
                                   ylim=ylim2, xticklabels=[], yticklabels=[])
    axins_num2.pcolormesh(x, y, helper.T, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    axes[1].indicate_inset_zoom(axins_num2, edgecolor='black', linewidth=1)

    # Plot colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.75)
    cbar.ax.set_ylabel('Stress $\sigma_{xz}$ / E',
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
def plot_error_stiff():
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
    axes[0].set_xlabel('Number of voxels $n$', fontsize=size_large)
    axes[0].set_ylabel('Relative error (%)', fontsize=size_large)
    axes[0].tick_params(axis='both', which='major', labelsize=tick_size)

    axes[0].plot(Nx_cyl, err_stiff_cyl, marker='s', color=color1, linestyle='--')
    axes[0].plot(Nx_sq, err_stiff_sq, marker='o', color=color2)

    x = 1.2 * Nx_cyl[0]
    y = 0.8 * err_stiff_cyl[0]
    axes[0].text(x, y, 'circular cross-section', color=color1, fontsize=size_large)

    x = 1.1 * Nx_sq[0]
    y = -0.55 * err_stiff_sq[0]
    axes[0].text(x, y, 'square cross-section', color=color2, fontsize=size_large, ha='left')

    # Plot bending stiffness
    axes[1].set_xlabel('Number of voxels $n$', fontsize=size_large)
    axes[1].tick_params(axis='both', which='major', labelsize=tick_size)
    axes[1].plot(N_bend, err_stiff_bend, marker='o', color=color2)

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

    color_para = 'black'
    linewidth_para = 1.5
    fontsize_para = 16
    weight_para = 'normal'
    style_para = 'italic'

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
    ax.plot([122 + 17, 290 + 17], [280 + 22, 238 + 22], linewidth=scale_width, color=scale_color)
    ax.text(61 + 145 + 15 + 20, 140 + 119 + 20 + 20, '0.5mm', fontsize=fontsize, color=scale_color,
            ha='center', va='center', rotation=15)
    ax_left = ax

    ax = fig.add_subplot(gs_left[0, 1])
    ax.axis('off')
    implot = ax.imshow(im_muspectre)
    ax.text(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) / 2,
            0, '(b) Voxelized unit cell', color='black', fontsize=fontsize, weight='bold', ha='center')
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
    ax.text(70, 290, r'$\alpha$', fontsize=fontsize_para, color=color_para)
    ax.plot((90, 140), (270, 240), linewidth=1.25, color=color_para)

    # Plot radius
    Rx = 208
    Ry = 163
    R = 76
    ax.arrow(Rx, Ry, R / 2**0.5, -R / 2**0.5 - 3, length_includes_head=True, color=color_para,
             linewidth=linewidth_para, head_length=10, head_width=5)
    ax.text(Rx + R / 2 / 2**0.5, Ry - R / 2 / 2**0.5, r'$r$', color=color_para, va='top',
            fontsize=fontsize_para)

    # Plot beam thickness
    ax.arrow(240, 37, 0, 20, length_includes_head=True, color=color_para,
             linewidth=linewidth_para, head_length=10, head_width=5)
    ax.arrow(240, 88, 0, -20, length_includes_head=True, color=color_para,
             linewidth=linewidth_para, head_length=10, head_width=5)
    ax.text(243, 43, r'$t$', color=color_para, fontsize=fontsize_para)

    ax.arrow(235, 185, 20/2**0.5, 20/2**0.5, length_includes_head=True, color=color_para,
             linewidth=linewidth_para, head_length=10, head_width=5)
    ax.arrow(270, 220, -20/2**0.5, -20/2**0.5, length_includes_head=True, color=color_para,
             linewidth=linewidth_para, head_length=10, head_width=5)
    ax.text(240, 185, r'$t$', color=color_para, fontsize=fontsize_para)

    # Plot side length
    ax.arrow(120, 279, 170, -42, length_includes_head=True, color=color_para,
             linewidth=linewidth_para, head_length=10, head_width=5)
    ax.text(205, 277, r'$a$', color=color_para, fontsize=fontsize_para)
    ax.arrow(25, 209, 0, -180, length_includes_head=True, color=color_para,
             linewidth=linewidth_para, head_length=10, head_width=5)
    ax.text(0, 130, r'$a$', color=color_para, fontsize=fontsize_para)
    ax.arrow(120, 279, -95, -70, length_includes_head=True, color=color_para,
             linewidth=linewidth_para, head_length=10, head_width=5)
    ax.text(55, 265, r'$a$', color=color_para, fontsize=fontsize_para)

    # Plot coordinate system
    Lc = 70
    color_coord = 'black'
    width_c = 2
    head_width_c = 12
    head_length_c = 17
    coord_y = 320
    helper = np.arctan(42 / 170)
    ax.arrow(0, coord_y, Lc * np.cos(helper), -Lc * np.sin(helper),
             color=color_coord, width=width_c,
             length_includes_head=True, head_width = head_width_c,
             head_length=head_length_c)
    ax.arrow(0, coord_y, 0, -1.25 * Lc, color=color_coord, width=width_c,
             length_includes_head=True, head_width = head_width_c,
             head_length=head_length_c)
    helper = np.arctan(70 / 95)
    ax.arrow(0, coord_y, -Lc * np.sin(helper), -Lc * np.cos(helper),
             color=color_coord, width=width_c,
             length_includes_head=True, head_width = head_width_c,
             head_length=head_length_c)
    ax.text(30, coord_y + 20, 'x', color=color_coord, fontsize=fontsize)
    ax.text(-40, coord_y, 'y', color=color_coord, fontsize=fontsize)
    ax.text(-27, coord_y - Lc, 'z', color=color_coord, fontsize=fontsize)


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
    figsize = (8, 3)
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
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'bottom':0.2, 'wspace':0.4, 'top':0.8})

    # Plot Comsol convergence
    def format_ticks_a(x, pos):
        x = x / 1e6
        return f'${x:.0f}$' + r'$\times$' + f'$10^6$' 

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

    # Number subplots
    axes[0].text(-0.8e6, 0.01184, '(a) COMSOL', fontsize=fontsize, va='bottom', ha='left', weight='bold')
    axes[1].text(0, 0.00776, '(b) $\mathbf{\mu}$Spectre', fontsize=fontsize, ha='left', weight='bold')

    ### ----- Finish ----- ###
    name = 'results_paper/chiral_mesh_refinement.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.show()
    plt.close(fig)

##################################################################
### ------------- Plot force vs. nb_unit_cells_z ------------- ###
##################################################################
def plot_force_vs_parameters():
    print('Plot force vs. parameters.')
    ### ----- Parameters ------ ###
    # Data
    folder = 'results_paper/'
    nb_unit_cell_xy_list = [1, 3]
    nb_unit_cell_z_list = [1, 2, 3, 4]
    max_el_size = ['0.5e-5', '0.2e-5']
    name_muspectre = 'results_paper/chiral_muspectre/chiral_comp_cylinder/data.txt'
    name_comsol_1x1x4 = 'results_paper/chiral_comsol/chiral_comsol_1x1x4_unit_cells/influence_twist_overview.txt'
    name_comsol_3x3x4 = 'results_paper/chiral_comsol/chiral_comsol_3x3x4_unit_cells/influence_twist_overview.txt'

    # Plotting (both)
    figsize = (9, 4)
    fontsize = 14
    tick_size = 12

    # Plotting (vs. twist)
    color_cyl = 'red'
    color_chi_1 = 'blue'
    color_chi_2 = 'orange'
    style_com = '--'
    marker_cyl = 's'
    marker_chi_1 = 'o'
    marker_chi_2 = 'v'

    # Plotting (vs. nb_unit_cell_z)
    color_1x1 = 'blue'
    color_3x3 = 'red'
    linestyle_mu = '-'
    linestyle_com = '--'
    marker_1x1 = 's'
    marker_3x3 = 'o'

    ### ----- Read data ----- ###
    # Comsol (vs. nb_unit_cell_z)
    forces_comsol = np.empty((len(nb_unit_cell_xy_list), len(nb_unit_cell_z_list)))
    for index_xy in range(len(nb_unit_cell_xy_list)):
        N_xy = nb_unit_cell_xy_list[index_xy]
        name = folder + f'chiral_comsol/chiral_comsol_{N_xy}x{N_xy}x{1}_unit_cells/'
        name += f'average_force_per_rot_max_el=' + max_el_size[index_xy] + '.txt'
        data = np.loadtxt(name, skiprows=5)
        for index_z in range(0, len(nb_unit_cell_z_list)):
            N_z = nb_unit_cell_z_list[index_z]
            name = folder + f'chiral_comsol/chiral_comsol_{N_xy}x{N_xy}x{N_z}_unit_cells/'
            name += 'average_force_per_rot.txt'
            data = np.loadtxt(name, skiprows=5)
            forces_comsol[index_xy, index_z] = data[1]

    # muSpectre (vs. nb_unit_cell_z)
    N_xy = nb_unit_cell_xy_list[0]
    name = folder + f'chiral_muspectre/chiral_Nuc={N_xy}x{N_xy}xNucz/data.txt'
    data = np.loadtxt(name, skiprows=1)
    N_z_1 = data[:, 0].astype(int)
    force_muspectre_1 = data[:, 4]

    N_xy = nb_unit_cell_xy_list[1]
    name = folder + f'chiral_muspectre/chiral_Nuc={N_xy}x{N_xy}xNucz/data.txt'
    data = np.loadtxt(name, skiprows=1)
    N_z_2 = data[:, 0].astype(int)
    force_muspectre_2 = data[:, 4]

    # Read muSpectre data (vs. twist)
    twists = np.loadtxt(name_muspectre, skiprows=1, max_rows=1)
    force_cyl = np.loadtxt(name_muspectre, skiprows=5, max_rows=1)
    force_chi_1x1x1 = np.loadtxt(name_muspectre, skiprows=9, max_rows=1)
    force_chi_3x3x1 = np.loadtxt(name_muspectre, skiprows=13, max_rows=1)

    twists = twists * 1000 # twists in 1/m

    # Read comsol data (vs. twist)
    data_com_1x1x4 = np.loadtxt(name_comsol_1x1x4, skiprows=1)
    data_com_3x3x4 = np.loadtxt(name_comsol_3x3x4, skiprows=1)

    case = 2

    ### ----- Plotting ----- ###
    # Prepare figure
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'bottom':0.15, 'wspace':0.4, 'top':0.9})

    # Plot force vs. nb_unit_cells_z
    ax = axes[0]
    ax.set_xlabel('Number of unit cells in z-direction $N_z$', fontsize=fontsize)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
    ax.tick_params(labelsize=tick_size)
    #ax.set_ylabel('Force in z-direction (N)', fontsize=fontsize)
    #ax.plot(nb_unit_cell_z_list, forces_comsol[0], marker=marker_1x1, color=color_1x1,
    #        linestyle=linestyle_com, label=f'Comsol N_xy={1}')
    #ax.plot(nb_unit_cell_z_list, forces_comsol[1], marker=marker_3x3, color=color_3x3,
    #        linestyle=linestyle_com, label=f'Comsol N_xy={3}')
    #ax.plot(N_z_1, force_muspectre_1, marker=marker_1x1, color=color_1x1,
    #        linestyle=linestyle_mu, label=f'muSpectre N_xy={nb_unit_cell_xy_list[0]}')
    #ax.plot(N_z_2, force_muspectre_2, marker=marker_3x3, color=color_3x3,
    #        linestyle=linestyle_mu, label=f'muSpectre N_xy={nb_unit_cell_xy_list[1]}')
    #ax.set_ylim(0, None)
    ax.set_ylabel('Force in z-direction (mN)', fontsize=fontsize)
    ax.plot(nb_unit_cell_z_list, forces_comsol[0]*1000, marker=marker_1x1, color=color_1x1,
            linestyle=linestyle_com, label=f'Comsol N_xy={1}')
    ax.plot(nb_unit_cell_z_list, forces_comsol[1]*1000, marker=marker_3x3, color=color_3x3,
            linestyle=linestyle_com, label=f'Comsol N_xy={3}')
    ax.plot(N_z_1, force_muspectre_1*1000, marker=marker_1x1, color=color_1x1,
            linestyle=linestyle_mu, label=f'muSpectre N_xy={nb_unit_cell_xy_list[0]}')
    ax.plot(N_z_2, force_muspectre_2*1000, marker=marker_3x3, color=color_3x3,
            linestyle=linestyle_mu, label=f'muSpectre N_xy={nb_unit_cell_xy_list[1]}')
    ax.set_ylim(0, None)

    # Labels
    #ax.text(1.82, 1e-2, 'COMSOL: $N_{x}$=$N_{y}$=1', fontsize=fontsize, color=color_1x1)
    #ax.text(1.5, 4.5e-3, 'COMSOL: $N_{x}$=$N_{y}$=3', fontsize=fontsize, color=color_3x3)
    #ax.text(1.05, 7.6e-3, r'$\mathrm{\mu}$Spectre: $N_{x}$=$N_{y}$=1', fontsize=fontsize, color=color_1x1)
    #ax.text(1.05, 2.5e-3, r'$\mathrm{\mu}$Spectre: $N_{x}$=$N_{y}$=3', fontsize=fontsize, color=color_3x3)
    ax.text(1.82, 10, 'COMSOL: $N_{x}$=$N_{y}$=1', fontsize=fontsize, color=color_1x1)
    ax.text(1.5, 4.5, 'COMSOL: $N_{x}$=$N_{y}$=3', fontsize=fontsize, color=color_3x3)
    ax.text(1.05, 7.6, r'$\mathrm{\mu}$Spectre: $N_{x}$=$N_{y}$=1', fontsize=fontsize, color=color_1x1)
    ax.text(1.05, 2.5, r'$\mathrm{\mu}$Spectre: $N_{x}$=$N_{y}$=3', fontsize=fontsize, color=color_3x3)

    # Plot force vs twist
    ax = axes[1]
    ax.set_xlabel('Twist angle per length $w$ (1/m)', fontsize=fontsize)
    ax.set_ylabel('Force in z-direction (N)', fontsize=fontsize)
    ax.tick_params(labelsize=tick_size)

    if case == 1:
        line_mu_cyl, = ax.plot(twists, force_cyl, marker=marker_cyl, color=color_cyl, label=r'$\mathrm{\mu}$Spectre: Cylinder')
        line_mu_1, = ax.plot(twists, force_chi_1x1x1, marker=marker_chi_1, color=color_chi_1, label=r'$\mathrm{\mu}$Spectre: $N_x=N_y=1$')
        line_mu_3, = ax.plot(twists, force_chi_3x3x1, marker=marker_chi_2, color=color_chi_2, label=r'$\mathrm{\mu}$Spectre: $N_x=N_y=3$')

        line_com_1, = ax.plot(data_com_1x1x4[:, 0], data_com_1x1x4[:, 2], linestyle=style_com, marker=marker_chi_1,
                              color=color_chi_1, label='COMSOL: $N_x=N_y=1$')
        line_com_3, = ax.plot(data_com_3x3x4[:, 0], data_com_3x3x4[:, 2], linestyle=style_com, marker=marker_chi_2,
                              color=color_chi_2, label='COMSOL: $N_x=N_y=3$')
    if case == 2:
        line_mu_cyl, = ax.plot(twists, force_cyl, marker=marker_cyl, color=color_cyl, label='Cylinder')
        line_mu_1, = ax.plot(twists, force_chi_1x1x1, marker=marker_chi_1, color=color_chi_1, label='$N_x=N_y=1$')
        line_mu_3, = ax.plot(twists, force_chi_3x3x1, marker=marker_chi_2, color=color_chi_2, label='$N_x=N_y=3$')

        line_com_1, = ax.plot(data_com_1x1x4[:, 0], data_com_1x1x4[:, 2], linestyle=style_com, marker=marker_chi_1,
                              color=color_chi_1, label='$N_x=N_y=1$')
        line_com_3, = ax.plot(data_com_3x3x4[:, 0], data_com_3x3x4[:, 2], linestyle=style_com, marker=marker_chi_2,
                              color=color_chi_2, label='$N_x=N_y=3$')

    # Legends
    #ax.legend(fontsize=tick_size, frameon=False, labelspacing=0.05, borderpad=0, handlelength=1.5, handletextpad=0.2)
    if case == 1:
        first_legend = ax.legend(handles=[line_mu_cyl, line_mu_1, line_mu_3], loc='upper left', fontsize=tick_size,
                                 labelspacing=0.1, frameon=False, borderpad=0, handletextpad=0.2)
        ax.add_artist(first_legend)
        ax.legend(handles=[line_com_1, line_com_3], loc='lower right', fontsize=tick_size, labelspacing=0.1,
                  frameon=False, borderpad=0, handletextpad=0.2)
    if case == 2:
        first_legend = ax.legend(handles=[line_mu_cyl, line_mu_1, line_mu_3], loc='upper left',
                                 fontsize=tick_size, title=r'$\mathrm{\mu}$Spectre', title_fontsize=fontsize,
                                 labelspacing=0.1)
        ax.add_artist(first_legend)
        ax.legend(handles=[line_com_1, line_com_3], loc='lower right', fontsize=tick_size, labelspacing=0.1,
                  title=r'COMSOL', title_fontsize=fontsize)

    # Number subplots
    #axes[0].text(0.25, 0.012, '(a) Number of unit cells in z-direction', fontsize=fontsize, weight='bold')
    axes[0].text(0.25, 12, '(a) Number of unit cells in z-direction', fontsize=fontsize, weight='bold')
    axes[1].text(-150, 0.0205, '(b) Twist angle per length', fontsize=fontsize, weight='bold')

    # Print out stiffnesses
    print('Stiffness = Average force / twist angle')
    print(f'muSpectre 1x1x1uc: {force_chi_1x1x1[-1] / twists[-1]}')
    print(f'muSpectre 3x3x1uc: {force_chi_3x3x1[-1] / twists[-1]}')
    print(f'Comsol 1x1x4uc: {data_com_1x1x4[-1, 2] / data_com_1x1x4[-1, 0]}')
    print(f'Comsol 3x3x4uc: {data_com_3x3x4[-1, 2] / data_com_3x3x4[-1, 0]}')

    ### ----- Finish ----- ###
    name = 'results_paper/chiral_average_force_z.pdf'
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
            ax.get_ylim()[1] + 0.035 * ax.get_ylim()[1],
            r'(a) $\mathbf{\mu}$Spectre', fontsize=fontsize, weight='bold', ha='left')
    ax1.text(ax1.get_xlim()[0],
             ax1.get_ylim()[1],
             '(b) COMSOL', fontsize=fontsize, weight='bold', ha='left')


    ### ----- Finish ----- ###
    name = 'results_paper/stress_zz_chiral.png'
    fig.savefig(name, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close(fig)

##################################################################
### ------------------ Make plots for paper ------------------ ###
##################################################################
def plots_for_paper():
    # Plots
    #plot_torsion_model()
    #plot_error_stiff()
    #plot_2D_stress_cylinder()
    #plot_geometry_chiral_metamat()
    #plot_results_mesh_refinement()
    #plot_force_vs_parameters()
    #plot_force_vs_unit_cells()
    plot_2D_stress_chiral()
    #plot_force_vs_twist()


if __name__ == "__main__":
    plots_for_paper()

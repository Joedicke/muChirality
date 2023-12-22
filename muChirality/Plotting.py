"""
@file   Plotting.py

@author Indre  Joedicke <indre.joedicke@imtek.uni-freiburg.de>

@date   18 Jul 2023

@brief  Helper functions for plotting 3D-geometries.

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

# Default path of the library
sys.path.insert(0, os.path.join(os.getcwd(), "./muspectre/build/language_bindings/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "./muspectre/build/language_bindings/libmufft/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "./muspectre/build/language_bindings/libmugrid/python"))

import numpy as np
import matplotlib.pyplot as plt

import muSpectre as µ

import muChirality.Geometries as geo

def plot_2D_projection(ax, mask, lengths, plane = 2):
    """
    Create a 2D plot of a 3D geometry by projecting the geometry to one plane.
    Input
    -----
    ax: matplotlib.pyplot axis object
        Axis where the plot will be made
    mask: np.array((Nx, Ny, Nz)) of booleans
          Mask which pixels have material.
    lengths: list of 3 floats
             Length of the system in each cartesian direction.
    plane: int
           Which plane is plotted. Must be:
           0: Plot 12 - plane
           1: Plot 01 - plane
           2: Plot 01 - plane. Default.
    """
    # Nb_grid_pts
    nb_grid_pts = mask.shape
    if len(nb_grid_pts) != 3:
        message = f'The mask has shape {mask.shape}, but '
        message += 'it must have 3 dimensions.'
        raise ValueError(message)

    # Label axes
    ax.set_aspect('equal')
    if plane == 0:
        ax.set_xlabel('y')
        ax.set_ylabel('z')
    elif plane == 1:
        ax.set_xlabel('x')
        ax.set_ylabel('z')
    elif plane == 2:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    else:
        message = 'The parameter plane must be 0, 1 or 2.'
        raise ValueError(message)

    # Coordinates of grid pts
    if plane == 0:
        x_plot = np.linspace(0, lengths[1], nb_grid_pts[1]+1)
        y_plot = np.linspace(0, lengths[2], nb_grid_pts[2]+1)
    elif plane == 1:
        x_plot = np.linspace(0, lengths[0], nb_grid_pts[0]+1)
        y_plot = np.linspace(0, lengths[2], nb_grid_pts[2]+1)
    elif plane == 2:
        x_plot = np.linspace(0, lengths[0], nb_grid_pts[0]+1)
        y_plot = np.linspace(0, lengths[1], nb_grid_pts[1]+1)
    else:
        message = 'The parameter plane must be 0, 1 or 2.'
        raise ValueError(message)

    # Projection
    values = np.any(mask, axis=plane)

    # Plot projection
    tpc = ax.pcolormesh(x_plot, y_plot, values.T)

def plot_2D_cut(ax, mask, lengths, index = 0, plane = 2):
    """
    Create a 2D plot of a 3D geometry by cutting the geometry at one plane.
    Input
    -----
    ax: matplotlib.pyplot axis object
        Axis where the plot will be made
    mask: np.array((Nx, Ny, Nz)) of booleans
          Mask which pixels have material.
    lengths: list of 3 floats
             Length of the system in each cartesian direction.
    index: int
           index of the plane which shall be plotted.
    plane: int
           Which plane is plotted. Must be:
           0: Plot 12 - plane
           1: Plot 01 - plane
           2: Plot 01 - plane. Default.
    """
    # Discretization
    nb_grid_pts = mask.shape
    if len(nb_grid_pts) != 3:
        message = f'The mask has shape {mask.shape}, but '
        message += 'it must have 3 dimensions.'
        raise ValueError(message)
    if index >= nb_grid_pts[plane]:
        message = f'The mask has only {nb_grid_pts[plane]} entries '
        message += f'for plane = {plane}, but the index is {index}.'
        raise ValueError(message)

    # Label axes
    ax.set_aspect('equal')
    if plane == 0:
        ax.set_xlabel('y')
        ax.set_ylabel('z')
    elif plane == 1:
        ax.set_xlabel('x')
        ax.set_ylabel('z')
    elif plane == 2:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    else:
        message = 'The parameter plane must be 0, 1 or 2.'
        raise ValueError(message)

    # Coordinates of grid pts
    if plane == 0:
        x_plot = np.linspace(0, lengths[1], nb_grid_pts[1]+1)
        y_plot = np.linspace(0, lengths[2], nb_grid_pts[2]+1)
    elif plane == 1:
        x_plot = np.linspace(0, lengths[0], nb_grid_pts[0]+1)
        y_plot = np.linspace(0, lengths[2], nb_grid_pts[2]+1)
    elif plane == 2:
        x_plot = np.linspace(0, lengths[0], nb_grid_pts[0]+1)
        y_plot = np.linspace(0, lengths[1], nb_grid_pts[1]+1)
    else:
        message = 'The parameter plane must be 0, 1 or 2.'
        raise ValueError(message)

    # Cut
    if plane == 0:
        values = mask[index, :, :]
    elif plane == 1:
        values = mask[:, index, :]
    elif plane == 2:
        values = mask[:, :, index]
    else:
        message = 'The parameter plane must be 0, 1 or 2.'
        raise ValueError(message)

    # Plotting
    tpc = ax.pcolormesh(x_plot, y_plot, values.T)

def plot_2D_chiral(nb_grid_pts, lengths, radius, thickness, alpha):
    """
    Create 2D plots of the first chiral metamaterial. Helper lines show
    the non-discretized structure.

    Arguments
    ---------
    nb_grid_pts: list of 3 ints
                 Number of grid pts in each direction.
    lengths: list of 3 floats
             Lengths of unit cell in each direction.
    radius: float
            Radius of the cirlces
    thickness: float
               Thickness of the cirles and beams.
    alpha: float
           Angle at wich the connecting beams are inclined.
           Default is 0, meaning the beams are vertical.
    Returns
    -------
    fig1: matplotlib.pyplot.figure object
          figure showing the projection on the xz-plane
    fig2: matplotlib.pyplot.figure object
          figure showing the projection on the yz-plane
    fig3: matplotlib.pyplot.figure object
          figure showing the projection on the xy-plane
    """

    ### ----- Parameter definitions ----- ###
    Lx = lengths[0]
    Ly = lengths[1]
    Lz = lengths[2]

    # Plot thick lines
    lw = 3

    # Center of circles
    x_axis = nb_grid_pts[0] // 2 * (lengths[0] / nb_grid_pts[0])
    y_axis = nb_grid_pts[1] // 2 * (lengths[1] / nb_grid_pts[1])

    # Analytical construction of metamaterial
    R = radius - thickness / 2
    b = (Lz - 2 * thickness) * np.tan(alpha)
    c = R - (R**2 - b**2) ** 0.5
    c2 = R + (R**2 - b**2) ** 0.5

    # Material
    mask = geo.chiral_metamaterial(nb_grid_pts, lengths, radius, thickness,
                                   alpha=alpha)

    ### ----- x-z-plot ----- ###
    # Plot projection of metamaterial
    fig1, ax = plt.subplots()
    plot_2D_projection(ax, mask, lengths, plane = 1)

    # Plot projection of circles
    ax.plot([Lx/2 - radius, Lx/2 + radius],
            [0, 0], color='red', linewidth=lw)
    ax.plot([Lx/2 - radius, Lx/2 + radius],
            [thickness, thickness], color='red', linewidth=lw)
    ax.plot([Lx/2 - radius, Lx/2 + radius],
            [Lz, Lz], color='red', linewidth=lw)
    ax.plot([Lx/2 - radius, Lx/2 + radius],
            [Lz-thickness, Lz-thickness],
            color='red', linewidth=lw)

    # Plot projection of one beam in bold
    ax.plot([Lx/2, Lx/2 + b], [thickness, Lz-thickness],
            color='black', linewidth=lw, linestyle='--')
    ax.plot([Lx/2-thickness/2, Lx/2-thickness/2+b], [thickness, Lz-thickness],
            color='red', linewidth=lw)
    ax.plot([Lx/2+thickness/2, Lx/2+thickness/2+b], [thickness, Lz-thickness],
            color='red', linewidth=lw)

    # Plot projection of other beams
    ax.plot([Lx/2 + radius - thickness, Lx/2 + radius - thickness - c],
            [thickness, Lz - thickness], color='red')
    ax.plot([Lx/2 + radius, Lx/2 + radius - c],
            [thickness, Lz - thickness], color='red')
    ax.plot([Lx/2 + thickness/2, Lx/2 + thickness/2 - b],
            [thickness, Lz - thickness], color='red')
    ax.plot([Lx/2 - thickness/2, Lx/2 - thickness/2 - b],
            [thickness, Lz - thickness], color='red')
    ax.plot([Lx/2 - radius + thickness, Lx/2 - radius + thickness + c],
            [thickness, Lz - thickness], color='red')
    ax.plot([Lx/2 - radius, Lx/2 - radius + c],
            [thickness, Lz - thickness], color='red')

    ### ----- y-z-plot ----- ###
    # Plot projection of metamaterial
    fig2, ax = plt.subplots()
    plot_2D_projection(ax, mask, lengths, plane = 0)

    # Plot projection of circles
    ax.plot([Ly/2 - radius, Ly/2 + radius],
            [0, 0], color='red', linewidth=lw)
    ax.plot([Ly/2 - radius, Ly/2 + radius],
            [thickness, thickness], color='red', linewidth=lw)
    ax.plot([Ly/2 - radius, Ly/2 + radius],
            [Lz, Lz], color='red', linewidth=lw)
    ax.plot([Ly/2 - radius, Ly/2 + radius],
            [Lz-thickness, Lz-thickness],
            color='red', linewidth=lw)

    # Plot projection of one beam in bold
    ax.plot([Ly/2-radius+thickness/2, Ly/2 - radius + thickness/2 + c],
            [thickness, Lz - thickness],
            color='black', linewidth=lw, linestyle='--')
    ax.plot([Ly/2 - radius, Ly/2 - radius + c],
            [thickness, Lz - thickness],
            color='red', linewidth=lw)
    ax.plot([Ly/2 - radius + thickness, Ly/2 - radius + thickness + c],
            [thickness, Lz - thickness],
            color='red', linewidth=lw)

    # Plot projection of other beams
    ax.plot([Ly/2 - thickness/2, Ly/2 - thickness/2 + b],
            [thickness, Lz - thickness], color='red')
    ax.plot([Ly/2 + thickness/2, Ly/2 + thickness/2 + b],
            [thickness, Lz - thickness], color='red')
    ax.plot([Ly/2 + radius - thickness, Ly/2 + radius - thickness - c],
            [thickness, Lz - thickness], color='red')
    ax.plot([Ly/2 + radius, Ly/2 + radius - c],
            [thickness, Lz - thickness], color='red')
    ax.plot([Ly/2 + thickness/2, Ly/2 + thickness/2 - b],
            [thickness, Lz - thickness], color='red')
    ax.plot([Ly/2 - thickness/2, Ly/2 - thickness/2 - b],
            [thickness, Lz - thickness], color='red')

    ### ----- x-y-plot ----- ###
    # Plot projection of metamaterial
    fig3, ax = plt.subplots()
    plot_2D_projection(ax, mask, lengths, plane = 2)

    # Plot projection of circles
    circle1 = plt.Circle((x_axis, y_axis), radius, color='red', fill=False,
                         linewidth=lw)
    circle2 = plt.Circle((x_axis, y_axis), radius-thickness, color='red',
                         fill=False, linewidth=lw)
    ax.add_patch(circle1)
    ax.add_patch(circle2)

    # Plot projection of one beam in bold
    ax.plot([Lx/2, Lx/2 + b], [Ly/2 - radius + thickness/2,
                               Ly/2 - radius + thickness/2 + c],
            color='black', linestyle='--', linewidth=lw)
    ax.plot([Lx/2 - thickness/2, Lx/2 + thickness/2],
            [Ly/2 - radius, Ly/2 - radius],
            color='red', linewidth=lw)
    ax.plot([Lx/2 + thickness/2, Lx/2 + thickness/2],
            [Ly/2 - radius, Ly/2 - radius + thickness],
            color='red', linewidth=lw)
    ax.plot([Lx/2 + thickness/2, Lx/2 - thickness/2],
            [Ly/2 - radius + thickness, Ly/2 - radius + thickness],
            color='red', linewidth=lw)
    ax.plot([Lx/2 - thickness/2, Lx/2 - thickness/2],
            [Ly/2 - radius + thickness, Ly/2 - radius],
            color='red', linewidth=lw)
    ax.plot([Lx/2 - thickness/2 + b, Lx/2 + thickness/2 + b],
            [Ly/2 - radius + c, Ly/2 - radius + c],
            color='red', linewidth=lw)
    ax.plot([Lx/2 + thickness/2 + b, Lx/2 + thickness/2 + b],
            [Ly/2 - radius + c, Ly/2 - radius + thickness + c],
            color='red', linewidth=lw)
    ax.plot([Lx/2 + thickness/2 + b, Lx/2 - thickness/2 + b],
            [Ly/2 - radius + thickness + c, Ly/2 - radius + thickness + c],
            color='red', linewidth=lw)
    ax.plot([Lx/2 - thickness/2 + b, Lx/2 - thickness/2 + b],
            [Ly/2 - radius + thickness + c, Ly/2 - radius + c],
            color='red', linewidth=lw)

    # Plot projections of one corner of the four beams
    ax.plot([Lx/2 - thickness/2, Lx/2 - thickness/2 + b],
            [Ly/2 - radius + thickness, Ly/2 - radius + thickness + c],
            color='red')
    ax.plot([Lx/2 + radius - thickness, Lx/2 + radius - thickness - c],
            [Ly/2 - thickness/2, Ly/2 - thickness/2 + b], color='red')
    ax.plot([Lx/2 + thickness/2, Lx/2 + thickness/2 - b],
            [Ly/2 + radius - thickness, Ly/2 + radius - thickness - c],
            color='red')
    ax.plot([Lx/2 - radius + thickness, Lx/2 - radius + thickness + c],
            [Ly/2 + thickness/2, Ly/2 + thickness/2 - b], color='red')

    return fig1, fig2, fig3

def plot_2D_chiral_2(nb_grid_pts, lengths, radius_out, radius_inn,
                     thickness, alpha):
    """
    Create 2D plots of the second chiral metamaterial. Helper lines show
    the non-discretized structure.

    Arguments
    ---------
    nb_grid_pts: list of 3 ints
                 Number of grid pts in each direction.
    lengths: list of 3 floats
             Lengths of unit cell in each direction. Note that the size of
             the RVE corresponds to lengths[2], so that lengths[0] and
             lengths[1] must be larger than lengths[2] to break the periodicity.
    radius_out: float
                Outer radius of the circles.
    radius_inn: float
                Inner radius of the circles.
    thickness: float
               Thickness of the connecting beams.
    alpha: float
           Angle at wich the connecting beams are inclined.
           Default is 0.
    Returns
    -------
    fig1: matplotlib.pyplot.figure object
          figure showing one cut in a xz-plane
    fig2: matplotlib.pyplot.figure object
          figure showing one cut in a yz-plane
    fig3: matplotlib.pyplot.figure object
          figure showing one cut in a xy-plane
    """

    ### ----- Parameter definitions ----- ###
    Lx = lengths[0]
    Ly = lengths[1]
    Lz = lengths[2]

    # Coordinates of grid pts
    x = np.linspace(0, lengths[0], nb_grid_pts[0]+1)
    y = np.linspace(0, lengths[1], nb_grid_pts[1]+1)
    z = np.linspace(0, lengths[2], nb_grid_pts[2]+1)

    # Plot thick lines
    lw = 3

    # Analytical construction of metamaterial
    a = Lz
    b = 1.5 * thickness
    beta = np.pi / 4 - alpha
    helper = 2 ** 0.5 * (a/2 - b/2) * np.cos(alpha)
    rect_triangle_short = helper * np.sin(beta)
    rect_triangle_long = helper * np.cos(beta)
    helper_a = 1 + np.tan(beta) ** 2
    helper_b = - 2 * (np.tan(beta) + 1) * (a/2 - b/2)
    helper_c = 2 * (a/2 - b/2) ** 2 - (radius_out/2 + radius_inn/2) ** 2
    helper = helper_b ** 2 - 4 * helper_a * helper_c
    triangle_long = (- helper_b - helper ** 0.5 ) / 2 / helper_a
    triangle_short = np.tan(beta) * triangle_long

    # Material
    mask = geo.chiral_metamaterial_2(nb_grid_pts, lengths, radius_out,
                                     radius_inn, thickness, alpha=alpha)

    ### ----- x-z-plot (at specific y) ----- ###
    ind_y = round((lengths[1]/2 - a/2) / lengths[1] * nb_grid_pts[1])

    # Plot geometry
    fig1, ax = plt.subplots()
    fig1.suptitle(f'y = {ind_y * lengths[1] / nb_grid_pts[1]:.2}')
    plot_2D_cut(ax, mask, lengths, index = ind_y, plane = 1)

    # Plot boundaries of RVE
    ax.plot([Lx/2 - a/2, Lx/2 + a/2],
            [0, 0], color='orange', linewidth=lw)
    ax.plot([Lx/2 - a/2, Lx/2 + a/2],
            [Lz, Lz], color='orange', linewidth=lw)
    ax.plot([Lx/2 - a/2, Lx/2 - a/2],
            [0, Lz], color='orange', linewidth=lw)
    ax.plot([Lx/2 + a/2, Lx/2 + a/2],
            [0, Lz], color='orange', linewidth=lw)

    # Plot projection of circles in x-z-plane
    x0 = lengths[0] / 2
    z0 = lengths[2] / 2
    circle1 = plt.Circle((x0, z0), radius_out, color='red', fill=False,
                         linewidth=lw)
    circle2 = plt.Circle((x0, z0), radius_inn, color='red',
                         fill=False, linewidth=lw)
    circle3 = plt.Circle((x0, z0), radius_inn/2 + radius_out/2, color='black',
                         fill=False, linestyle='--')
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)

    # Plot one beam in bold
    Ax = Lx / 2 - a / 2 + b / 2
    Az = b / 2
    ax.plot([Ax, Lx/2], [Az, Lz/2], '--', color='black')
    ax.plot([Ax, Ax + rect_triangle_long], [Az, Az + rect_triangle_short],
            '--', color='black')
    ax.plot([Ax, Ax + triangle_long], [Az, Az + triangle_short],
            linewidth=lw, color='black')
    ax.plot([Ax + thickness/2, Ax + triangle_long + thickness/2],
            [Az - thickness/2, Az + triangle_short - thickness/2],
            linewidth=lw, color='red')
    ax.plot([Ax - thickness/2, Ax + triangle_long - thickness/2],
            [Az + thickness/2, Az + triangle_short + thickness/2],
            linewidth=lw, color='red')

    # Plot other beams
    ax.plot([Lx/2 + a/2 - b/2, Lx/2 + a/2 - b/2 - triangle_short],
            [b/2, b/2 + triangle_long], color='black')
    ax.plot([Lx/2 + a/2 - b/2, Lx/2 + a/2 - b/2 - triangle_long],
            [Lz - b/2, Lz - b/2 - triangle_short], color='black')
    ax.plot([Lx/2 - a/2 + b/2, Lx/2 - a/2 + b/2 + triangle_short],
            [Lz - b/2, Lz - b/2 - triangle_long], color='black')

    ### ----- y-z-plot (at specific x) ----- ###
    # Projection on x-z-plane
    ind_x = round((lengths[0]/2 - a/2) / lengths[0] * nb_grid_pts[0])

    # Plot geometry
    fig2, ax = plt.subplots()
    fig2.suptitle(f'x = {ind_x * lengths[0] / nb_grid_pts[0]:.2}')
    plot_2D_cut(ax, mask, lengths, index = ind_x, plane = 0)

    # Plot boundaries of RVE
    ax.plot([Ly/2 - a/2, Ly/2 + a/2],
            [0, 0], color='orange', linewidth=lw)
    ax.plot([Ly/2 - a/2, Ly/2 + a/2],
            [Lz, Lz], color='orange', linewidth=lw)
    ax.plot([Ly/2 - a/2, Ly/2 - a/2],
            [0, Lz], color='orange', linewidth=lw)
    ax.plot([Ly/2 + a/2, Ly/2 + a/2],
            [0, Lz], color='orange', linewidth=lw)

    # Plot projection of circles in x-z-plane
    y0 = lengths[1] / 2
    z0 = lengths[2] / 2
    circle1 = plt.Circle((y0, z0), radius_out, color='red', fill=False,
                         linewidth=lw)
    circle2 = plt.Circle((y0, z0), radius_inn, color='red',
                         fill=False, linewidth=lw)
    circle3 = plt.Circle((y0, z0), radius_inn/2 + radius_out/2, color='black',
                         fill=False, linestyle='--')
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)

    # Plot one beam in bold
    Ay = Ly / 2 - a / 2 + b / 2
    Az = b / 2
    ax.plot([Ay, Ly/2], [Az, Lz/2], '--', color='black')
    ax.plot([Ay, Ay + rect_triangle_long], [Az, Az + rect_triangle_short],
            '--', color='black')
    ax.plot([Ay, Ay + triangle_long], [Az, Az + triangle_short],
            linewidth=lw, color='black')
    ax.plot([Ay + thickness/2, Ay + triangle_long + thickness/2],
            [Az - thickness/2, Az + triangle_short - thickness/2],
            linewidth=lw, color='red')
    ax.plot([Ay - thickness/2, Ay + triangle_long - thickness/2],
            [Az + thickness/2, Az + triangle_short + thickness/2],
            linewidth=lw, color='red')

    # Plot other beams
    ax.plot([Ly/2 + a/2 - b/2, Ly/2 + a/2 - b/2 - triangle_short],
            [b/2, b/2 + triangle_long], color='black')
    ax.plot([Ly/2 + a/2 - b/2, Ly/2 + a/2 - b/2 - triangle_long],
            [Lz - b/2, Lz - b/2 - triangle_short], color='black')
    ax.plot([Ly/2 - a/2 + b/2, Ly/2 - a/2 + b/2 + triangle_short],
            [Lz - b/2, Lz - b/2 - triangle_long], color='black')


    ### ----- x-y-plot (specific z) ----- ###
    # Plot geometry
    fig3, ax = plt.subplots()
    fig3.suptitle(f'z = 0')
    plot_2D_cut(ax, mask, lengths, index = 0, plane = 2)

    # Plot boundaries of RVE
    ax.plot([Lx/2 - a/2, Lx/2 + a/2],
            [Ly/2 - a/2, Ly/2 - a/2],
            color='orange', linewidth=lw)
    ax.plot([Lx/2 + a/2, Lx/2 + a/2],
            [Ly/2 - a/2, Ly/2 + a/2],
            color='orange', linewidth=lw)
    ax.plot([Lx/2 - a/2, Lx/2 + a/2],
            [Ly/2 + a/2, Ly/2 + a/2],
            color='orange', linewidth=lw)
    ax.plot([Lx/2 - a/2, Lx/2 - a/2],
            [Ly/2 - a/2, Ly/2 + a/2],
            color='orange', linewidth=lw)

    # Plot projection of circles in y-z-plane
    x0 = lengths[0] / 2
    y0 = lengths[1] / 2
    circle1 = plt.Circle((x0, y0), radius_out, color='red', fill=False)
    circle2 = plt.Circle((x0, y0), radius_inn, color='red',
                         fill=False)
    circle3 = plt.Circle((x0, y0), radius_inn/2 + radius_out/2, color='black',
                         fill=False, linestyle='--')
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)

    # Plot one beam in bold
    Ax = Lx / 2 - a / 2 + b / 2
    Ay = Ly / 2 - a / 2 + b / 2
    ax.plot([Ax, Lx/2], [Ay, Ly/2], '--', color='black')
    ax.plot([Ax, Ax + rect_triangle_long], [Ay, Ay + rect_triangle_short],
            '--', color='black')
    ax.plot([Ax, Ax + triangle_long], [Ay, Ay + triangle_short],
            linewidth=lw, color='black')
    ax.plot([Ax + thickness/2, Ax + triangle_long + thickness/2],
            [Ay - thickness/2, Ay + triangle_short - thickness/2],
            linewidth=lw, color='red')
    ax.plot([Ax - thickness/2, Ax + triangle_long - thickness/2],
            [Ay + thickness/2, Ay + triangle_short + thickness/2],
            linewidth=lw, color='red')

    # Plot other beams
    ax.plot([Lx/2 + a/2 - b/2, Lx/2 + a/2 - b/2 - triangle_short],
            [Ly/2 - a/2 + b/2, Ly/2 - a/2 + b/2 + triangle_long],
            color='black')
    ax.plot([Lx/2 + a/2 - b/2, Lx/2 + a/2 - b/2 - triangle_long],
            [Ly/2 + a/2 - b/2, Ly/2 + a/2 - b/2 - triangle_short],
            color='black')
    ax.plot([Lx/2 - a/2 + b/2, Lx/2 - a/2 + b/2 + triangle_short],
            [Ly/2 + a/2 - b/2, Ly/2 + a/2 - b/2 - triangle_long],
            color='black')

    return fig1, fig2, fig3

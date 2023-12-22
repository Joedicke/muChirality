"""
@file   plotting.py

@author Indre  Joedicke <indre.joedicke@imtek.uni-freiburg.de>

@date   18 Dec 2023

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
import muChirality.Plotting as plot

def plot_beam():
    """ Plot a rectangular beam for test purposes. """
    ### ----- Parameter definitions ----- ###
    # Unit cell
    nb_grid_pts = [20, 20, 20]
    lengths = [1, 1, 1]

    # Geometry
    Lbx = 0.6
    Lby = 0.3

    # Material
    mask = geo.rectangular_beam(nb_grid_pts, lengths, Lbx, Lby)

    ### ---- Plots ----- ###
    # Prepare figure
    fig, axes = plt.subplots(1, 2)
    fig.suptitle('Rectangular beam')

    # Plot in xz-plane
    plot.plot_2D_projection(axes[0], mask, lengths, plane = 1)

    # Plot in yz-plane
    plot.plot_2D_projection(axes[1], mask, lengths, plane = 0)

    plt.show()

def plot_hollow_cylinder():
    """ Plot a hollow cylinder for test purposes. """
    ### ----- Parameter definitions ----- ###
    # Unit cell
    nb_grid_pts = [20, 20, 20]
    lengths = [1, 1, 1]

    # Geometry
    radius_in = 0.3
    thickness = 0.1

    # Material
    mask = geo.hollow_cylinder(nb_grid_pts, lengths, radius_in, thickness)

    ### ---- Plots ----- ###
    # Plot in xz-plane
    fig, axes = plt.subplots(1, 2)
    fig.suptitle('Hollow cylinder - xz plane')
    axes[0].set_title('Projection')
    plot.plot_2D_projection(axes[0], mask, lengths, plane = 1)
    axes[1].set_title('Cut')
    plot.plot_2D_cut(axes[1], mask, lengths, index = nb_grid_pts[1]//2, plane = 1)

    # Plot in xy-plane
    fig, ax = plt.subplots()
    fig.suptitle('Hollow cylinder - xy plane')
    plot.plot_2D_cut(ax, mask, lengths, plane = 2)

    plt.show()

def plot_chiral_metamaterial():
    ### ----- Parameter definitions ----- ###
    # Unit cell
    nb_grid_pts = [20, 20, 20]
    lengths = [1, 1, 1]

    # Geometry
    radius = 0.3
    n = 2 # thickness in voxels
    thickness = lengths[0] / nb_grid_pts[0] * n
    alpha = 0.2

    ### ----- Show + save figures ----- ###
    fig1, fig2, fig3 = plot.plot_2D_chiral(nb_grid_pts, lengths,
                                           radius, thickness, alpha)

    # Saving
    # TODO: DO I NEED THESE PICTURES SOMEWHERE ?
    #if saving:
    #    folder = 'example/plots/chiral_metamaterial/'
    #    name = folder + 'projection_on_xz.pdf'
    #    fig1.savefig(name, bbox_inches='tight')
    #    name = 'plots_testing/test_metamat_one_beam_yz.pdf'
    #    fig2.savefig(name, bbox_inches='tight')
    #    name = 'plots_testing/test_metamat_one_beam_xy.pdf'
    #    fig3.savefig(name, bbox_inches='tight')

    # Show
    plt.show()

def plot_chiral_metamaterial_2():
    ### ----- Parameter definitions ----- ###
    # Unit cell
    nb_grid_pts = [20, 20, 20]
    Lxy = 1

    # Geometry
    a = 0.9
    radius = 0.25
    n = 2 # thickness in voxels
    thickness = Lxy / nb_grid_pts[0] * n
    radius_inn = radius - thickness
    alpha = 0.2
    lengths = [Lxy, Lxy, a - 1.5 * thickness]

    ### ----- Show + save figures ----- ###
    figures = plot.plot_2D_chiral_2(nb_grid_pts, lengths, radius,
                                    radius_inn, thickness, alpha)
    fig1, fig2, fig3 = figures

    # Show
    plt.show()

if __name__ == "__main__":
    plot_beam()
    plot_hollow_cylinder()
    plot_chiral_metamaterial()
    plot_chiral_metamaterial_2()

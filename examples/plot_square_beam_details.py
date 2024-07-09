import numpy as np
import matplotlib.pyplot as plt

from torsion_cylinder import plotting

def plot_square_beam_details():
    ### ----- Parameters ----- ###
    # Standard
    m = 'method2'
    L_beam_stand = 0.6
    Lz_stand = 5
    Young_stand = 100
    Poisson_stand = 0
    twist_stand = 0.1
    Nz_stand = 30

    # Changes
    L_beam_list = [0.4, 0.8]
    Lz_list = [3, 7]
    Young_list = [80, 120]
    Poisson_list = [0.1, 0.2]
    twist_list = [0.05, 0.15]
    Nz_list = [45, 60]

    # Read data - standard case
    folder = 'results/square_beam/' + m + f'/thick={L_beam_stand}_Lz={Lz_stand}'
    folder += f'_Young={Young_stand}_Poisson={Poisson_stand}'
    folder += f'_twist={twist_stand}_Nz={Nz_stand}/'
    data = np.loadtxt((folder + 'data.txt'), skiprows=1)
    Nx_stand = data[:, 0].astype(int)
    err_stand = data[:, 5]

    print('Finished standard case.')

    ### ----- Prepare figures ----- ###
    # Colors
    import matplotlib.colors as mcolors
    #print(mcolors.get_named_colors_mapping()[0])
    colors = [mcolors.CSS4_COLORS['black'], mcolors.CSS4_COLORS['silver'],
              mcolors.CSS4_COLORS['brown'], mcolors.CSS4_COLORS['red'],
              mcolors.CSS4_COLORS['peru'], mcolors.CSS4_COLORS['orange'],
              mcolors.CSS4_COLORS['gold'], mcolors.CSS4_COLORS['olive'],
              mcolors.CSS4_COLORS['yellowgreen'], mcolors.CSS4_COLORS['lawngreen'],
              mcolors.CSS4_COLORS['green'], mcolors.CSS4_COLORS['deepskyblue'],
              mcolors.CSS4_COLORS['blue']]

    # Prepare figure
    fig = plt.figure(figsize=(7, 5))
    gs_left = fig.add_gridspec(nrows=1, ncols=1, top=0.92, bottom=0.09,
                                          left=0.08, right=0.75)
    gs_right = fig.add_gridspec(nrows=1, ncols=1, top=0.92, bottom=0.09,
                                          left=0.78, right=0.99)
    ax1 = fig.add_subplot(gs_left[0, 0])
    ax2 = fig.add_subplot(gs_right[0, 0])
    ax2.axis('off')

    fig.suptitle('Relative error of torsion stiffness')
    ax1.set_xlabel('Nx = Ny')
    ax1.set_ylabel('Err. of stiff (%)')


    ### ----- Plot standard case ----- ###
    # Print standard parameter
    ax2.text(0, 0.9, 'Standard', fontsize='large')
    ax2.text(0, 0.86, '   parameters', fontsize='large')
    ax2.text(0, 0.8, f'L_beam={L_beam_stand}')
    ax2.text(0, 0.75, f'Lz={Lz_stand}')
    ax2.text(0, 0.7, f'Young={Young_stand}')
    ax2.text(0, 0.65, f'Poisson={Poisson_stand}')
    ax2.text(0, 0.6, f'twist={twist_stand}')
    ax2.text(0, 0.55, f'Nz={Nz_stand}')

    # Plot standard cases
    label = 'Standard'
    i = 0
    ax1.plot(Nx_stand, err_stand, label=label, color=colors[i])
    i += 1

    ### ----- Radius ----- ###
    for L_beam in L_beam_list:
        # Folder
        folder = 'results/square_beam/' + m + f'/thick={L_beam}_Lz={Lz_stand}'
        folder += f'_Young={Young_stand}_Poisson={Poisson_stand}'
        folder += f'_twist={twist_stand}_Nz={Nz_stand}/'

        # Read data
        data = np.loadtxt((folder + 'data.txt'), skiprows=1)
        Nx = data[:, 0].astype(int)
        err = data[:, 5]

        # Plot on figure all cases
        label = f'L_beam={L_beam}'
        ax1.plot(Nx, err, label=label, color = colors[i])
        i += 1

    print('Finished L_beam.')

    ### ----- Lz ----- ###
    for Lz in Lz_list:
        # Folder
        folder = 'results/square_beam/' + m + f'/thick={L_beam_stand}_Lz={Lz}'
        folder += f'_Young={Young_stand}_Poisson={Poisson_stand}'
        folder += f'_twist={twist_stand}_Nz={Nz_stand}/'

        # Read data
        data = np.loadtxt((folder + 'data.txt'), skiprows=1)
        Nx = data[:, 0].astype(int)
        err = data[:, 5]

        # Plot on figure all cases
        label = f'Lz={Lz}'
        ax1.plot(Nx, err, label=label, color = colors[i])
        i += 1


    print('Finished Lz.')

    ### ----- Young ----- ###
    for Young in Young_list:
        # Folder
        folder = 'results/square_beam/' + m + f'/thick={L_beam_stand}_Lz={Lz_stand}'
        folder += f'_Young={Young}_Poisson={Poisson_stand}'
        folder += f'_twist={twist_stand}_Nz={Nz_stand}/'

        # Read data
        data = np.loadtxt((folder + 'data.txt'), skiprows=1)
        Nx = data[:, 0].astype(int)
        err = data[:, 5]

        # Plot on figure all cases
        label = f'Young={Young}'
        ax1.plot(Nx, err, label=label, color = colors[i])
        i += 1


    print('Finished Young.')

    ### ----- Poisson ----- ###
    for Poisson in Poisson_list:
        # Folder
        folder = 'results/square_beam/' + m + f'/thick={L_beam_stand}_Lz={Lz_stand}'
        folder += f'_Young={Young_stand}_Poisson={Poisson}'
        folder += f'_twist={twist_stand}_Nz={Nz_stand}/'

        # Read data
        data = np.loadtxt((folder + 'data.txt'), skiprows=1)
        Nx = data[:, 0].astype(int)
        err = data[:, 5]

        # Plot on figure all cases
        label = f'Poisson={Poisson}'
        ax1.plot(Nx, err, label=label, color = colors[i])
        i += 1

    print('Finished Poisson.')

    ### ----- twist ----- ###
    for twist in twist_list:
        # Folder
        folder = 'results/square_beam/' + m + f'/thick={L_beam_stand}_Lz={Lz_stand}'
        folder += f'_Young={Young_stand}_Poisson={Poisson_stand}'
        folder += f'_twist={twist}_Nz={Nz_stand}/'

        # Read data
        data = np.loadtxt((folder + 'data.txt'), skiprows=1)
        Nx = data[:, 0].astype(int)
        err = data[:, 5]

        # Plot on figure all cases
        label = f'twist={twist}'
        ax1.plot(Nx, err, label=label, color = colors[i])
        i += 1

    print('Finished twist.')

    ### ----- Nz ----- ###
    for Nz in Nz_list:
        # Folder
        folder = 'results/square_beam/' + m + f'/thick={L_beam_stand}_Lz={Lz_stand}'
        folder += f'_Young={Young_stand}_Poisson={Poisson_stand}'
        folder += f'_twist={twist_stand}_Nz={Nz}/'

        # Read data
        data = np.loadtxt((folder + 'data.txt'), skiprows=1)
        Nx = data[:, 0].astype(int)
        err = data[:, 5]

        # Plot on figure all cases
        label = f'Nz={Nz}'
        ax1.plot(Nx, err, label=label, color = colors[i])
        i += 1

    print('Finished Nz.')

    ### ----- Finish ----- ###
    # Legend
    ax1.legend()

    # Show figure
    plt.show()

    # Save figure
    folder = 'results/square_beam/' + m + '/'
    name = folder + 'overview_err_stiff.pdf'
    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)




if __name__ == "__main__":
    plot_square_beam_details()

import numpy as np
import matplotlib.pyplot as plt

from torsion_cylinder import plotting

def plot_cylinder_details():
    ### ----- Parameters ----- ###
    # Standard
    m = 'method1'
    radius_stand = 0.4
    Lz_stand = 10
    Young_stand = 100
    Poisson_stand = 0
    twist_stand = 0.15
    Nz_stand = 30

    # Changes
    radius_list = [0.375, 0.425]
    Lz_list = [5, 15]
    Young_list = [80, 120]
    Poisson_list = [0.1, 0.2]
    twist_list = [0.1, 0.05]
    Nz_list = [45, 60]

    # Read data - standard case
    folder = 'results/cylinder/' + m + f'/radius={radius_stand}_Lz={Lz_stand}'
    folder += f'_Young={Young_stand}_Poisson={Poisson_stand}'
    folder += f'_twist={twist_stand}_Nz={Nz_stand}/'
    data = np.loadtxt((folder + 'data.txt'), skiprows=1)
    Nx_stand = data[:, 0].astype(int)
    err_stress_stand = data[:, 3]
    err_stiff_stand = data[:, 7]

    # Displacement data - standard case
    data = np.loadtxt((folder + 'data_displacement.txt'), skiprows=1)
    err_displ_stand = data[:, 3]

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

    # Prepare figure: Stress
    fig_all_stress = plt.figure(figsize=(7, 5))
    gs_left = fig_all_stress.add_gridspec(nrows=1, ncols=1, top=0.92, bottom=0.09,
                                          left=0.08, right=0.75)
    gs_right = fig_all_stress.add_gridspec(nrows=1, ncols=1, top=0.93, bottom=0.07,
                                          left=0.78, right=0.99)
    ax1_all_stress = fig_all_stress.add_subplot(gs_left[0, 0])
    ax2_all_stress = fig_all_stress.add_subplot(gs_right[0, 0])
    ax2_all_stress.axis('off')

    fig_all_stress.suptitle('Relative norm of error of stress')
    ax1_all_stress.set_xlabel('Nx = Ny')
    ax1_all_stress.set_ylabel('Err. of stress (%)')

    # Prepare figure: Stiffness
    fig_all_stiff = plt.figure(figsize=(7, 5))
    gs_left = fig_all_stiff.add_gridspec(nrows=1, ncols=1, top=0.92, bottom=0.09,
                                          left=0.08, right=0.75)
    gs_right = fig_all_stiff.add_gridspec(nrows=1, ncols=1, top=0.92, bottom=0.09,
                                          left=0.78, right=0.99)
    ax1_all_stiff = fig_all_stiff.add_subplot(gs_left[0, 0])
    ax2_all_stiff = fig_all_stiff.add_subplot(gs_right[0, 0])
    #ax2_all_stiff.set_ylim(0, 1)
    ax2_all_stiff.axis('off')

    fig_all_stiff.suptitle('Relative error of torsion stiffness')
    ax1_all_stiff.set_xlabel('Nx = Ny')
    ax1_all_stiff.set_ylabel('Err. of stiff (%)')

    # Prepare figure: Displacement
    fig_all_displ = plt.figure(figsize=(7, 5))
    gs_left = fig_all_displ.add_gridspec(nrows=1, ncols=1, top=0.92, bottom=0.09,
                                          left=0.08, right=0.75)
    gs_right = fig_all_displ.add_gridspec(nrows=1, ncols=1, top=0.93, bottom=0.07,
                                          left=0.78, right=0.99)
    ax1_all_displ = fig_all_displ.add_subplot(gs_left[0, 0])
    ax2_all_displ = fig_all_displ.add_subplot(gs_right[0, 0])
    ax2_all_displ.axis('off')

    fig_all_displ.suptitle('Relative norm of error of displacements')
    ax1_all_displ.set_xlabel('Nx = Ny')
    ax1_all_displ.set_ylabel('Err. of displ (%)')


    ### ----- Plot standard case ----- ###
    # Print standard parameter (stress-figure)
    ax2_all_stress.text(0, 0.9, 'Standard', fontsize='large')
    ax2_all_stress.text(0, 0.86, '   parameters', fontsize='large')
    ax2_all_stress.text(0, 0.8, f'radius={radius_stand}')
    ax2_all_stress.text(0, 0.75, f'Lz={Lz_stand}')
    ax2_all_stress.text(0, 0.7, f'Young={Young_stand}')
    ax2_all_stress.text(0, 0.65, f'Poisson={Poisson_stand}')
    ax2_all_stress.text(0, 0.6, f'twist={twist_stand}')
    ax2_all_stress.text(0, 0.55, f'Nz={Nz_stand}')

    # Print standard parameter (stiff-figure)
    ax2_all_stiff.text(0, 0.9, 'Standard', fontsize='large')
    ax2_all_stiff.text(0, 0.86, '   parameters', fontsize='large')
    ax2_all_stiff.text(0, 0.8, f'radius={radius_stand}')
    ax2_all_stiff.text(0, 0.75, f'Lz={Lz_stand}')
    ax2_all_stiff.text(0, 0.7, f'Young={Young_stand}')
    ax2_all_stiff.text(0, 0.65, f'Poisson={Poisson_stand}')
    ax2_all_stiff.text(0, 0.6, f'twist={twist_stand}')
    ax2_all_stiff.text(0, 0.55, f'Nz={Nz_stand}')

    # Print standard parameter (displ-figure)
    ax2_all_displ.text(0, 0.9, 'Standard', fontsize='large')
    ax2_all_displ.text(0, 0.86, '   parameters', fontsize='large')
    ax2_all_displ.text(0, 0.8, f'radius={radius_stand}')
    ax2_all_displ.text(0, 0.75, f'Lz={Lz_stand}')
    ax2_all_displ.text(0, 0.7, f'Young={Young_stand}')
    ax2_all_displ.text(0, 0.65, f'Poisson={Poisson_stand}')
    ax2_all_displ.text(0, 0.6, f'twist={twist_stand}')
    ax2_all_displ.text(0, 0.55, f'Nz={Nz_stand}')

    # Plot standard cases
    label = 'Standard'
    i = 0
    ax1_all_stress.plot(Nx_stand, err_stress_stand, label=label, color=colors[i])
    ax1_all_stiff.plot(Nx_stand, err_stiff_stand, label=label, color=colors[i])
    ax1_all_displ.plot(Nx_stand, err_displ_stand, label=label, color=colors[i])
    i += 1

    ### ----- Radius ----- ###
    for radius in radius_list:
        # Folder
        folder = 'results/cylinder/' + m + f'/radius={radius}_Lz={Lz_stand}'
        folder += f'_Young={Young_stand}_Poisson={Poisson_stand}'
        folder += f'_twist={twist_stand}_Nz={Nz_stand}/'

        # Read data
        data = np.loadtxt((folder + 'data.txt'), skiprows=1)
        Nx = data[:, 0].astype(int)
        err_stress = data[:, 3]
        err_stiff = data[:, 7]
        data = np.loadtxt((folder + 'data_displacement.txt'), skiprows=1)
        err_displ = data[:, 3]

        # Plot on figure all cases
        label = f'radius={radius}'
        ax1_all_stress.plot(Nx, err_stress, label=label, color = colors[i])
        ax1_all_stiff.plot(Nx, err_stiff, label=label, color = colors[i])
        ax1_all_displ.plot(Nx, err_displ, label=label, color = colors[i])
        i += 1

    print('Finished radius.')

    ### ----- Lz ----- ###
    for Lz in Lz_list:
        # Folder
        folder = 'results/cylinder/' + m + f'/radius={radius_stand}_Lz={Lz}'
        folder += f'_Young={Young_stand}_Poisson={Poisson_stand}'
        folder += f'_twist={twist_stand}_Nz={Nz_stand}/'

        # Read data
        data = np.loadtxt((folder + 'data.txt'), skiprows=1)
        Nx = data[:, 0].astype(int)
        err_stress = data[:, 3]
        err_stiff = data[:, 7]
        data = np.loadtxt((folder + 'data_displacement.txt'), skiprows=1)
        err_displ = data[:, 3]

        # Plot on figure all cases
        label = f'Lz={Lz}'
        ax1_all_stress.plot(Nx, err_stress, label=label, color = colors[i])
        ax1_all_stiff.plot(Nx, err_stiff, label=label, color = colors[i])
        ax1_all_displ.plot(Nx, err_displ, label=label, color = colors[i])
        i += 1


    print('Finished Lz.')

    ### ----- Young ----- ###
    for Young in Young_list:
        # Folder
        folder = 'results/cylinder/' + m + f'/radius={radius_stand}_Lz={Lz_stand}'
        folder += f'_Young={Young}_Poisson={Poisson_stand}'
        folder += f'_twist={twist_stand}_Nz={Nz_stand}/'

        # Read data
        data = np.loadtxt((folder + 'data.txt'), skiprows=1)
        Nx = data[:, 0].astype(int)
        err_stress = data[:, 3]
        err_stiff = data[:, 7]
        data = np.loadtxt((folder + 'data_displacement.txt'), skiprows=1)
        err_displ = data[:, 3]

        # Plot on figure all cases
        label = f'Young={Young}'
        ax1_all_stress.plot(Nx, err_stress, label=label, color = colors[i])
        ax1_all_stiff.plot(Nx, err_stiff, label=label, color = colors[i])
        ax1_all_displ.plot(Nx, err_displ, label=label, color = colors[i])
        i += 1


    print('Finished Young.')

    ### ----- Poisson ----- ###
    for Poisson in Poisson_list:
        # Folder
        folder = 'results/cylinder/' + m + f'/radius={radius_stand}_Lz={Lz_stand}'
        folder += f'_Young={Young_stand}_Poisson={Poisson}'
        folder += f'_twist={twist_stand}_Nz={Nz_stand}/'

        # Read data
        data = np.loadtxt((folder + 'data.txt'), skiprows=1)
        Nx = data[:, 0].astype(int)
        err_stress = data[:, 3]
        err_stiff = data[:, 7]
        data = np.loadtxt((folder + 'data_displacement.txt'), skiprows=1)
        err_displ = data[:, 3]

        # Plot on figure all cases
        label = f'Poisson={Poisson}'
        ax1_all_stress.plot(Nx, err_stress, label=label, color = colors[i])
        ax1_all_stiff.plot(Nx, err_stiff, label=label, color = colors[i])
        ax1_all_displ.plot(Nx, err_displ, label=label, color = colors[i])
        i += 1

    print('Finished Poisson.')

    ### ----- twist ----- ###
    for twist in twist_list:
        # Folder
        folder = 'results/cylinder/' + m + f'/radius={radius_stand}_Lz={Lz_stand}'
        folder += f'_Young={Young_stand}_Poisson={Poisson_stand}'
        folder += f'_twist={twist}_Nz={Nz_stand}/'

        # Read data
        data = np.loadtxt((folder + 'data.txt'), skiprows=1)
        Nx = data[:, 0].astype(int)
        err_stress = data[:, 3]
        err_stiff = data[:, 7]
        data = np.loadtxt((folder + 'data_displacement.txt'), skiprows=1)
        err_displ = data[:, 3]

        # Plot on figure all cases
        label = f'twist={twist}'
        ax1_all_stress.plot(Nx, err_stress, label=label, color = colors[i])
        ax1_all_stiff.plot(Nx, err_stiff, label=label, color = colors[i])
        ax1_all_displ.plot(Nx, err_displ, label=label, color = colors[i])
        i += 1

    print('Finished twist.')

    ### ----- Nz ----- ###
    for Nz in Nz_list:
        # Folder
        folder = 'results/cylinder/' + m + f'/radius={radius_stand}_Lz={Lz_stand}'
        folder += f'_Young={Young_stand}_Poisson={Poisson_stand}'
        folder += f'_twist={twist_stand}_Nz={Nz}/'

        # Read data
        data = np.loadtxt((folder + 'data.txt'), skiprows=1)
        Nx = data[:, 0].astype(int)
        err_stress = data[:, 3]
        err_stiff = data[:, 7]
        data = np.loadtxt((folder + 'data_displacement.txt'), skiprows=1)
        err_displ = data[:, 3]

        # Plot on figure all cases
        label = f'Nz={Nz}'
        ax1_all_stress.plot(Nx, err_stress, label=label, color = colors[i])
        ax1_all_stiff.plot(Nx, err_stiff, label=label, color = colors[i])
        ax1_all_displ.plot(Nx, err_displ, label=label, color = colors[i])
        i += 1

    print('Finished Nz.')

    ### ----- Finish ----- ###
    # Legends
    ax1_all_stress.legend()
    ax1_all_stiff.legend()
    ax1_all_displ.legend()

    # Show figures
    plt.show()

    # Save figures
    folder = 'results/cylinder/' + m + '/'
    name = folder + 'overview_err_stress.pdf'
    fig_all_stress.savefig(name, bbox_inches='tight')
    plt.close(fig_all_stress)
    name = folder + 'overview_err_stiff.pdf'
    fig_all_stiff.savefig(name, bbox_inches='tight')
    plt.close(fig_all_stiff)
    name = folder + 'overview_err_displ.pdf'
    fig_all_displ.savefig(name, bbox_inches='tight')
    plt.close(fig_all_displ)




if __name__ == "__main__":
    plot_cylinder_details()

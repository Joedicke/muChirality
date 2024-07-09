import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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
### ------------- Plot error torsion stiffnesses ------------- ###
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
folder_cyl = f'results/cylinder/method2/radius={0.4}_Lz={10}_Young={100}_'
folder_cyl += f'Poisson={0}_twist={0.15}_Nz={30}/'
folder_sq = f'results/square_beam/method2/thick={0.6}_Lz={10}_Young={100}_'
folder_sq += f'Poisson={0}_twist={0.15}_Nz={30}/'

# Read error stiffness cylinder
data = np.loadtxt((folder_cyl + 'data.txt'), skiprows=1)
Nx_cyl = data[:, 0].astype(int)
err_stiff_cyl = data[:, 7]

# Read error stiffness square beam
data = np.loadtxt((folder_sq + 'data.txt'), skiprows=1)
Nx_sq = data[:, 0].astype(int)
err_stiff_sq = data[:, 5]

### ----- Plot data ----- ###
fig, ax = plt.subplots()
ax.set_xlabel('Number of voxels in x- and y-direction (-)', fontsize=size_large)
ax.set_ylabel('Difference between stiffnesses (%)', fontsize=size_large)
ax.tick_params(axis='both', which='major', labelsize=tick_size)

ax.plot(Nx_cyl, err_stiff_cyl, marker='x', color=color1)
ax.plot(Nx_sq, err_stiff_sq, marker='x', color=color2)

x = 1.2 * Nx_cyl[0]
y = 0.8 * err_stiff_cyl[0]
ax.text(x, y, 'circular cross-section', color=color1, fontsize=size_large)

#x = 0.93 * Nx_sq[0]
#y = 1.9 * err_stiff_sq[0]
x = 1.02 * Nx_sq[-1]
y = 0.5 * err_stiff_sq[0]
ax.text(x, y, 'square cross-section', color=color2, fontsize=size_large, ha='right')


### ----- Finish ----- ###
name = 'results/torsion_stiffnesses.pdf'
fig.savefig(name, bbox_inches='tight')
plt.show()
plt.close(fig)

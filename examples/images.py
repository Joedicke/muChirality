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
### --------------- Plot small strain rotation --------------- ###
##################################################################
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
#D1_x = U1_x + twist * np.sin(np.pi / 4)
#D1_y = U1_y - twist * np.cos(np.pi / 4)
#D2_x = U2_x - twist * np.cos(np.pi / 4)
#D2_y = U2_y - twist * np.sin(np.pi / 4)
#D3_x = U3_x - twist * np.sin(np.pi / 4)
#D3_y = U3_y + twist * np.cos(np.pi / 4)
#D4_x = U4_x + twist * np.cos(np.pi / 4)
#D4_y = U4_y + twist * np.sin(np.pi / 4)
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
#ax.text(D1_x + 0 / 5 * (D1_x - U1_x), D1_y + (D1_y - U1_y) / 1, '$u_{max}$',
#        color=color_twist, fontstyle='italic', ha='left', fontsize=fontsize)
ax.text(U2_x + (D2_x - U2_x) / 2, U2_y + (D2_y - U2_y) / 4, '$u_{max}$',
        color=color_twist, fontstyle='italic', ha='right', fontsize=fontsize)
#ax.text(U3_x + (D3_x - U3_x) / 2, U3_y + (D3_y - U3_y) / 2, '$u_{max}$',
#        color=color_twist, fontstyle='italic', fontsize=fontsize)
#ax.text(U4_x + (D4_x - U4_x) / 2, U4_y + (D4_y - U4_y) / 2, '$u_{max}$',
#        color=color_twist, fontstyle='italic', va='top', fontsize=fontsize)






### ----- Save figure ----- ###
name = 'results/rotation.pdf'
fig.savefig(name, bbox_inches='tight')
plt.show()
plt.close(fig)

##################################################################
### -------------- Plot discretization of cube --------------- ###
##################################################################
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


### ----- Plotting small strain rotation ----- ###
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
name = 'results/discretization.pdf'
fig.savefig(name, bbox_inches='tight')
#plt.show()
plt.close(fig)

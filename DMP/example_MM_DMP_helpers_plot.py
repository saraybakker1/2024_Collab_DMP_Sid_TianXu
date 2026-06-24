
import numpy as np
from io import BytesIO
import imageio
import pdb

def gen_3d_plot2(plt, gamma, gamma2, measured_xyz, desired_xyz):
    measured_x = measured_xyz[0]
    measured_y = measured_xyz[1]
    measured_z = measured_xyz[2]
    desired_x = desired_xyz[0]
    desired_y = desired_xyz[1]
    desired_z = desired_xyz[2]

    # Create a 3D figure
    fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(111, projection='3d')

    x1_positions_m, y1_positions_m, z1_positions_m = measured_x[0], measured_y[0], measured_z[0]
    split_1 = len(x1_positions_m) // 3
    split_2 = 2 * len(x1_positions_m) // 3
    x1_part1_m, y1_part1_m, z1_part1_m = x1_positions_m[:split_1], y1_positions_m[:split_1], z1_positions_m[:split_1]
    x1_part2_m, y1_part2_m, z1_part2_m = x1_positions_m[:split_2], y1_positions_m[:split_2], z1_positions_m[:split_2]
    x1_part3_m, y1_part3_m, z1_part3_m = x1_positions_m, y1_positions_m, z1_positions_m

    x1_positions_d, y1_positions_d, z1_positions_d = desired_x[0], desired_y[0], desired_z[0]
    split_1 = len(x1_positions_d) // 3
    split_2 = 2 * len(x1_positions_d) // 3
    x1_part1_d, y1_part1_d, z1_part1_d = x1_positions_d[:split_1], y1_positions_d[:split_1], z1_positions_d[:split_1]
    x1_part2_d, y1_part2_d, z1_part2_d = x1_positions_d[:split_2], y1_positions_d[:split_2], z1_positions_d[:split_2]
    x1_part3_d, y1_part3_d, z1_part3_d = x1_positions_d, y1_positions_d, z1_positions_d

    x2_positions_m, y2_positions_m, z2_positions_m = measured_x[1], measured_y[1], measured_z[1]
    split_1 = len(x1_positions_m) // 3
    split_2 = 2 * len(x1_positions_m) // 3
    x2_part1_m, y2_part1_m, z2_part1_m = x2_positions_m[:split_1], y2_positions_m[:split_1], z2_positions_m[:split_1]
    x2_part2_m, y2_part2_m, z2_part2_m = x2_positions_m[:split_2], y2_positions_m[:split_2], z2_positions_m[:split_2]
    x2_part3_m, y2_part3_m, z2_part3_m = x2_positions_m, y2_positions_m, z2_positions_m

    x2_positions_d, y2_positions_d, z2_positions_d = desired_x[1], desired_y[1], desired_z[1]
    split_1 = len(x1_positions_d) // 3
    split_2 = 2 * len(x1_positions_d) // 3
    x2_part1_d, y2_part1_d, z2_part1_d = x2_positions_d[:split_1], y2_positions_d[:split_1], z2_positions_d[:split_1]
    x2_part2_d, y2_part2_d, z2_part2_d = x2_positions_d[:split_2], y2_positions_d[:split_2], z2_positions_d[:split_2]
    x2_part3_d, y2_part3_d, z2_part3_d = x2_positions_d, y2_positions_d, z2_positions_d

    ref = np.array(gamma).T
    k = 25
    x1_ref, y1_ref, z1_ref = ref[0, ::k].tolist(), ref[1, ::k].tolist(), ref[2, ::k].tolist()
    split_1 = len(x1_ref) // 3
    split_2 = 2 * len(x1_ref) // 3
    x1_ref1, y1_ref1, z1_ref1 = x1_ref[:split_1], y1_ref[:split_1], z1_ref[:split_1]
    x1_ref2, y1_ref2, z1_ref2 = x1_ref[:split_2], y1_ref[:split_2], z1_ref[:split_2]
    x1_ref3, y1_ref3, z1_ref3 = x1_ref, y1_ref, z1_ref

    ref2 = np.array(gamma2).T
    x2_ref, y2_ref, z2_ref = ref2[0, ::k].tolist(), ref2[1, ::k].tolist(), ref2[2, ::k].tolist()
    split_1 = len(x2_ref) // 3
    split_2 = 2 * len(x2_ref) // 3
    x2_ref1, y2_ref1, z2_ref1 = x2_ref[:split_1], y2_ref[:split_1], z2_ref[:split_1]
    x2_ref2, y2_ref2, z2_ref2 = x2_ref[:split_2], y2_ref[:split_2], z2_ref[:split_2]
    x2_ref3, y2_ref3, z2_ref3 = x2_ref, y2_ref, z2_ref

    x_max = np.array(desired_x[0]).max() + 0.1
    y_max = np.array(desired_y[0]).max() + 0.1
    z_max = np.array(desired_z[0]).max() + 0.1
    x_min = np.array(desired_x[0]).min() - 0.1
    y_min = np.array(desired_y[0]).min() - 0.1
    z_min = np.array(desired_z[0]).min() - 0.1

    # Plot the first part
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(x1_ref1, y1_ref1, z1_ref1, color='yellow', label='Reference 1', s=8, marker='o', edgecolors='black')
    # ax1.scatter(x2_ref1, y2_ref1, z2_ref1,  color='brown', label='Reference 2', s=8, marker='o')
    ax1.scatter(x1_part1_m, y1_part1_m, z1_part1_m, color='r', label='Measured 1')
    ax1.scatter(x1_part1_d, y1_part1_d, z1_part1_d, color='g', label='Desired 1')
    ax1.scatter(x2_ref1, y2_ref1, z2_ref1, color='brown', label='Reference 2', s=8, marker='o', edgecolors='black')
    ax1.scatter(x2_part1_m, y2_part1_m, z2_part1_m, color='orange', label='Measured 2')
    ax1.scatter(x2_part1_d, y2_part1_d, z2_part1_d, color='blue', label='Desired 2')
    ax1.set_title('t = T/3')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.grid(True)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_zlim(z_min, z_max)
    ax1.legend()

    # Plot the second part
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(x1_ref2, y1_ref2, z1_ref2, color='yellow', label='Reference 1', s=8, marker='o', edgecolors='black')
    # ax2.scatter(x2_ref2, y2_ref2, z2_ref2,  color='brown', label='Reference 2', s=8, marker='o')
    ax2.scatter(x1_part2_m, y1_part2_m, z1_part2_m, color='r', label='Measured 1')
    ax2.scatter(x1_part2_d, y1_part2_d, z1_part2_d, color='g', label='Desired 1')
    ax2.scatter(x2_ref2, y2_ref2, z2_ref2, color='brown', label='Reference 2', s=8, marker='o', edgecolors='black')
    ax2.scatter(x2_part2_m, y2_part2_m, z2_part2_m, color='orange', label='Measured 2')
    ax2.scatter(x2_part2_d, y2_part2_d, z2_part2_d, color='blue', label='Desired 2')
    ax2.set_title('t = 2T/3')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.grid(True)
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_zlim(z_min, z_max)
    ax2.legend()

    # Plot the third part
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(x1_ref3, y1_ref3, z1_ref3, color='yellow', label='Reference 1', s=8, marker='o', edgecolors='black')
    # ax3.scatter(x2_ref3, y2_ref3, z2_ref3, color='brown', label='Reference 2', s=8, marker='o')
    ax3.scatter(x1_part3_m, y1_part3_m, z1_part3_m, color='r', label='Measured 1')
    ax3.scatter(x1_part3_d, y1_part3_d, z1_part3_d, color='g', label='Desired 1')
    ax3.scatter(x2_ref3, y2_ref3, z2_ref3, color='brown', label='Reference 2', s=8, marker='o', edgecolors='black')
    ax3.scatter(x2_part3_m, y2_part3_m, z2_part3_m, color='orange', label='Measured 2')
    ax3.scatter(x2_part3_d, y2_part3_d, z2_part3_d, color='blue', label='Desired 2')
    ax3.set_title('t = T')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.grid(True)
    ax3.set_xlim(x_min, x_max)
    ax3.set_ylim(y_min, y_max)
    ax3.set_zlim(z_min, z_max)
    ax3.legend()

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()
    pdb.set_trace()




############## redundant plots (not used) #####################
def plot3d_2(ax, points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Scatter plot for the points
    ax.scatter(x, y, z, color='y', marker='+')
    ax.plot(x, y, z, color='k')

    # Optionally label the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def plot3d_(ax, points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Scatter plot for the points
    ax.scatter(x, y, z, color='r', marker='o')
    ax.plot(x, y, z, color='b')

    # Optionally label the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # plt.axes()
    # Set equal scaling for all axes
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0

    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-2, 2)

# DMP 3D GIF
def plotresult(writer, plt, gamma, gamma2, gamma_tra, gamma2_tra, x, y, z):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(gamma[:, 0], gamma[:, 1], gamma[:, 2], '--k', linewidth=1.5)
    ax.plot(gamma2[:, 0], gamma2[:, 1], gamma2[:, 2], '--k', linewidth=1.5)
    # ax.plot(gamma[:, 0], gamma[:, 1], gamma[:, 2],'-g', linewidth=1.5)
    ax.plot(gamma_tra[:, 0], gamma_tra[:, 1], gamma_tra[:, 2], '-r', linewidth=1.5)
    ax.plot(gamma2_tra[:, 0], gamma2_tra[:, 1], gamma2_tra[:, 2], '-r', linewidth=1.5)
    ax.plot_surface(x, y, z, color='b', alpha=0.3, edgecolor=(0, 0, 0, 0.1))  # Use alpha for transparency
    # ax.plot_surface(x_, y_, z_, color='r', alpha=0.3, edgecolor=(0, 0, 0, 0.1))  # Use alpha for transparency

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    ax.view_init(elev=10, azim=20)

    # Manually adjust the subplot margins
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    # plt.draw()         # Redraw the current figure
    # plt.pause(0.1)

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)

    buf.seek(0)
    image = imageio.imread(buf)
    writer.append_data(image)

    buf.close()

# DMP 3D plot
def gen_3d_plot(plt, gamma, gamma2, measured_xyz, desired_xyz):
    measured_x = measured_xyz[0]
    measured_y = measured_xyz[1]
    measured_z = measured_xyz[2]
    desired_x = desired_xyz[0]
    desired_y = desired_xyz[1]
    desired_z = desired_xyz[2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot reference values
    ref = np.array(gamma).T
    ax.scatter(ref[0, :], ref[1, :], ref[2, :], color='grey', label='Reference 1', s=6)

    # Plot measured values
    ax.scatter(measured_x[0], measured_y[0], measured_z[0], color='blue', label='Measured 1', s=10)

    # Plot desired values
    ax.scatter(desired_x[0], desired_y[0], desired_z[0], color='red', label='Desired 1', s=10)

    # Plot reference values
    ref2 = np.array(gamma2).T
    ax.scatter(ref2[0, :], ref2[1, :], ref2[2, :], color='pink', label='Reference 2', s=6)

    # Plot measured values
    ax.scatter(measured_x[1], measured_y[1], measured_z[1], color='orange', label='Measured 2', s=10)

    # Plot desired values
    ax.scatter(desired_x[1], desired_y[1], desired_z[1], color='green', label='Desired 2', s=10)

    # Add labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.legend()

    plt.show()
    pdb.set_trace()
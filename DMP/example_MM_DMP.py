import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

from io import BytesIO
import imageio

from dmp.dmp import DMPs_cartesian
from dmp.dmp_ import DMPs_cartesian_
# from dmp.obstacle_superquadric import Obstacle_Dynamic as sq_dyn
from dmp.obstacle_superquadric import Obstacle_Static as sq_stat
from example_MM_DMP_helpers import getLinkFeatures, gen_3d_plot2, Update_link_Data
from DMP_map_Rotation import rotation_matrix_from_vectors, tra_map_param, tra_map_forward, tra_map_inverse
from Minimal_Ellipsoid import rotate_ellipsoid, draw_ellipsoid, find_minimum_volume_ellipsoid, plot_ellipsoid_and_points
from GlobalPose import getlink_global_pose, get_semiaxes_Constants

import pybullet as pb
import pybullet_data
import time
import yaml
import pdb
from types import SimpleNamespace as SN

# from Robot_Sim.robots import ur5_robotiq
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from Robot_Sim.robots.ur5_robotiq import UR5_Robotiq
from Robot_Sim.robots.dinova import Dinova

# number of links
num_links = 9

# Potential Field Dictionary
Ellipsoid_dict_agent1 = {}
Ellipsoid_dict_agent2 = {}
used_links = [1, 2, 3, 13, 14, 15, 16, 17, 18]

# define potential field
def perturb_dynamic_agent1(x, v):
    out = 0
    for i in used_links:
        key = f'link{i}'
        # out+=Ellipsoid_dict_agent2[key].gen_external_force_rotate(x, v-alpha1*R1@velo_agent2[key], R1@rotm_agent2[key])
        out += Ellipsoid_dict_agent2[key].gen_external_force_rotate(x, R1 @ rotm_agent2[key])
    # print("Perturbance 1", out)
    return out


def perturb_dynamic_agent2(x, v):
    out = 0
    for i in used_links:
        key = f'link{i}'
        # out+=Ellipsoid_dict_agent1[key].gen_external_force_rotate(x, v-alpha2*R2@velo_agent1[key], R2@rotm_agent1[key])
        out += Ellipsoid_dict_agent1[key].gen_external_force_rotate(x, R2 @ rotm_agent1[key])
        # print('local',key, Ellipsoid_dict_agent1[key].gen_external_force_rotate(x, v-alpha2*R2@velo_agent1[key], R2@rotm_agent1[key]))
    # print("Perturbance 2", out)
    return out

if __name__ == "__main__":
    pb.connect(pb.GUI)
    # pb.connect(pb.DIRECT)
    pb.setGravity(0, 0, -9.81)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
    plane_id = pb.loadURDF("plane.urdf")

    '''
    Load Robots
    '''

    robot1 = Dinova()
    robot2 = Dinova()

    robot1.initialize(base_pos=[0.0, 0.2, 0.0], base_ori=[0, 0, 0, 1])
    robot2.initialize(base_pos=[0.65, 0.2, 0.0], base_ori=[0, 0, 0, 1])

    pb.setTimeStep(1 / 100)

    # Trajectory 2
    # Trajectory

    t_des = np.linspace(0.0, 1.0, 1000)

    '''
    Line Trajectory
    '''
    pb_state_start1 = robot1._getLinkState(robot1.end_effector_index)
    start1 = pb_state_start1[0]
    startorientation1 =  pb_state_start1[1] #todo: these orientations are very different from the ones computed with ds_collaboration, could be because of links, but maybe check.
    end1 = [0.0, 1.5, start1[-1]]
    pb_state_start2 = robot2._getLinkState(robot2.end_effector_index)
    start2 = pb_state_start2[0]
    startorientation2 = pb_state_start2[1]
    end2 = [0.5, -0.5, start2[-1]]

    visual_shape_id = pb.createVisualShape(
        shapeType=pb.GEOM_SPHERE,
        radius=0.05,
        rgbaColor=[1, 0, 0, 1]
    )

    goal_id1 = pb.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=end1
    )

    goal_id2 = pb.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=end2
    )

    print('start1', start1)
    print('start2', start2)

    gamma = np.array([np.linspace(start1[i], end1[i], len(t_des)) for i in range(3)]).T
    gamma2 = np.array([np.linspace(start2[i], end2[i], len(t_des)) for i in range(3)]).T

    alpha1, T1, R1 = tra_map_param(gamma)
    gamma_map = tra_map_forward(gamma, alpha1, T1, R1)

    alpha2, T2, R2 = tra_map_param(gamma2)
    gamma2_map = tra_map_forward(gamma2, alpha2, T2, R2)

    # DMP Ini
    K = 1050.0
    alpha_s = 4.0
    tol = 0.01
    dt = 0.01

    MP = DMPs_cartesian(K=K, alpha_s=alpha_s, tol=tol)
    MP2 = DMPs_cartesian_(K=K, alpha_s=alpha_s, tol=tol)

    MP.imitate_path(t_des=t_des, x_des=gamma_map)
    MP2.imitate_path(t_des=t_des, x_des=gamma2_map)

    gamma_, _, _, _ = MP.rollout()
    gamma2_, _, _, _ = MP2.rollout()
    t_des_ = np.linspace(0.0, 1.0, np.shape(gamma)[0])

    # print(gamma_[0], gamma_[-1])
    # print(gamma2_[0], gamma2_[-1])
    # pdb.set_trace()

    # Plot Ellipsoid (Need to be modified for better plot)
    # Define the ellipsoid parameters
    a = 0.2  # Semi-axis length in the x-direction
    b = 0.1  # Semi-axis length in the y-direction
    c = 0.15  # Semi-axis length in the z-direction

    # Create a grid of values for theta and phi
    theta = np.linspace(0, 2 * np.pi, 20)
    phi = np.linspace(0, np.pi, 10)
    theta, phi = np.meshgrid(theta, phi)

    # Parametric equations for the ellipsoid
    x = a * np.sin(phi) * np.cos(theta) + 0.0
    y = b * np.sin(phi) * np.sin(theta) + 1.0
    z = c * np.cos(phi) + 0.25

    MP.x_0 = gamma_[0]
    MP.x_goal = gamma_[-1]

    MP2.x_0 = gamma2_[0]
    MP2.x_goal = gamma2_[-1]

    MP.reset_state()
    MP2.reset_state()
    gamma_tra = np.array([gamma[0]])
    gamma2_tra = np.array([gamma2[0]])

    # dyn_lambda = 6.74
    # dyn_lambda = 100
    # dyn_eta = 1.14
    # dyn_beta = 0.55
    Potential_Params = [0.9220, 0.2862]
    A = Potential_Params[0]
    eta = Potential_Params[1]
    # dyn_beta = Potential_Params[2]

    # contact save axes
    axes_safe = np.array([0.15, 0.15, 0.15])

    cent_agent1, axes_agent1, velo_agent1, rotm_agent1 = getLinkFeatures(robot1)
    cent_agent2, axes_agent2, velo_agent2, rotm_agent2 = getLinkFeatures(robot2)

    for i in robot1.used_links:
        center_map = cent_agent1[f"link{i}"]
        center_map = tra_map_forward(center_map, alpha2, T2, R2)
        axes_map = axes_agent1[f"link{i}"]
        axes_map = alpha2 * axes_map
        Ellipsoid_dict_agent1[f"link{i}"] = sq_stat(center=center_map, axis=axes_map + axes_safe,
                                                    coeffs=np.array([1, 1, 1]), A=A, eta=eta)

        center_map = cent_agent2[f"link{i}"]
        center_map = tra_map_forward(center_map, alpha1, T1, R1)
        axes_map = axes_agent1[f"link{i}"]
        axes_map = alpha1 * axes_map
        Ellipsoid_dict_agent2[f"link{i}"] = sq_stat(center=center_map, axis=axes_map + axes_safe,
                                                    coeffs=np.array([1, 1, 1]), A=A, eta=eta)

    Ellipsoid_dict_agent1, Ellipsoid_dict_agent2 = Update_link_Data(cent_agent1, cent_agent2, alpha1, alpha2, T1, T2,
                                                                    R1, R2, Ellipsoid_dict_agent1, Ellipsoid_dict_agent2, num_links, robot1.used_links)  # , alpha1, alpha2, T1, T2, R1, R2)

    plt.ion()

    desired_x = [[], []]
    measured_x = [[], []]

    desired_y = [[], []]
    measured_y = [[], []]

    desired_z = [[], []]
    measured_z = [[], []]

    flag_conv = False
    flag_conv2 = False
    total_time = 0
    # pdb.set_trace()
    with imageio.get_writer('Controlled_alpha.gif', mode='I', fps=10) as writer:
        while total_time < 10.8:

            '''
            TODO: Change while loop condition.
            '''
            while total_time < 10.8:
            # while not (flag_conv and flag_conv2):
                # obst_dynamic.center=obst_dynamic.center+v_e*dt
                total_time += dt
                print("total_time: ", total_time)

                # close_flag= np.linalg.norm(MP.x - MP2.x)>0.2
                close_flag = True

                # pdb.set_trace()
                if not close_flag:
                    print("agent1 stop")

                if (not flag_conv) and close_flag:
                    # apply potential field
                    print('ready to update agent1')
                    MP.step(external_force=perturb_dynamic_agent1)
                    # flag_conv = np.linalg.norm(MP.x - MP.x_goal) < MP.tol
                    print('agent1 update')

                    # Map coordinations
                    center_agent1_origin = tra_map_inverse(np.array(MP.x), alpha1, T1, R1)

                    # store trajecory in world map
                    gamma_tra = np.append(gamma_tra, np.array([center_agent1_origin]), axis=0)
                    position_ee = gamma_tra[-1, :]
                    rotation_ee = startorientation1       #todo
                    # rotation_ee = [0.707,0,0,0.707]
                    joint_angle = robot1._calculateIK(position_ee, rotation_ee)
                    # robot1._sendPositionCommand(joint_angle)
                    robot1._resetJointState(joint_angle)
                    pb.stepSimulation()

                    # print("Robot 1 -> ", gamma_tra[-1])
                    l = robot1._getLinkState(robot1.end_effector_index)

                    desired_x[0].append(gamma_tra[-1][0])
                    measured_x[0].append(l[0][0])

                    desired_y[0].append(gamma_tra[-1][1])
                    measured_y[0].append(l[0][1])

                    desired_z[0].append(gamma_tra[-1][2])
                    measured_z[0].append(l[0][2])

                    meas1 = tra_map_forward(np.array(l[0]), alpha1, T1, R1)
                    flag_conv = np.linalg.norm(meas1 - MP.x_goal) < 0.05

                    print("distance to goal MP1:", np.linalg.norm(meas1 - MP.x_goal))

                    # call the function to obtain data_dict_agent1

                    cent_agent1, axes_agent1, velo_agent1, rotm_agent1 = getLinkFeatures(robot1)
                    cent_agent2, axes_agent2, velo_agent2, rotm_agent2 = getLinkFeatures(robot2)
                    Update_link_Data(cent_agent1, cent_agent2, alpha1, alpha2, T1, T2, R1, R2, Ellipsoid_dict_agent1, Ellipsoid_dict_agent2, num_links, robot1.used_links)  # , alpha1, alpha2, T1, T2, R1, R2)

                    for i in range(1, num_links + 1):
                        key = f'link{i}'

                if not flag_conv2:
                    # apply potential field
                    print('ready to update agent2')
                    # if total_time == 0.03:
                    #     pdb.set_trace()
                    # elif total_time >= 0.3 and total_time <= 0.32:
                    #     pdb.set_trace()
                    # elif total_time >= 0.6 and total_time <= 0.62:
                    #     pdb.set_trace()

                    r_s = 0.2
                    distance = np.linalg.norm(gamma_tra[-1] - gamma2_tra[-1])
                    distance = distance if distance < r_s else np.inf
                    MP2.step(external_force=perturb_dynamic_agent2, distance=distance)

                    print('agent2 update')
                    # flag_conv2 = np.linalg.norm(MP2.x - MP2.x_goal) < MP2.tol
                    # Map coordinations
                    center_agent2_origin = tra_map_inverse(np.array(MP2.x), alpha2, T2, R2)

                    # store trajecory in world map
                    gamma2_tra = np.append(gamma2_tra, np.array([center_agent2_origin]), axis=0)
                    position_ee = gamma2_tra[-1, :]
                    rotation_ee = startorientation2
                    # rotation_ee = [0.707,0,0,0.707]
                    joint_angle = robot2._calculateIK(position_ee, rotation_ee)
                    # robot2._sendPositionCommand(joint_angle)
                    robot2._resetJointState(joint_angle)
                    pb.stepSimulation()
                    # print("Robot 2 -> ", gamma2_tra[-1])

                    l = robot2._getLinkState(robot2.end_effector_index)

                    desired_x[1].append(gamma2_tra[-1][0])
                    measured_x[1].append(l[0][0])

                    desired_y[1].append(gamma2_tra[-1][1])
                    measured_y[1].append(l[0][1])

                    desired_z[1].append(gamma2_tra[-1][2])
                    measured_z[1].append(l[0][2])

                    meas2 = tra_map_forward(np.array(l[0]), alpha2, T2, R2)
                    flag_conv2 = np.linalg.norm(meas2 - MP2.x_goal) < 0.05
                    print("distance to goal MP2:", np.linalg.norm(meas2 - MP2.x_goal))

                    # call the function to obtain data_dict_agent1

                    cent_agent1, axes_agent1, velo_agent1, rotm_agent1 = getLinkFeatures(robot1)
                    cent_agent2, axes_agent2, velo_agent2, rotm_agent2 = getLinkFeatures(robot2)
                    Update_link_Data(cent_agent1, cent_agent2, alpha1, alpha2, T1, T2, R1, R2, Ellipsoid_dict_agent1, Ellipsoid_dict_agent2, num_links, robot1.used_links)  # , alpha1, alpha2, T1, T2, R1, R2)

                time.sleep(0.01)

                if flag_conv:
                    print("agent1 achieved goal")

                if flag_conv2:
                    print("agent2 achieved goal")

                # plotresult(writer)

    # Create a 3D plot
    measured_xyz = [measured_x, measured_y, measured_z]
    desired_xyz = [measured_x, measured_y, measured_z]
    gen_3d_plot2(plt, gamma, gamma2, measured_xyz, desired_xyz)

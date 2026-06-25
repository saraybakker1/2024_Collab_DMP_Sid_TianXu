import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

from io import BytesIO
import imageio
# from dmp.obstacle_superquadric import Obstacle_Dynamic as sq_dyn
from dmp.obstacle_superquadric import Obstacle_Static as sq_stat
from example_MM_DMP_helpers import getLinkFeatures, gen_3d_plot2, Update_link_Data, DMP_Coll
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
    dt = 0.01

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

    dmp_c = DMP_Coll(start1, end1, start2, end2)
    R1 = dmp_c.R1
    R2 = dmp_c.R2

    # define potential field
    def perturb_dynamic_agent1(x, v):
        out = 0
        for i in used_links:
            key = f'link{i}'
            # out+=Ellipsoid_dict_agent2[key].gen_external_force_rotate(x, v-alpha1*R1@velo_agent2[key], R1@rotm_agent2[key])
            out += Ellipsoid_dict_agent2[key].gen_external_force_rotate(x, R1 @ rotm_agent2[key])  # todo
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

    gamma_tra = np.array([start1])
    gamma2_tra = np.array([start2])

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
        center_map = tra_map_forward(center_map, dmp_c.alpha2, dmp_c.T2, dmp_c.R2)
        axes_map = axes_agent1[f"link{i}"]
        axes_map = dmp_c.alpha2 * axes_map
        Ellipsoid_dict_agent1[f"link{i}"] = sq_stat(center=center_map, axis=axes_map + axes_safe,
                                                    coeffs=np.array([1, 1, 1]), A=A, eta=eta)

        center_map = cent_agent2[f"link{i}"]
        center_map = tra_map_forward(center_map, dmp_c.alpha1, dmp_c.T1, dmp_c.R1)
        axes_map = axes_agent1[f"link{i}"]
        axes_map = dmp_c.alpha1 * axes_map
        Ellipsoid_dict_agent2[f"link{i}"] = sq_stat(center=center_map, axis=axes_map + axes_safe,
                                                    coeffs=np.array([1, 1, 1]), A=A, eta=eta)

    Ellipsoid_dict_agent1, Ellipsoid_dict_agent2 = Update_link_Data(cent_agent1, cent_agent2, dmp_c.alpha1, dmp_c.alpha2, dmp_c.T1, dmp_c.T2,
                                                                    dmp_c.R1, dmp_c.R2, Ellipsoid_dict_agent1, Ellipsoid_dict_agent2, num_links, robot1.used_links)  # , alpha1, alpha2, T1, T2, R1, R2)

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
                    dmp_c.MP.step(external_force=perturb_dynamic_agent1)
                    # flag_conv = np.linalg.norm(MP.x - MP.x_goal) < MP.tol
                    print('agent1 update')

                    # Map coordinations
                    center_agent1_origin = tra_map_inverse(np.array(dmp_c.MP.x), dmp_c.alpha1, dmp_c.T1, dmp_c.R1)

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

                    meas1 = tra_map_forward(np.array(l[0]), dmp_c.alpha1, dmp_c.T1, dmp_c.R1)
                    flag_conv = np.linalg.norm(meas1 - dmp_c.MP.x_goal) < 0.05

                    print("distance to goal MP1:", np.linalg.norm(meas1 - dmp_c.MP.x_goal))

                    # call the function to obtain data_dict_agent1

                    cent_agent1, axes_agent1, velo_agent1, rotm_agent1 = getLinkFeatures(robot1)
                    cent_agent2, axes_agent2, velo_agent2, rotm_agent2 = getLinkFeatures(robot2)
                    Ellipsoid_dict_agent1, Ellipsoid_dict_agent2 = Update_link_Data(cent_agent1, cent_agent2, dmp_c.alpha1, dmp_c.alpha2, dmp_c.T1, dmp_c.T2, dmp_c.R1, dmp_c.R2, Ellipsoid_dict_agent1, Ellipsoid_dict_agent2, num_links, robot1.used_links)  # , alpha1, alpha2, T1, T2, R1, R2)

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
                    dmp_c.MP2.step(external_force=perturb_dynamic_agent2, distance=distance)

                    print('agent2 update')
                    # flag_conv2 = np.linalg.norm(MP2.x - MP2.x_goal) < MP2.tol
                    # Map coordinations
                    center_agent2_origin = tra_map_inverse(np.array(dmp_c.MP2.x), dmp_c.alpha2, dmp_c.T2, dmp_c.R2)

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

                    meas2 = tra_map_forward(np.array(l[0]), dmp_c.alpha2, dmp_c.T2, dmp_c.R2)
                    flag_conv2 = np.linalg.norm(meas2 - dmp_c.MP2.x_goal) < 0.05
                    print("distance to goal MP2:", np.linalg.norm(meas2 - dmp_c.MP2.x_goal))

                    # call the function to obtain data_dict_agent1

                    cent_agent1, axes_agent1, velo_agent1, rotm_agent1 = getLinkFeatures(robot1)
                    cent_agent2, axes_agent2, velo_agent2, rotm_agent2 = getLinkFeatures(robot2)
                    # Ellipsoid_dict_agent1, Ellipsoid_dict_agent2 = Update_link_Data(cent_agent1, cent_agent2, dmp_c.alpha1, dmp_c.alpha2, dmp_c.T1, dmp_c.T2, dmp_c.R1, dmp_c.R2, Ellipsoid_dict_agent1, Ellipsoid_dict_agent2, num_links, robot1.used_links)  # , alpha1, alpha2, T1, T2, R1, R2)

                time.sleep(0.01)

                if flag_conv:
                    print("agent1 achieved goal")

                if flag_conv2:
                    print("agent2 achieved goal")

                # plotresult(writer)

    # Create a 3D plot
    measured_xyz = [measured_x, measured_y, measured_z]
    desired_xyz = [measured_x, measured_y, measured_z]
    gen_3d_plot2(plt, dmp_c.gamma, dmp_c.gamma2, measured_xyz, desired_xyz)

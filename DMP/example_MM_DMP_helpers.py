from scipy.spatial.transform import Rotation as R
import numpy as np
from dmp.dmp import DMPs_cartesian
from dmp.dmp_ import DMPs_cartesian_
from DMP_map_Rotation import rotation_matrix_from_vectors, tra_map_param, tra_map_forward, tra_map_inverse
from example_MM_DMP_helpers_plot import *

class DMP_Coll():
    def __init__(self, start1, end1, start2, end2):
        t_des = np.linspace(0.0, 1.0, 1000)

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

        self.MP = MP
        self.MP2 = MP2
        self.alpha1 = alpha1
        self.T1 = T1
        self.R1 = R1
        self.alpha2 = alpha2
        self.T2 = T2
        self.R2 = R2
        self.gamma = gamma
        self.gamma2 = gamma2


# Get global pose of each link
def getLinkFeatures(robot, nr_links=9):
    Link_gloablpos = {}
    Link_globalvel = {}
    Link_globalrot = {}
    Link_axes = {}

    if robot.used_links[-1] == 8:
        from GlobalPose import getlink_global_pose, get_semiaxes_Constants
    else:
        from GlobalPoseDinova import getlink_global_pose, get_semiaxes_Constants

    for l in robot.used_links:
    # for l in range(1, nr_links): #todo
        if l < robot.used_links[-2]:
            LinkState = robot._getLinkState(l)[0:2]
            Pos = LinkState[0]
            Ori = R.from_quat(LinkState[1])
            Rotatation_matrix = Ori.as_matrix()
            semi_axes_cons = get_semiaxes_Constants(f"link{l}")
            Link_gloablpos[f"link{l}"] = np.array(Pos)
            Link_axes[f"link{l}"] = np.array(semi_axes_cons)
            Link_globalrot[f"link{l}"] = np.array(Rotatation_matrix).T
            Link_globalvel[f"link{l}"] = np.array(robot._getLinkState(l)[6])
        elif robot.robot_name == "UR5":
            l = 10
            LinkState = robot._getLinkState(l)[0:2]
            Pos = LinkState[0]
            Ori = R.from_quat(LinkState[1])
            Rotatation_matrix = Ori.as_matrix()
            semi_axes_cons = get_semiaxes_Constants(f"link{7}")
            Link_gloablpos[f"link7"] = np.array(Pos)
            Link_axes[f"link7"] = np.array(semi_axes_cons)
            Link_globalrot[f"link7"] = np.array(Rotatation_matrix).T
            Link_globalvel[f"link7"] = np.array(robot._getLinkState(l)[6])
        elif robot.robot_name == "dinova":
            LinkState = robot._getLinkState(l)[0:2]
            Pos = LinkState[0]
            Ori = R.from_quat(LinkState[1])
            Rotatation_matrix = Ori.as_matrix()
            semi_axes_cons = get_semiaxes_Constants(f"link{l}")
            Link_gloablpos[f"link{l}"] = np.array(Pos)
            Link_axes[f"link{l}"] = np.array(semi_axes_cons)
            Link_globalrot[f"link{l}"] = np.array(Rotatation_matrix).T
            Link_globalvel[f"link{l}"] = np.array(robot._getLinkState(l)[6])
    return Link_gloablpos, Link_axes, Link_globalvel, Link_globalrot


# define potential field
def perturb_dynamic_agent1(x, v, Ellipsoid_dict_agent2, num_links, R1, rotm_agent2):
    out = 0
    for i in range(1, num_links + 1):
        key = f'link{i}'
        # out+=Ellipsoid_dict_agent2[key].gen_external_force_rotate(x, v-alpha1*R1@velo_agent2[key], R1@rotm_agent2[key])
        out += Ellipsoid_dict_agent2[key].gen_external_force_rotate(x, R1 @ rotm_agent2[key])
    # print("Perturbance 1", out)
    return out


def perturb_dynamic_agent2(x, v, Ellipsoid_dict_agent1, num_links, R2, rotm_agent1):
    out = 0
    for i in range(1, num_links + 1):
        key = f'link{i}'
        # out+=Ellipsoid_dict_agent1[key].gen_external_force_rotate(x, v-alpha2*R2@velo_agent1[key], R2@rotm_agent1[key])
        out += Ellipsoid_dict_agent1[key].gen_external_force_rotate(x, R2 @ rotm_agent1[key])
        # print('local',key, Ellipsoid_dict_agent1[key].gen_external_force_rotate(x, v-alpha2*R2@velo_agent1[key], R2@rotm_agent1[key]))
    # print("Perturbance 2", out)
    return out


# Define link data update mechanism
def Update_link_Data(cent_agent1, cent_agent2, alpha1, alpha2,
                     T1, T2, R1, R2,
                     Ellipsoid_dict_agent1, Ellipsoid_dict_agent2,
                     num_links, used_links=None):
    if used_links is None:
        used_links = range(1, num_links + 1)
    # print('Arm Info Update')
    for i in used_links:
        key = f"link{i}"

        # center_map=cent_agent1[f"link{i}"]
        center_map = cent_agent1[key]
        center_map = tra_map_forward(center_map, alpha2, T2, R2)
        Ellipsoid_dict_agent1[key].center = center_map

        center_map = cent_agent2[key]
        center_map = tra_map_forward(center_map, alpha1, T1, R1)
        Ellipsoid_dict_agent2[key].center = center_map
    return Ellipsoid_dict_agent1, Ellipsoid_dict_agent2
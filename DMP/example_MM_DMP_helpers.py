from scipy.spatial.transform import Rotation as R
import numpy as np
from GlobalPose import getlink_global_pose, get_semiaxes_Constants
from DMP_map_Rotation import rotation_matrix_from_vectors, tra_map_param, tra_map_forward, tra_map_inverse
from example_MM_DMP_helpers_plot import *

# Get global pose of each link
def getLinkFeatures(robot):
    Link_gloablpos = {}
    Link_globalvel = {}
    Link_globalrot = {}
    Link_axes = {}

    for l in range(1, 8): #todo
        if l < 7:
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
                     num_links):
    print('Arm Info Update')
    for i in range(1, num_links + 1):
        key = f"link{i}"

        # center_map=cent_agent1[f"link{i}"]
        center_map = cent_agent1[key]
        center_map = tra_map_forward(center_map, alpha2, T2, R2)
        Ellipsoid_dict_agent1[key].center = center_map

        center_map = cent_agent2[key]
        center_map = tra_map_forward(center_map, alpha1, T1, R1)
        Ellipsoid_dict_agent2[key].center = center_map
    return Ellipsoid_dict_agent1, Ellipsoid_dict_agent2
import numpy as np
from pyquaternion import Quaternion 
import pdb

Ur5e_dim = {
    'link1':[0.149,0.1625], #Stored as [radius, length]
    'link2':[0.149,0.4250],
    'link3':[0.135,0.3920],
    'link13':[0.135,0.1330],
    'link14':[0.135,0.0997],
    'link15':[0.135,0.0417],
    'link16':[0.153,0.075],
    'link17':[0.153,0.075],
    'link18':[0.153,0.075]
}


Ur5e_global_pts = {
    'link1':[], #Stored as [radius, length]
    'link2':[],
    'link3':[],
    'link13':[],
    'link14':[],
    'link15':[],
    'link16': [],
    'link17': [],
    'link18': []
}

Ur5e_global_vel = {
    'link1':[], #Stored as [radius, length]
    'link2':[],
    'link3':[],
    'link13':[],
    'link14':[],
    'link15':[],
    'link16': [],
    'link17': [],
    'link18': []
}


def getlink_global_pose(positions=None, orientations=None, velocities=None):

    for i,k in enumerate(list(Ur5e_dim.keys())):
        ori = orientations[i+1]
        global_ori = [ori[3], *ori[:3]]
        global_ori = Quaternion(global_ori)
        rotation_matrix = global_ori.rotation_matrix

        radius = Ur5e_dim[k][0]/2
        length = Ur5e_dim[k][1]/2
        # local_points = np.array([[length, 0, 0],
        #                 [-length, 0, 0],
        #                 [0, radius, 0],
        #                 [0, -radius, 0],
        #                 [0, 0, radius],
        #                 [0, 0, -radius]]).T
        local_points = np.array([[radius, 0, 0],
                        [-radius, 0, 0],
                        [0, radius, 0],
                        [0, -radius, 0],
                        [0, 0, length],
                        [0, 0, -length]]).T
        var = rotation_matrix @ local_points + np.array(positions[i+1]).reshape([3,1])
        var = np.hstack((var,np.array(positions[i+1]).reshape(3,1)))
        Ur5e_global_pts[k] = var.T  


        Ur5e_global_vel[k] = velocities[i+1] 
        # pdb.set_trace()
    
    return Ur5e_global_pts, Ur5e_global_vel

def get_semiaxes_Constants(key):
    radius = Ur5e_dim[key][0]/2
    length = Ur5e_dim[key][1]/2
    a = radius
    b = radius
    c = length
    semi_axes_cons = [a,b,c]
    
    return semi_axes_cons


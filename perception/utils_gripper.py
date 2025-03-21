import sys

import numpy
import open3d

sys.path.append('/home/albert/github/robopack')

import numpy as np
import open3d as o3d

from perception.utils_pc import pos_quat_to_matrix, pos_rot_to_matrix, generate_point_cloud, sample_points_from_mesh, \
    farthest_point_sampling_dgl_pc
from transforms3d.quaternions import *

"""
For all methods, note that  bubble 1 and finger 1 refers to the left bubble and finger 
    *looking from the robot's* frame when the end effector pose is [0, 1, 0, 0] (wxyz).
"""


bubble1_cam_pos = np.array([-0.000041, 0.096359, -0.07268])
bubble1_cam_orient = np.array([0.398504, 0.584119, -0.398504, -0.584119])  # wxyz
bubble1_cam_mat = pos_quat_to_matrix(bubble1_cam_pos, bubble1_cam_orient, True)

finger1_pos = np.array([0, 0, 0])        # location of the finger joint
finger1_orient = np.array([0.0, -1.0, 0.0, 0.0])
finger1_mat = pos_quat_to_matrix(finger1_pos, finger1_orient, True)


def get_bubble1_to_finger1_transform():
    """
    Computes the transformation matrix from the bubble1 to finger 1,
    based on locations in blender 3D.
    :return:
    """
    bubble1_to_finger1 = np.linalg.inv(finger1_mat) @ bubble1_cam_mat  # world to finger * cam to world
    return bubble1_to_finger1


def get_finger1_to_ee_transform(finger_dist):
    """
    Returns the transformation matrix from the finger 1 to EE, given the distance between the finger
    :param finger_dist: the distance between fingers, a negative value
    :return: the transformation matrix
    """
    finger_dist = -abs(finger_dist)
    PHYSICAL_DIST_BETWEEN_FINGERS = -0.025       # measured by ruler
    finger_dist = finger_dist + PHYSICAL_DIST_BETWEEN_FINGERS
    return pos_quat_to_matrix(np.array([0, finger_dist/2, 0]), np.array([1, 0, 0, 0]), True)


# some measurement results in Blender
# b1_center_to_blender = pos_quat_to_matrix([0.00183551, 0.00258492, 0.00269504],
#                                           [11.435, 1.085, -0.012, -0.285])
b1_finger_to_blender = pos_quat_to_matrix(finger1_pos, finger1_orient)


# def get_bubble1_center_to_finger1():
#     """
#     Computes the transformation matrix from bubble 1 center (defined in .stl file) to the finger 1.
#     :return: the transformation matrix
#     """
#     return np.linalg.inv(b1_finger_to_blender) @ b1_center_to_blender


def get_blender_origin_to_finger1():
    """
    Computes the transformation matrix from the blender origin to the bubble 1 finger.
    The reason you need this is that, when we export a mesh file in Blender, the world origin will
    be the file center. If you want to visualize this 3d model placed in Open3D, any transformations will
    be applied to the center of the model, which is actually the origin of the Blender world. You need to
    make sure the applied transformation is with respect to the Blender world origin, instead of
    the actual mesh center, i.e., apply the transformation matrix returned by this function first.
    :return: the transformation matrix
    """
    return np.linalg.inv(b1_finger_to_blender)


rotate_around_z_180 = np.array(
    [[-1, 0, 0, 0],
     [0, -1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]]
)


def get_origin_ee_point_cloud(ee_pos):
    """
    Returns two cuboids to represent the origin and the end effector,
    as well as a z = 0 plane for visualization purpose
    :param ee_pos: the pose of the end effector, where the first three values are x y z
    :return: the three point clouds corresponding to the plane, the origin and ee
    """
    N_POINTS = 500
    plane = generate_point_cloud(N_POINTS * 10, [0.25, 0.8], [-0.5, 0.5])

    ee = generate_point_cloud(1000, [ee_pos[0] - 0.02, ee_pos[0] + 0.02],
                              [ee_pos[1] - 0.02, ee_pos[1] + 0.02],
                              [ee_pos[2] - 0.02, ee_pos[2] + 0.02])

    origin = generate_point_cloud(1000, [0. - 0.02, 0. + 0.02],
                                  [0. - 0.02, 0. + 0.02],
                                  [0. - 0.02, 0. + 0.02])

    return plane, ee, origin


def get_gripper_models_to_robot(ee_pos, ee_orient, gripper_dist):
    """
    Computes the transformation matrix from the two gripper models (exported from Blender) to the robot.
    :param ee_pos: end effector xyz
    :param ee_orient: end effector wxyz, note that it's NOT xyzw
    :param gripper_dist: distance between the fingers
    :return: the two transformation matrices
    """
    # computes finger1 to robot
    ee_to_robot = pos_quat_to_matrix(ee_pos, ee_orient, True)
    finger1_to_ee = get_finger1_to_ee_transform(gripper_dist)

    # computes blender center to robot
    bubble1_model_to_robot = ee_to_robot @ finger1_to_ee @ get_blender_origin_to_finger1()
    bubble2_model_to_robot = ee_to_robot @ rotate_around_z_180 @ finger1_to_ee @ np.linalg.inv(b1_finger_to_blender)
    return bubble1_model_to_robot, bubble2_model_to_robot


def get_bubble_cameras_to_robot(ee_pos, ee_orient, gripper_dist):
    """
    Returns the transformation matrix from the two bubbles to the robot.
    :param ee_pos: xyz
    :param ee_orient: wxyz, note it's not xyzw
    :param gripper_dist: distance between the gripper fingers, directly returned by deoxys
    :return: the two transformation matrices
    """
    # ee_to_robot = pos_quat_to_matrix(ee_pos, ee_orient, True)
    # finger1_to_ee = get_finger1_to_ee_transform(gripper_dist)
    # bubble1_to_finger1 = get_bubble1_to_finger1_transform() @ get_blender_origin_to_bubble1_center()
    # bubble1_to_robot = ee_to_robot @ finger1_to_ee @ bubble1_to_finger1
    # bubble2_to_robot = ee_to_robot @ rotate_around_z_180 @ finger1_to_ee @ bubble1_to_finger1

    bubble1_model_to_robot, bubble2_model_to_robot = get_gripper_models_to_robot(ee_pos, ee_orient, gripper_dist)
    bubble1_to_robot = bubble1_model_to_robot @ bubble1_cam_mat
    bubble2_to_robot = bubble2_model_to_robot @ bubble1_cam_mat
    return bubble1_to_robot, bubble2_to_robot


def get_gripper_mesh(ee_pos, ee_orient, gripper_dist):
    gripper_model_path = '/home/albert/Data/gripper-connector-fixed-july10.ply'
    model1 = o3d.io.read_triangle_mesh(gripper_model_path)
    model2 = o3d.io.read_triangle_mesh(gripper_model_path)

    bubble1_model_to_robot, bubble2_model_to_robot = get_gripper_models_to_robot(ee_pos, ee_orient, gripper_dist)
    model1.transform(bubble1_model_to_robot)
    model2.transform(bubble2_model_to_robot)

    return model1, model2


def get_rod_points_from_model(rod_model_path, ee_pos, n_points=20, seed=None):
    assert n_points < 1000, 'does not support n_points > 1000'
    rod_model = o3d.io.read_triangle_mesh(rod_model_path)
    rod_model_points = sample_points_from_mesh(rod_model, 1000, seed=seed)
    rod_model_points = farthest_point_sampling_dgl_pc(rod_model_points, n_points, seed=seed)
    rod_model_points = np.asarray(rod_model_points.points)
    rod_model_points = rod_model_points + \
                       (np.array([0.49805392, 0.01808551, 0.12164542]) - np.array([0.5, 0., 0.325])) + \
                       ee_pos
    return rod_model_points

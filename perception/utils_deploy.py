import cv2
import numpy as np
import open3d as o3d
import open3d as o3d

from config.task_config import intrinsic_matrices
from perception.utils_gripper import get_bubble_cameras_to_robot, get_rod_points_from_model, get_origin_ee_point_cloud
from perception.utils_pc import rgbd_feat_to_point_cloud, remove_distant_points, random_select_rows, denoise_by_cluster, \
    farthest_point_sampling_dgl, find_indices, project_points, convert_pc_optical_color_to_link_frame, xyz_to_pc


def visualize_first_frame(first_frame, ee_pose):
    points = []
    for topic, array in first_frame.items():
        if topic != 'bubble_pcs' and topic != 'inhand_object_pcs':
            continue
        array = np.array(array)
        if len(array.shape) == 3:
            for i in range(array.shape[0]):
                points.append(xyz_to_pc(array[i]))
        elif len(array.shape) == 2:
            points.append(xyz_to_pc(array))
        else:
            raise NotImplementedError(f'the array shape is {array.shape}')
    world_objects = list(get_origin_ee_point_cloud(ee_pose))
    o3d.visualization.draw_geometries(points + world_objects)
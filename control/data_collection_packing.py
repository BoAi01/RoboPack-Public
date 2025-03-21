import random
import sys

sys.path.append('/home/albert/github/robopack')

from utils_deoxys import init_franka_interface

import numpy as np
import pdb

import rclpy
from rclpy.executors import MultiThreadedExecutor

import threading

from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.config_utils import (add_robot_config_arguments,
                                       get_default_controller_config)
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.log_utils import get_deoxys_example_logger

from control.motion_utils import *
from perception.utils_sam import *
from perception.utils_cv import *
from perception.utils_ros import *
from perception.utils_pc import *
from utils_general import *

from perception.camera_subscriber import CameraSubscriber

from config.task_config import *

from robot_state_publisher import RobotStatePublisherNode

from input_utils import input2action


logger = get_deoxys_example_logger()
controller_cfg = YamlConfig(
        os.path.join(config_root, "osc-position-controller.yml")
    ).as_easydict()


top_left = np.array([0.623, 0.178, 0.228])
top_right = np.array([0.623, -0.178, 0.228])
bottom_left = np.array([0.329, 0.178, 0.228])
bottom_right = np.array([0.329, -0.178, 0.228])
workspace_center = np.array([0.45, 0, 0])

# 0.30 is exactly the height needed to grasp the rod on table

pushing_ee_height = 0.32
N_POINTS_PER_OBJ = 500
time_to_collect = 60 * 25       # the effective time is only slightly better than half
data_root = ("/media/albert/ExternalH"
             "DD/bubble_data/v11_0828")

sugar_box_color = np.array([0.45904095, 0.42684464, 0.28072357])
blue_rod_color = np.array([0.03888071, 0.17960485, 0.27454443])
colors = {
    'box': sugar_box_color,
    'rod': blue_rod_color
}


def reconstruct_single_object_and_save(cam0_pc, cam1_pc, model, predictor, target):
    assert target in ['box', 'rod']

    cam0_rgb, cam0_depth = point_cloud_to_rgbd(cam0_pc, intrinsic_matrices['cam_0'], 720, 1280)
    cam1_rgb, cam1_depth = point_cloud_to_rgbd(cam1_pc, intrinsic_matrices['cam_1'], 720, 1280)

    cam0_masks, boxes_filt, pred_phrases = predict(model, predictor, cam0_rgb, "objects")
    cam1_masks, boxes_filt, pred_phrases = predict(model, predictor, cam1_rgb, "objects")

    cam0_masks = remove_redundant_images(cam0_masks.squeeze(1), threshold=100)
    cam1_masks = remove_redundant_images(cam1_masks.squeeze(1), threshold=100)

    cam0_pcs = get_point_cloud_of_every_object(cam0_masks, cam0_rgb, cam0_depth, intrinsics_depth['cam_0'],
                                               extrinsics['cam_0'])
    cam1_pcs = get_point_cloud_of_every_object(cam1_masks, cam1_rgb, cam1_depth, intrinsics_depth['cam_1'],
                                               extrinsics['cam_1'])
    object_pcs = merge_multiview_object_pcs(cam0_pcs, cam1_pcs)

    # get the largest object
    # max_cloud = object_pcs[0]
    # for pc in object_pcs:
    #     if np.asarray(pc.points).shape[0] < np.asarray(max_cloud.points).shape[0]:
    #         max_cloud = pc
    #
    # object_pcs = [max_cloud]

    # find the object by color
    closest_pc = object_pcs[0]
    for pc in object_pcs:
        closest_pc_color = np.asarray(closest_pc.colors).mean(0)
        pc_color = np.asarray(pc.colors).mean(0)
        if np.linalg.norm(pc_color - colors[target]) < np.linalg.norm(closest_pc_color - colors[target]):
            closest_pc = pc
    object_pcs = [closest_pc]

    # o3d.visualization.draw_geometries(object_pcs)
    # import pdb
    # pdb.set_trace()

    denoised_sampled_pcs = []
    for i, object_pc in enumerate(object_pcs):
        object_pc += fill_point_cloud_by_downward_projection(object_pc)
        # object_pc2 = denoise_pc_by_stats(object_pc1, denoise_depth=2)
        # denoised_sampled_pc = sample_points_from_pc(object_pc, n_points=N_POINTS_PER_OBJ, ratio=None,
        #                                             mesh_fix=True)
        # object_pc = farthest_point_sampling_dgl(object_pc, N_POINTS_PER_OBJ)
        denoised_sampled_pcs.append(object_pc)

    # assert len(denoised_sampled_pcs) == 1
    # o3d.visualization.draw_geometries(denoised_sampled_pcs)
    # pdb.set_trace()

    # save
    data_dict = {
        'cam0_rgb': cam0_rgb,
        'cam1_rgb': cam1_rgb,
        'cam0_depth': cam0_depth,
        'cam1_depth': cam1_depth,
        'object_pc': np.concatenate([get_pc_xyz_color_array(pc) for pc in object_pcs], axis=0),
        'sampled_points': np.concatenate([get_pc_xyz_color_array(pc)
                                          for pc in denoised_sampled_pcs], axis=0)
    }

    return data_dict


def filter_point_cloud_by_height(point_cloud, min_height, max_height):
    # Convert the Open3D point cloud to a numpy array
    points = np.asarray(point_cloud.points)

    # Extract the z coordinates (heights) from the point cloud
    heights = points[:, 2]

    # Find the indices of points that are within the height range
    height_mask = np.logical_and(heights >= min_height, heights <= max_height)

    # Create a new point cloud containing only the points within the height range
    filtered_point_cloud = point_cloud.select_by_index(np.where(height_mask)[0])

    return filtered_point_cloud


def get_box_point_cloud(cam0_pc, cam1_pc):
    min_height = 0.005
    max_height = 0.05

    pcs = []
    for i, pc in enumerate([cam0_pc, cam1_pc]):
        pc = project_pc(convert_pc_optical_color_to_link_frame(pc, i), extrinsics[f'cam_{i}'])
        pc = extract_workspace_pc(pc, xrange, yrange, zrange)
        pcs.append(pc)

    object_pc = filter_point_cloud_by_height(pcs[0] + pcs[1], min_height, max_height)

    return fill_point_cloud_by_downward_projection(object_pc) + object_pc


def compute_k_b(point1, point2):
    delta = point1 - point2
    k = delta[1]
    b = point1[1] - k * point1[0]
    return k, b


def find_point_on_line(point_A, point_B, distance):
    """
    Find a point on the line connecting A and B and is at a specific distance
    to one of the points.
    :param point_A: starting point
    :param point_B: end point
    :param distance: distance to the end point along the line AB
    :return: the coordinate for point C
    """
    # Convert the points to NumPy arrays for vector operations
    A = np.array(point_A)
    B = np.array(point_B)

    # Calculate the direction vector from A to B
    direction_vector = B - A

    # Normalize the direction vector to get the unit direction vector
    unit_direction_vector = direction_vector / np.linalg.norm(direction_vector)

    # Calculate the coordinates of the third point C at distance 'd' from point A
    C = B + distance * unit_direction_vector

    return C


def sample_xy_in_range(xrange, yrange):
    """
    Randomly samples a coordinate (x, y) in the given range
    :param xrange: range of x
    :param yrange: range of y
    :return: (x, y)
    """
    x = random.uniform(*xrange)
    y = random.uniform(*yrange)
    return [x, y]


def is_in_workspace(pos):
    return 0.20 < pos[0] < 0.70 and -0.40 < pos[1] < 0.40 and 0.15 < pos[2] < 0.60


high_z = 0.5        # so that the rod is slightly above the sugar box


def move_to_safely(robot_interface, current_pos, target_pos, **args):
    current_pos[-1] = high_z
    target_high_pos = target_pos.copy()
    target_high_pos[-1] = high_z
    position_only_gripper_move_to(robot_interface, current_pos, **args)
    position_only_gripper_move_to(robot_interface, target_high_pos, **args)
    position_only_gripper_move_to(robot_interface, target_pos, **args)


def grasp_vertical_rod(robot_interface, rod_pc):
    xy_center = rod_pc.get_center()
    position_only_gripper_move_to(robot_interface, np.array([xy_center[0], xy_center[1], 0.30]))
    close_gripper(robot_interface)


def is_coordinate_within_point_cloud(point_cloud, coordinate, threshold=1e-6):
    # Convert the 3D coordinate to an Open3D point cloud with a single point
    single_point_cloud = o3d.geometry.PointCloud()
    single_point_cloud.points = o3d.utility.Vector3dVector([coordinate])

    # Compute the distance to the closest point in the point cloud
    distances = point_cloud.compute_point_cloud_distance(single_point_cloud)

    # Get the minimum distance
    min_distance = np.min(distances)

    # o3d.visualization.draw_geometries([point_cloud, single_point_cloud])

    # Check if the minimum distance is below the threshold
    return min_distance < threshold


def main(args):
    global pushing_ee_height

    robot_interface = init_franka_interface()
    ee_home_pose = np.array([[0.35], [0], [0.50]])  # back off and stay high to make space
    position_only_gripper_move_to(
        robot_interface, ee_home_pose, grasp=args.no_init
    )

    rclpy.init(args=None)
    camera_node = CameraSubscriber()  # listen for two cameras

    robot_node = RobotStatePublisherNode(robot_interface)
    device = SpaceMouse(vendor_id=9583, product_id=50734)
    device.start_control()

    executor = MultiThreadedExecutor()
    executor.add_node(camera_node)
    executor.add_node(robot_node)
    background_thread = threading.Thread(target=lambda: executor.spin())
    background_thread.start()
    time.sleep(2)  # wait for the image to be received

    assert camera_node.get_last_reading() is not None, 'no messages received from the cameras'

    if not args.no_init:
        user_input = None
        while user_input != 'c':
            user_input = input("Place in-hand object on the table. "
                               "Enter c to start scanning: ")

        object_data_dir = create_sequential_folder(data_root)
        cam0_pc, cam1_pc = [decode_pointcloud_msg(msg) for msg in camera_node.get_last_reading()[:2]]
        # cam0_rgb, cam1_rgb, cam0_depth, cam1_depth = [bridge.imgmsg_to_cv2(x, desired_encoding='passthrough')
        #                                               for x in camera_node.get_last_reading()[-4:]]
        # cam0_pc = construct_pointcloud_from_rgbd(cam0_rgb, cam0_depth, intrinsics_depth['cam_0'], None)
        # cam1_pc = construct_pointcloud_from_rgbd(cam1_rgb, cam1_depth, intrinsics_depth['cam_1'], None)
        # o3d.visualization.draw_geometries([cam0_pc])
        model, predictor = load_models()
        model_data_path = os.path.join(object_data_dir, 'inhand_obj.h5')
        model_data_dict = reconstruct_single_object_and_save(cam0_pc, cam1_pc, model, predictor, 'rod')

        # start collecting data
        user_input = None
        while user_input != 'c':
            user_input = input("Next grab the tool. Press c when done: ")

        # grasp_vertical_rod(robot_interface, xyz_colors_to_pc(model_data_dict['object_pc']))
        ee_states, _ = robot_node.get_ee_states()
        fixed_z = ee_states[-1] + 0.02      # 2 cm higher, in case gripper height drops

        # record ee data
        from calibration.publish_joint_state_ros2 import get_joint_states, get_ee_states
        ee_pos, ee_rot = get_ee_states(robot_interface)
        model_data_dict['ee_states'] = np.array(ee_pos + ee_rot)
        joint_pos, joint_vel, gripper_state = get_joint_states(robot_interface)
        position = joint_pos + [gripper_state]
        model_data_dict['joint_states'] = np.array(position)
        save_dictionary_to_hdf5(model_data_dict, model_data_path)

        print(f'EE state is {ee_states}. Please check correctness')
        o3d.visualization.draw_geometries([xyz_colors_to_pc(model_data_dict['object_pc'])])

        pdb.set_trace()

    else:
        pass

    try:
        while True:
            # time.sleep(1 / 10)  # doesnt work
            start_time = time.time_ns()
            action, grasp = input2action(
                device=device,
                controller_type=controller_type,
            )
            if action is None:
                break

            # set unused orientation dims to 0
            if controller_type == "OSC_YAW":
                action[3:5] = 0.0
            elif controller_type == "OSC_POSITION":
                action[3:6] = 0.0
            elif controller_type == "JOINT_IMPEDANCE":
                action[:7] = robot_interface.last_q.tolist()
            logger.info(action)

            action[:3] = np.clip(action[:3], -0.3, 0.3)
            robot_interface.control(
                controller_type=controller_type,
                action=action,
                controller_cfg=controller_cfg,
            )

            end_time = time.time_ns()
            print(f"Time profile: {(end_time - start_time) / 10 ** 9}")

    except KeyboardInterrupt:
        print('keyboard interrupted')

    robot_interface.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="dynamics")
    parser.add_argument("--no_init", default=0, type=int, help="Skip init routine")
    args = parser.parse_args()
    main(args)

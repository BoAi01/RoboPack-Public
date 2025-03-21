import random
import sys
import pdb

sys.path.append('/home/albert/github/robopack')

from utils_deoxys import init_franka_interface

import numpy as np
import pdb

import rclpy
from rclpy.executors import MultiThreadedExecutor

import threading

from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig, transform_utils
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

from control.robot_state_publisher import RobotStatePublisherNode
from control.input_utils import input2action


logger = get_deoxys_example_logger()
controller_cfg = YamlConfig(
        os.path.join(config_root, "osc-position-controller.yml")
    ).as_easydict()


is_testing = not False

def classify_tactile(b1_tactile, b2_tactile, threshold=1.5):
    assert threshold > 0

    b1_mean, b2_mean = b1_tactile.reshape(-1, 2).mean(0), b2_tactile.reshape(-1, 2).mean(0)
    diff = b1_mean - b2_mean
    print(f'diff, b1 mean, b2 mean: {diff, b1_mean, b2_mean}')

    if b1_mean[0] < -2 and b2_mean[0] < -2:
        return "Encountering obstacle"
    if diff[0] > threshold:
        return "move to left"
    elif diff[0] < -threshold:
        return "move to right"
    else:
        return "fine"


def osc_move(robot_interface, controller_type, controller_cfg, target_pose, num_steps, grasp=False):
    target_pos, target_quat = target_pose

    assert isinstance(target_pos, np.ndarray) and target_pos.shape == (3, 1), f"Target pos shape is {target_pos.shape}"

    # print(target_quat)
    # import pdb
    # pdb.set_trace()
    target_axis_angle = transform_utils.quat2axisangle(target_quat)
    current_rot, current_pos = robot_interface.last_eef_rot_and_pos

    for _ in range(num_steps):
        current_pose = robot_interface.last_eef_pose
        current_pos = current_pose[:3, 3:]
        current_rot = current_pose[:3, :3]
        current_quat = transform_utils.mat2quat(current_rot)
        if np.dot(target_quat, current_quat) < 0.0:
            current_quat = -current_quat
        quat_diff = transform_utils.quat_distance(target_quat, current_quat)
        current_axis_angle = transform_utils.quat2axisangle(current_quat)
        axis_angle_diff = transform_utils.quat2axisangle(quat_diff)
        action_pos = (target_pos - current_pos).flatten() * 10
        action_axis_angle = axis_angle_diff.flatten() * 1
        action_pos = np.clip(action_pos, -0.3, 0.3)
        action_axis_angle = np.clip(action_axis_angle, -0.2, 0.2)

        action = action_pos.tolist() + action_axis_angle.tolist() + [2 * int(grasp) - 1]
        # logger.info(f"Axis angle action {action_axis_angle.tolist()}")
        # print(np.round(action, 2))
        robot_interface.control(
            controller_type=controller_type,
            action=action,
            controller_cfg=controller_cfg,
        )

    return action


def move_to_target_pose(
    robot_interface,
    controller_type,
    controller_cfg,
    target_delta_pose,
    num_steps,
    num_additional_steps,
    interpolation_method,
    grasp=False,
):
    while robot_interface.state_buffer_size == 0:
        logger.warn("Robot state not received")
        time.sleep(0.5)

    target_delta_pos, target_delta_axis_angle = (
        target_delta_pose[:3],
        target_delta_pose[3:],
    )
    current_ee_pose = robot_interface.last_eef_pose
    current_pos = current_ee_pose[:3, 3:]
    current_rot = current_ee_pose[:3, :3]
    current_quat = transform_utils.mat2quat(current_rot)
    current_axis_angle = transform_utils.quat2axisangle(current_quat)

    target_pos = np.array(target_delta_pos).reshape(3, 1) + current_pos
    logger.info(f"Current axis angle {current_axis_angle}")

    target_axis_angle = np.array(target_delta_axis_angle) + current_axis_angle

    logger.info(f"Before conversion {target_axis_angle}")
    target_quat = transform_utils.axisangle2quat(target_axis_angle)
    target_pose = target_pos.flatten().tolist() + target_quat.flatten().tolist()

    if np.dot(target_quat, current_quat) < 0.0:
        current_quat = -current_quat
    target_axis_angle = transform_utils.quat2axisangle(target_quat)
    logger.info(f"After conversion {target_axis_angle}")
    current_axis_angle = transform_utils.quat2axisangle(current_quat)

    start_pose = current_pos.flatten().tolist() + current_quat.flatten().tolist()

    # print(target_pos, target_quat)
    # import pdb
    # pdb.set_trace()
    osc_move(
        robot_interface,
        controller_type,
        controller_cfg,
        (target_pos, target_quat),
        num_steps,
        grasp=grasp,
    )


high_z = 0.5
def move_to_safely(robot_interface, current_pos, target_pos, **args):
    current_pos[-1] = high_z
    target_high_pos = target_pos.copy()
    target_high_pos[-1] = high_z
    position_only_gripper_move_to(robot_interface, current_pos, **args)
    position_only_gripper_move_to(robot_interface, target_high_pos, **args)
    position_only_gripper_move_to(robot_interface, target_pos, **args)

def move_to_safely_yaw(robot_interface, current_pos, target_pos, **args):
    current_pos[-1] = high_z
    target_high_pos = target_pos.copy()
    target_high_pos[-1] = high_z
    position_only_gripper_move_to_yaw(robot_interface, current_pos, **args)
    position_only_gripper_move_to_yaw(robot_interface, target_high_pos, **args)
    position_only_gripper_move_to_yaw(robot_interface, target_pos, **args)


def position_only_gripper_move_to_yaw(robot_interface, position, num_steps=100, grasp=False):
    current_pose = robot_interface.last_eef_pose
    yaw_config = get_default_controller_config("OSC_YAW")
    current_pos = current_pose[:3, 3:]
    current_rot = current_pose[:3, :3]
    current_quat = transform_utils.mat2quat(current_rot)
    osc_move(robot_interface=robot_interface,
             controller_type="OSC_YAW",
             controller_cfg=yaw_config,
             target_pose=(position, current_quat),
             num_steps=num_steps,
             grasp=grasp)

def rotate_gripper(robot_interface, angle_to_rotate):
    """
    :param robot_interface:
    :param angle_to_rotate: angle in radians
    :return:
    """
    # DO NOT REDEFINE controller_config in this function!
    move_to_target_pose(
        robot_interface,
        "OSC_YAW",
        get_default_controller_config("OSC_YAW"),
        target_delta_pose=[0.0, 0.0, 0.0, 0.0, angle_to_rotate * 2, 0.0],
        num_steps=80,
        num_additional_steps=40,
        interpolation_method="linear",
    )


def move_by_jump(robot_interface, yaw_config, target):
    move_to_target_pose(
        robot_interface,
        "OSC_YAW",
        yaw_config,
        [0, 0, 0.07, 0, 0, 0],
        num_steps=30,
        num_additional_steps=40,
        interpolation_method="linear",
        grasp=True,
    )
    target_xy = target.copy()
    target_xy[2] = 0    # only change x and y
    move_to_target_pose(
        robot_interface,
        "OSC_YAW",
        yaw_config,
        target_xy,
        num_steps=30,
        num_additional_steps=40,
        interpolation_method="linear",
        grasp=True,
    )
    move_to_target_pose(
        robot_interface,
        "OSC_YAW",
        yaw_config,
        [0, 0, -0.07, 0, 0, 0],
        num_steps=30,
        num_additional_steps=40,
        interpolation_method="linear",
        grasp=True,
    )


def get_sam_masks(sam_predictor, rgb):
    masks = sam_predictor.generate(rgb)
    plain_masks = [x['segmentation'] for x in masks]
    plain_masks = remove_redundant_images(plain_masks, threshold=500)

    # WARNING: plt causes processes using cv2 (e.g., tactile subscriber) to stuck
    # plt.figure(figsize=(10, 10))
    # plt.imshow(rgb)
    #
    # for mask in plain_masks:
    #     show_mask(mask, plt.gca(), random_color=True)

    return plain_masks


def extract_pcs_from_masks(rgb, depth, masks, cam_name):
    cam_pcs = []
    for mask in masks:
        # each mask is of shape (1, h, w)
        masked_rgb = crop_out_masked_region(rgb, mask)
        masked_depth = crop_out_masked_region(depth, mask)
        constructed_pc = construct_pointcloud_from_rgbd(masked_rgb, masked_depth, intrinsics_depth[cam_name])
        constructed_pc = project_pc(convert_pc_optical_to_link_frame(constructed_pc), extrinsics[cam_name])
        constructed_pc = extract_workspace_pc(constructed_pc, xrange=[0.37, 0.60], yrange=[-0.09, 0.25], zrange=[0.01, 0.2])
        constructed_pc = denoise_pc_by_stats(constructed_pc, denoise_depth=2)
        if constructed_pc.has_points():
            cam_pcs.append(constructed_pc)
    return cam_pcs


def get_occupancy_grid(point_clouds):
    def count_occurrences(data_list):
        value_counts = {}

        for value in data_list:
            if value in value_counts:
                value_counts[value] += 1
            else:
                value_counts[value] = 1

        return value_counts

    # Parameters
    grid_resolution = 0.03  # Adjust as needed
    xrange = [0.37, 0.60]
    yrange = [-0.09, 0.25]

    # Load point clouds
    # point_clouds = pcs  # Replace this with your list of point clouds

    # Calculate grid dimensions based on specified ranges
    grid_dim_x = int(np.ceil((xrange[1] - xrange[0]) / grid_resolution))
    grid_dim_y = int(np.ceil((yrange[1] - yrange[0]) / grid_resolution))

    # Create a 3D list to record object indices in each cell
    cell_object_indices = [[[] for _ in range(grid_dim_x)] for _ in range(grid_dim_y)]

    # Process each point cloud
    for pc_idx, pc in enumerate(point_clouds):
        # Convert point cloud to numpy array
        points = np.asarray(pc.points)

        # Project points onto 2D plane and calculate grid indices
        projected_points = points[:, :2]
        grid_indices_x = np.floor((projected_points[:, 0] - xrange[0]) / grid_resolution).astype(int)
        grid_indices_y = np.floor((projected_points[:, 1] - yrange[0]) / grid_resolution).astype(int)

        # Update cell_object_indices
        for i in range(len(projected_points)):
            x_idx = grid_indices_x[i]
            y_idx = grid_indices_y[i]
            if 0 <= x_idx < grid_dim_x and 0 <= y_idx < grid_dim_y:
                cell_object_indices[y_idx][x_idx].append(pc_idx)

    # Create an empty occupancy grid
    occupancy_grid = np.zeros((grid_dim_y, grid_dim_x), dtype=int)

    # Check for mixture of points from different objects
    for y_idx in range(grid_dim_y):
        for x_idx in range(grid_dim_x):
            object_indices = cell_object_indices[y_idx][x_idx]
            occurences = count_occurrences(object_indices)
            if len(occurences) >= 2 and min(occurences.values()) >= 2:
                occupancy_grid[y_idx, x_idx] = 1

    # Now, occupancy_grid is your 2D numpy array representing the occupancy grid for the specified region

    # flip the direction of the y axis
    occupancy_grid = np.flipud(occupancy_grid)

    return occupancy_grid


def get_insertion_points(gap_mask):
    object_y_size = np.round(14 / 3)
    object_x_size = np.round(5 / 3)
    top_k = 5
    candidate_ys, candidate_xs = np.where(gap_mask == 1)  # H, W
    H, W = gap_mask.shape

    # index = np.random.randint(0, y.shape[0], size=(n,))

    area_list = []
    mask_list = []
    insertion_point_list = []
    for i in range(candidate_ys.shape[0]):
        y_0 = candidate_ys[i]
        x_0 = candidate_xs[i]

        if (y_0 - object_y_size / 2 < 0 or y_0 + object_y_size / 2 > H or x_0 - object_x_size / 2 < 0
                or x_0 + object_x_size / 2 > W):
            continue

        tmp_mask = np.zeros_like(gap_mask, dtype=np.float32)
        tmp_mask[int(max(y_0 - object_y_size / 2, 0)):int(min(y_0 + object_y_size / 2, H)),
                 int(max(x_0 - object_x_size / 2, 0)):int(min(x_0 + object_x_size / 2, W))] = 1
        local_mask = tmp_mask.copy()

        tmp_mask = np.all(np.concatenate([tmp_mask[..., None], gap_mask[..., None]], axis=-1), axis=-1)
        area_list.append(np.sum(tmp_mask))
        mask_list.append(tmp_mask)

        # compute the actual center point for insertion
        center_ys, center_xs = np.where(local_mask == 1)
        insertion_point_list.append([center_ys.mean(), center_xs.mean()])

    sorted_ids = sorted(range(len(area_list)), key=lambda k: area_list[k], reverse=True)
    #     insertion_points = [[candidate_ys[i], candidate_xs[i]] for i in sorted_ids[:top_k]]
    insertion_points = [insertion_point_list[i] for i in sorted_ids[:top_k]]
    masks = [mask_list[i] for i in sorted_ids[:top_k]]

    print(f'area list: {area_list}')

    return insertion_points, masks


xrange = [0.37, 0.60]
yrange = [-0.09, 0.25]
grid_resolution = 0.03  # Adjust as needed


def pixel_to_metric(pixel_coord, xrange, yrange, grid_resolution):
    y_pixel, x_pixel = pixel_coord
    x_min, x_max = xrange
    y_min, y_max = yrange

    x_metric = x_min + x_pixel * grid_resolution
    y_metric = y_max - y_pixel * grid_resolution

    return x_metric, y_metric


def main(args=None):
    global fixed_z

    robot_interface = init_franka_interface()
    yaw_config = get_default_controller_config("OSC_YAW")

    reset_joint_positions = [
        0.08535331703543622,
        -0.5125424319854167,
        -0.017850128697897035,
        -2.369681038890256,
        -0.01402637640800741,
        1.8468557911292822,
        0.8491329694317945
    ]

    reset_joints_to(robot_interface, reset_joint_positions)

    rclpy.init(args=None)
    camera_node = CameraSubscriber()  # listen for two cameras
    robot_node = RobotStatePublisherNode(robot_interface)
    from perception.tactile_subscriber import TactileSubscriber
    tactile_node = TactileSubscriber()

    device = SpaceMouse(vendor_id=9583, product_id=50734)
    device.start_control()

    executor = MultiThreadedExecutor()
    executor.add_node(camera_node)
    executor.add_node(robot_node)
    executor.add_node(tactile_node)
    background_thread = threading.Thread(target=lambda: executor.spin())
    background_thread.start()
    time.sleep(2)  # wait for the image to be received

    ee_home_pose = np.array([[0.35], [0], [0.50]])  # back off and stay high to make space
    position_only_gripper_move_to(
        robot_interface, ee_home_pose, grasp=False
    )

    assert camera_node.get_last_reading() is not None, 'no messages received from the cameras'
    # obtain visual observation
    cam2_pc, cam3_pc, cam2_color, cam3_color, cam2_depth, cam3_depth = camera_node.get_last_reading()
    cam2_pc = ros2_pc_to_open3d(cam2_pc)
    cam2_pc = convert_pc_optical_to_link_frame(cam2_pc)
    cam2_pc = project_pc(cam2_pc, extrinsics[f'cam_{2}'])

    cam3_pc = ros2_pc_to_open3d(cam3_pc)
    cam3_pc = convert_pc_optical_to_link_frame(cam3_pc)
    cam3_pc = project_pc(cam3_pc, extrinsics[f'cam_{3}'])

    cam2_color = decode_img_msg(cam2_color)
    cam3_color = decode_img_msg(cam3_color)
    cam2_depth = decode_img_msg(cam2_depth)
    cam3_depth = decode_img_msg(cam3_depth)
    print('Visual observation obtained, start segmenting ')

    # perform segmentation
    from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
    sam = sam_model_registry["vit_h"]("/media/albert/ExternalHDD/pretrained/sam_vit_h_4b8939.pth").cuda()
    sam_predictor = SamAutomaticMaskGenerator(sam)
    cam3_masks = get_sam_masks(sam_predictor, cam3_color)
    pcs = extract_pcs_from_masks(cam3_color, cam3_depth, cam3_masks, 'cam_3')

    # Find the most-likely K point clouds
    # Set the value of K
    K = 8
    sorted_point_clouds = sorted(pcs, key=lambda x: len(x.points), reverse=True)
    largest_k_point_clouds = sorted_point_clouds[:K]
    pcs = largest_k_point_clouds
    # o3d.visualization.draw_geometries(largest_k_point_clouds)

    # paint the point clouds for visualization
    point_clouds = pcs  # Replace with actual point clouds
    cloud_colors = [np.random.rand(3) for _ in point_clouds]
    for pc, color in zip(point_clouds, cloud_colors):
        pc.paint_uniform_color(color)
    o3d.visualization.draw_geometries(point_clouds)

    # obtain occupancy grid
    occupancy_grid = get_occupancy_grid(point_clouds)
    # cv2.imshow('test', occupancy_grid * 255)

    # obtain insertion points
    insertion_points, masks = get_insertion_points(occupancy_grid)
    insertion_points = [pixel_to_metric(point, xrange, yrange, grid_resolution) for point in insertion_points]
    # insertion_points = [[None, None], [None, None]]

    # take the two most promising  insertion points, and add on the true one
    insertion_points = insertion_points[:2]
    if is_testing:
        insertion_points.append([0.445, 0.0100])
    ee_height_for_insertion = 0.18
    for i, point in enumerate(insertion_points):
        if is_testing:
            if i == 0:
                point = [0.53107452945227, 0.14580859482531 + 0.02]
            elif i == 1:
                point = [0.5309933467574436, 0.04654230481136232 + 0.02]
        insertion_points[i] = list(point) + [ee_height_for_insertion]
        # if i < 2:
        #     insertion_points[i][0] -= 0.07  # hack, move inside for 5cm to avoid collision with box edge

    # ground-truth insertion points
    # point 1: 0.53107452945227, 0.14580859482531
    # point 2: 0.5309933467574436, 0.04654230481136232
    # point 3: 0.4406440310984407, -0.0026552155999076442,

    # locate the box to grasp
    box_pc = extract_workspace_pc(cam2_pc, xrange=[0.25, 0.65], yrange=[-0.35, -0.15], zrange=[0.01, 0.3])
    box_pc += fill_point_cloud_by_downward_projection(box_pc)
    o3d.visualization.draw_geometries([box_pc])

    box_center = box_pc.get_center()
    grasp_pose = np.expand_dims(get_grasp_pos(box_center), 1)
    grasp_pose[-1] -= 0.02      # 2cm lower
    grasp_pose[0] += 0.015  # increase x due to partial point cloud

    pre_grasp_pose = grasp_pose.copy()
    pre_grasp_pose[-1] = 0.4

    position_only_gripper_move_to_yaw(
        robot_interface, pre_grasp_pose
    )

    position_only_gripper_move_to_yaw(
        robot_interface, grasp_pose
    )

    time.sleep(3)
    while robot_interface.last_gripper_q > 0.05:
        print(f'closing the gripper, curr gap = {robot_interface.last_gripper_q}')
        close_gripper(robot_interface)
        time.sleep(3)

    # start insertion process
    success = False
    # insertion_points = insertion_points[-1:]
    for point_id, point_to_insert in enumerate(insertion_points):
        if success:
            break

        is_last = point_id == len(insertion_points) - 1
        # point_to_insert = np.array([0.445, 0.0100, 0.18])

        ee_states, _ = robot_node.get_ee_states()
        move_to_safely_yaw(robot_interface, np.expand_dims(ee_states, 1),
                           np.expand_dims(get_grasp_pos(point_to_insert), 1),
                           grasp=True, num_steps=100)

        # time.sleep(0.5)

        if is_last:
            move_to_target_pose(
                robot_interface,
                "OSC_YAW",
                yaw_config,
                [0, 0, 0, 0, np.pi if is_last else 0, 0],
                num_steps=50,
                num_additional_steps=40,
                interpolation_method="linear",
                grasp=True,
            )

        # rotate_gripper(robot_interface, angle_to_rotate=np.pi/2)
        # time.sleep(1)

        # move_to_target_pose(
        #     robot_interface,
        #     "OSC_YAW",
        #     yaw_config,
        #     [0, 0, -0.05, 0, 0, 0],
        #     num_steps=100,
        #     num_additional_steps=40,
        #     interpolation_method="linear"
        # )

        # time.sleep(0.5)
        move_to_target_pose(
            robot_interface,
            "OSC_YAW",
            yaw_config,
            [0, -0.025, 0, 0, 0, 0],
            num_steps=100,
            num_additional_steps=40,
            interpolation_method="linear",
            grasp=True,
        )

        done = False

        num_attempts = 0
        last_has_left = False
        while not done:
            # time.sleep(0.5)

            # try to perform insertion
            move_to_target_pose(
                robot_interface,
                "OSC_YAW",
                yaw_config,
                [0, 0, -0.025, 0, 0, 0],
                num_steps=50,
                num_additional_steps=40,
                interpolation_method="linear",
                grasp=True,
            )

            # get tactile reading
            b1_force, b2_force = tactile_node.get_readings()
            msg = classify_tactile(b1_force, b2_force)

            # set it to move left for the last point
            if is_testing:
                if is_last:
                    if not last_has_left:
                        if msg != 'move to left' and msg != 'fine':
                            msg = 'move to left'
                            last_has_left = True
                        elif msg == 'move to left':
                            msg = 'fine'
                    else:
                        msg = 'fine'

            # if is_last and num_attempts == 0:
            #     msg = 'move to left'

            if msg == "fine":
                ee_states, _ = robot_node.get_ee_states()
                print("Status is fine, current ee pos is", ee_states)
                if ee_states[2] < 0.25:
                    success = True
                    done = True
                else:
                    continue

            elif msg == "Encountering obstacle":
                print("Obstacle detected, stopping")
                done = True
            elif msg == "move to left":
                print("Adjusting robot to the left")
                if is_last:
                    move_by_jump(robot_interface, yaw_config, [-0.04, 0, 0, 0, 0, 0])
                else:
                    move_by_jump(robot_interface, yaw_config, [0, -0.04, 0, 0, 0, 0])
            elif msg == "move to right":
                print("Adjusting robot to the right")
                if is_last:
                    move_by_jump(robot_interface, yaw_config, [0.04, 0, 0, 0, 0, 0])
                else:
                    move_by_jump(robot_interface, yaw_config, [0, 0.04, 0, 0, 0, 0])

            num_attempts += 1
            if not is_last and num_attempts > 2:
                break

    if success:
        # release the gripper
        move_to_target_pose(
            robot_interface,
            "OSC_YAW",
            yaw_config,
            [0, 0, 0, 0, 0, 0],
            num_steps=50,
            num_additional_steps=40,
            interpolation_method="linear",
            grasp=False,
        )
        print("success!")
        reset_joints_to(robot_interface, reset_joint_positions)


    # user_input = None
    # while user_input != 'c':
    #     user_input = input('Grasp the object, then enter c: ')

    # # press down 1cm first
    # position_only_gripper_m
    # ove_by(robot_interface, [0, 0, -0.005], grasp=True)
    #
    # get tactile reading
    # b1_force, b2_force = tactile_node.get_readings()
    # msg = classify_tactile(b1_force, b2_force)

    # diff = b1_force - b2_force
    # print(f'mean of b1 - b2: {diff.reshape(-1, 2).mean(0)}')
    #
    # print(classify_diff(diff))

    # # if we want tele-op
    # while True:
    #     action, grasp = input2action(
    #         device=device,
    #         controller_type=controller_type,
    #     )
    #     if action is None:
    #         break
    #
    #     # set unused orientation dims to 0
    #     if controller_type == "OSC_YAW":
    #         action[3:5] = 0.0
    #     elif controller_type == "OSC_POSITION":
    #         action[3:6] = 0.0
    #     elif controller_type == "JOINT_IMPEDANCE":
    #         action[:7] = robot_interface.last_q.tolist()
    #     # logger.info(action)
    #
    #     action[:3] = np.clip(action[:3], -0.3, 0.3)
    #     robot_interface.control(
    #         controller_type=controller_type,
    #         action=action,
    #         controller_cfg=controller_cfg,
    #     )


if __name__ == '__main__':
    main()

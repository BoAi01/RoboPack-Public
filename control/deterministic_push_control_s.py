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


top_left = np.array([0.623, 0.178, 0.228])
top_right = np.array([0.623, -0.178, 0.228])
bottom_left = np.array([0.329, 0.178, 0.228])
bottom_right = np.array([0.329, -0.178, 0.228])
workspace_center = np.array([0.45, 0, 0])

# 0.30 is exactly the height needed to grasp the rod on table

pushing_ee_height = 0.33      # 0.32 might touch the table
N_POINTS_PER_OBJ = 500
time_to_collect = 60 * 25       # the effective time is only slightly better than half
data_root = '/home/albert/Dataset/v17_0921'

sugar_box_color = np.array([0.45904095, 0.42684464, 0.28072357])
blue_rod_color = np.array([0.03888071, 0.17960485, 0.27454443])
colors = {
    'box': sugar_box_color,
    'rod': blue_rod_color
}

# start_end_pos = [
#     (np.array([0.45, -0.25, fixed_z]), np.array([0.45, +0.30, fixed_z])),
#     (np.array([0.37, -0.25, fixed_z]), np.array([0.37, +0.30, fixed_z])),
#     (np.array([0.45, -0.25, fixed_z]), np.array([0.45, +0.30, fixed_z])),
#     (np.array([0.37, -0.25, fixed_z]), np.array([0.37, +0.30, fixed_z])),
#     (np.array([0.28, 0, fixed_z]), np.array([0.60, 0, fixed_z])),
# ]

start_pos = np.array([0.40 - 0.05, -0.25, pushing_ee_height])
s_shape_trajectory = [
    np.array([0.40, -0.23, pushing_ee_height]),
    np.array([0.42, -0.23, pushing_ee_height]), np.array([0.42, -0.21, pushing_ee_height]),
    np.array([0.44, -0.21, pushing_ee_height]), np.array([0.44, -0.19, pushing_ee_height]),
    np.array([0.46, -0.19, pushing_ee_height]), np.array([0.46, -0.17, pushing_ee_height]),
    np.array([0.48, -0.17, pushing_ee_height]), np.array([0.46, -0.17, pushing_ee_height]),
    np.array([0.50, -0.15, pushing_ee_height]), np.array([0.50, -0.13, pushing_ee_height]),
]
end_pos = np.array([0.45, +0.45, pushing_ee_height])


def generate_s_trajectory(start_pos, num_steps, x_step_size, y_step_size):
    assert x_step_size < 0.1 and y_step_size < 0.1, 'stride size too large'
    x, y, z = start_pos.tolist()

    s_shape_trajectory = []
    for i in range(num_steps):
        # move forward
        y += y_step_size
        s_shape_trajectory.append(np.array([x, y, z]))
        # move right
        x += x_step_size
        s_shape_trajectory.append(np.array([x, y, z]))

    return s_shape_trajectory


def execute_s_shape_trajectory(robot_interface, s_shape_trajectory, num_steps=20):
    for pos in s_shape_trajectory:
        position_only_gripper_move_to(robot_interface, pos, num_steps=num_steps, grasp=True)


def is_in_workspace(pos):
    return 0.20 < pos[0] < 0.75 and -0.40 < pos[1] < 0.40 and 0.15 < pos[2] < 0.60


high_z = 0.4        # so that the rod is slightly above the sugar box


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
    ee_home_pose = np.array([[0.35], [0], [0.65]])  # back off and stay high to make space
    position_only_gripper_move_to(
        robot_interface, ee_home_pose, grasp=False, num_steps=100
    )

    rclpy.init(args=None)
    # camera_node = CameraSubscriber()  # listen for two cameras
    robot_node = RobotStatePublisherNode(robot_interface)
    executor = MultiThreadedExecutor()
    # executor.add_node(camera_node)
    executor.add_node(robot_node)
    background_thread = threading.Thread(target=lambda: executor.spin())
    background_thread.start()
    time.sleep(2)  # wait for the image to be received

    # assert camera_node.get_last_reading() is not None, 'no messages received from the cameras'

    if not args.no_init:
        object_data_dir = create_sequential_folder(data_root)
        print(f'Folder created at {object_data_dir}')

        # start collecting data
        user_input = None
        while user_input != 'c':
            user_input = input("Next grab the tool. Press c when done: ")

        # hard code rod location
        # observe the three back dots on the table
        position_only_gripper_move_to(robot_interface,
                                      [0.504314349575648, 0.16348319530917643, 0.2963322137274104 + 0.15],
                                      grasp=False, num_steps=50)
        position_only_gripper_move_to(robot_interface,
                                      [0.504314349575648, 0.16348319530917643, 0.2963322137274104],
                                      grasp=False, num_steps=50)
        close_gripper(robot_interface)

    else:
        pass

    try:
        start_time = time.time()
        while True:
            ee_states, _ = robot_node.get_ee_states()
            # init_pos, target_pos = start_end_pos[args.action]
            move_to_safely(robot_interface, ee_states, start_pos, grasp=True, num_steps=100)

            s_shape_trajectory = generate_s_trajectory(start_pos, num_steps=5, x_step_size=0.05, y_step_size=0.05)
            execute_s_shape_trajectory(robot_interface, s_shape_trajectory, num_steps=50)

            # position_only_gripper_move_by(robot_interface, [0, -0.05, 0], grasp=True)
            #
            # s_shape_trajectory = generate_s_trajectory(s_shape_trajectory[-1], num_steps=5,
            #                                            x_step_size=-0.05, y_step_size=0.05)
            # execute_s_shape_trajectory(robot_interface, s_shape_trajectory, num_steps=50)

            position_only_gripper_move_to(robot_interface, start_pos, grasp=True)
            position_only_gripper_move_to(robot_interface, end_pos, grasp=True)

            # check break condition
            time_passed = time.time() - start_time
            print(f'time has passed {time_passed:.3f}')
            break

    except KeyboardInterrupt:
        print('keyboard interrupted')

    robot_interface.close()
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="dynamics")
    parser.add_argument("--no_init", default=0, type=int, help="Skip init routine")
    # parser.add_argument("--action", type=int, help="action index")
    args = parser.parse_args()
    # assert args.action is not None, 'action index not specified'
    main(args)

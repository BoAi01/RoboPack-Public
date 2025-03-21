import sys

sys.path.append('/home/albert/github/robopack')

from utils_deoxys import init_franka_interface

import pdb

import rclpy
from rclpy.executors import MultiThreadedExecutor

import threading

from control.motion_utils import *
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

fixed_z = 0.325      # 0.32 might touch the table
N_POINTS_PER_OBJ = 500
time_to_collect = 60 * 25       # the effective time is only slightly better than half
data_root = "/home/albert/Dataset/test_delete"

sugar_box_color = np.array([0.45904095, 0.42684464, 0.28072357])
blue_rod_color = np.array([0.03888071, 0.17960485, 0.27454443])
colors = {
    'box': sugar_box_color,
    'rod': blue_rod_color
}


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


# def get_box_point_cloud(cam0_pc, cam1_pc):
#     min_height = 0.005
#     max_height = 0.05
#
#     pcs = []
#     for i, pc in enumerate([cam0_pc, cam1_pc]):
#         pc = project_pc(convert_pc_optical_color_to_link_frame(pc, i), extrinsics[f'cam_{i}'])
#         pc = extract_workspace_pc(pc, xrange, yrange, zrange)
#         pcs.append(pc)
#
#     object_pc = filter_point_cloud_by_height(pcs[0] + pcs[1], min_height, max_height)
#
#     return fill_point_cloud_by_downward_projection(object_pc) + object_pc


def get_box_point_cloud(cam0_pc):
    min_height = 0.005 + 0.010     # add 1 cm if there is an additional mat on table, and we are using high cameras
    max_height = 0.05

    pc = project_pc(convert_pc_optical_color_to_link_frame(cam0_pc, 0), extrinsics[f'cam_{0}'])
    pc = extract_workspace_pc(pc, xrange, yrange, zrange)

    object_pc = filter_point_cloud_by_height(pc, min_height, max_height)

    return fill_point_cloud_by_downward_projection(object_pc) + object_pc


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


def main(args):
    global pushing_ee_height

    if args.run_robot:
        robot_interface = init_franka_interface()
        ee_home_pose = np.array([[0.35], [0], [0.65]])  # back off and stay high to make space
        position_only_gripper_move_to(
            robot_interface, ee_home_pose, grasp=args.no_init, num_steps=70
        )

    rclpy.init(args=None)
    camera_node = CameraSubscriber()  # listen for two cameras
    executor = MultiThreadedExecutor()
    executor.add_node(camera_node)
    if args.run_robot:
        robot_node = RobotStatePublisherNode(robot_interface)
        executor.add_node(robot_node)
    background_thread = threading.Thread(target=lambda: executor.spin())
    background_thread.start()
    time.sleep(2)  # wait for the image to be received

    assert camera_node.get_last_reading() is not None, 'no messages received from the cameras'

    if not args.no_init:
        pass

        object_data_dir = create_sequential_folder(data_root)
        print(f'Folder created at {object_data_dir}')

        # start collecting data
        user_input = None
        while user_input != 'c':
            user_input = input("Next grab the tool. Press c when done: ")

        # hard code rod location
        # observe the three back dots on the table
        if args.run_robot:
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
            # find the object
            cam0_pc = decode_pointcloud_msg(camera_node.get_last_reading()[0])
            # model_data_dict = reconstruct_single_object_and_save(cam0_pc, cam1_pc, model, predictor, 'box')
            # object_pc = xyz_colors_to_pc(model_data_dict['object_pc'])
            object_pc = get_box_point_cloud(cam0_pc)
            # o3d.visualization.draw_geometries([object_pc])
            # import pdb; pdb.set_trace()
            object_center = np.asarray(object_pc.points).mean(0)   # geometric mean

            # # testing
            # position_only_gripper_move_to(robot_interface, np.array([object_center[0], object_center[1], fixed_z]))

            # object_center = np.array(sample_xy_in_range([0.3, 0.7], [-0.25, 0.25]) + [0.15])
            print(f'sampled object center: {object_center}')
            workspace_center = np.array([0.5, 0.0, 0.15])

            # find a suitable start and goal position pair as an action
            num_attempts_sample = 0
            init_pos, target_pos = None, None
            while True:
                if args.collision:
                    object_center[:2] += (np.random.random(2) * 0.04 - 0.02)  # inject noise to xy
                    workspace_center[:2] += (np.random.random(2) * 0.04 - 0.02)  # inject noise to center

                    # compute actuation trajectory
                    if args.run_robot:
                        ee_states, _ = robot_node.get_ee_states()
                    else:
                        ee_states = [0.5, 0.4, 0.3]

                    init_distance = 0.1
                    while True:
                        init_pos = find_point_on_line(workspace_center, object_center, distance=init_distance)
                        if is_coordinate_inside_point_cloud(object_pc,
                                                            np.array([init_pos[0], init_pos[1], 0.01]), 0.08):
                            init_distance += 0.02
                            init_pos = find_point_on_line(workspace_center, object_center, distance=init_distance)
                        else:
                            break
                    target_pos = find_point_on_line(object_center, workspace_center, distance=0.28)

                else:
                    # if we do not want any collision, we should sample the initial and end pose in a large region,
                    # then filter out ones that lead to collision
                    found, counter = False, 0
                    while not found:
                        init_pos = object_center.copy()
                        init_pos[:2] += (np.random.random(2) * 0.6 - 0.30)  # sample in a radius of 30cm

                        target_pos = object_center.copy()
                        target_pos[:2] += (np.random.random(2) * 0.6 - 0.30)  # sample in a radius of 30cm

                        # sample a set of points on the line, check their distance to the closest point in the box
                        n_samples_collision, ratio, distance_threshold = 100, 0.10, 0.01

                        line_points = sample_points_on_line(init_pos, target_pos, n_samples_collision)
                        in_box, distances = are_coordinates_within_point_cloud_np(object_pc, line_points,
                                                                                  threshold=distance_threshold,
                                                                                  return_distance=True)

                        # check the sampled trajectory go through a corner of the box
                        # this can be done by constraining the portion of points on the line
                        # that are close to the box points
                        # adjust the parameters if this is too conservative or aggressive
                        init_pos[-1], target_pos[-1] = fixed_z, fixed_z
                        if is_in_workspace(init_pos) and is_in_workspace(target_pos) and \
                                np.linalg.norm(target_pos - init_pos) > 0.20 \
                                and 0 < in_box.sum() < ratio * n_samples_collision:
                            found = True

                        print(counter, in_box.sum(), np.linalg.norm(target_pos - init_pos))

                        counter += 1
                        if counter > 300:
                            break

                    if not found:
                        print(f'init_pos and target_pos not found, please check why')
                        pdb.set_trace()

                init_pos[-1], target_pos[-1] = fixed_z, fixed_z
                print(f'Action candidate: {init_pos} ---> {target_pos}, '
                      f'distance = {np.linalg.norm(init_pos - target_pos):.3f}')

                if is_in_workspace(init_pos) and is_in_workspace(target_pos):
                    break
                elif num_attempts_sample < 10:
                    num_attempts_sample += 1
                    continue
                else:
                    print(f'one of the positions {init_pos, target_pos} not in workspace ')
                    user_input = None
                    while user_input != 'c':
                        o3d.visualization.draw_geometries([object_pc])
                        user_input = input(f'stop recording, re-arrange object, and enter c: ')
                    continue

            if args.run_robot:
                ee_states, _ = robot_node.get_ee_states()
                print(f'ee_states = {ee_states}')
                move_to_safely(robot_interface, ee_states, init_pos, grasp=True, num_steps=100)
                position_only_gripper_move_to(robot_interface, target_pos, grasp=True, num_steps=150)

            # check break condition
            time_passed = time.time() - start_time
            print(f'time has passed {time_passed:.3f}')
            if time_passed > time_to_collect:
                break

    except KeyboardInterrupt:
        print('keyboard interrupted')

    robot_interface.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="dynamics")
    parser.add_argument("--no_init", default=0, type=int, help="Skip init routine")
    parser.add_argument("--collision", default=1, type=int, help="want trajectories to go through box or not")
    parser.add_argument("--run_robot", action="store_true", help="run script with real robot")
    args = parser.parse_args()
    main(args)

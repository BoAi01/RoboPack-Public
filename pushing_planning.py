import random
import sys
import time

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

sys.path.append('/home/albert/github/robopack')

from utils_deoxys import init_franka_interface
from deoxys import config_root

import numpy as np
import pdb

import threading

from control.motion_utils import *
from perception.utils_sam import *
from perception.utils_cv import *
from perception.utils_ros import *
from perception.utils_pc import *
from perception.camera_subscriber import CameraSubscriber
from perception.utils_gripper import get_origin_ee_point_cloud, get_bubble_cameras_to_robot, get_rod_points_from_model

from utils_general import *

from config.task_config import *

from control.robot_state_publisher import RobotStatePublisherNode

from dynamics.config_parser import ConfigParser
# from dynamics.models.dynamics_obj_latent_lstm import DynamicsPredictor
from dynamics.models.estimator_predictor_obj_latent_lstm import DynamicsPredictor

from planning.samplers import CorrelatedNoiseSampler
from planning.planner import MPPIOptimizer

import rclpy
from rclpy.executors import MultiThreadedExecutor
from custom_msg.msg import StampedFloat32MultiArray
# import Image later to prevent it from being aliased as the PIL Image class -- Important!!
# Otherwise results in very ambiguous error "AttributeError: type object 'module' has no attribute '_TYPE_SUPPORT'"
from sensor_msgs.msg import PointCloud2, Image, JointState
from std_msgs.msg import Float32MultiArray, Float32
from message_filters import Subscriber

import torch_geometric as pyg

high_z = 0.4
very_high_z = 0.6       # this height is such that the EE should not block the view of any camera


def move_to_safely(robot_interface, current_pos, target_pos, **args):
    current_pos[-1] = high_z
    target_high_pos = target_pos.copy()
    target_high_pos[-1] = high_z
    position_only_gripper_move_to(robot_interface, current_pos, **args)
    position_only_gripper_move_to(robot_interface, target_high_pos, **args)
    position_only_gripper_move_to(robot_interface, target_pos, **args)


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


def get_box_point_cloud(cam0_pc):
    min_height = 0.005 + 0.010     # add 1 cm if there is an additional mat on table, and we are using high cameras
    max_height = 0.05

    pc = project_pc(convert_pc_optical_color_to_link_frame(cam0_pc, 0), extrinsics[f'cam_{0}'])
    pc = extract_workspace_pc(pc, xrange, yrange, zrange)

    object_pc = filter_point_cloud_by_height(pc, min_height, max_height)
    # TODO check to see if using only the box surface may work better

    # object_pc = fill_point_cloud_by_downward_projection(object_pc) + object_pc    # we only need surface points?
    object_pc = denoise_by_cluster(object_pc, 0.01, 10, 1)
    return object_pc


def get_canonical_box_point_cloud(box_target_position, box_target_orientation):
    # Get the "canonical" point cloud of a box, picked from a random trajectory in the data, but
    # transform it to the target position and orientation. This can then be used to compute losses
    # for a particular planning goal.
    # the orientation is specified in degrees (not radians) and is [roll, pitch, yaw]
    box_points = load_h5_data('/home/albert/github/robopack/asset/box3_seq_1.h5')['object_pcs'][0][0]   # (N, 3)
    from perception.utils_pc import center_and_rotate_to, farthest_point_sampling_dgl
    goal = center_and_rotate_to(box_points, box_target_position, box_target_orientation)
    goal = farthest_point_sampling_dgl(goal, 20)
    return goal


bubble1_mask, bubble2_mask = None, None

def get_first_frame(ee_pos, ee_orient, gripper_dist, b1_flow, b2_flow, b1_force, b2_force,
                    b1_depth, b2_depth):

    # get bubble sampled pcs
    b1_flow, b2_flow = np.array(b1_flow).reshape(240, 320, 2), np.array(b2_flow).reshape(240, 320, 2)

    b1_flow = cv2.resize(b1_flow, (640, 480))
    b2_flow = cv2.resize(b2_flow, (640, 480))

    b1_feat_points = rgbd_feat_to_point_cloud(b1_flow, b1_depth, intrinsic_matrices['bubble_1'])
    b2_feat_points = rgbd_feat_to_point_cloud(b2_flow, b2_depth, intrinsic_matrices['bubble_2'])

    t_bubbles_to_robot = get_bubble_cameras_to_robot(ee_pos, ee_orient, gripper_dist)

    # use the same mask throughout the program
    global bubble1_mask, bubble2_mask
    if bubble1_mask is None and bubble2_mask is None:
        bubble_pcs = [b1_feat_points, b2_feat_points]
        bubble_pcs = [remove_distant_points(pc, 0.15) for pc in bubble_pcs]
        bubble_pcs = [random_select_rows(pc, 10000) for pc in bubble_pcs]
        bubble_pcs = [denoise_by_cluster(pc, 0.01, 10, 1) for pc in bubble_pcs]
        bubble_sampled_pcs = [farthest_point_sampling_dgl(pc, n_points=20)
                              for pc in bubble_pcs]
        bubble1_mask = find_indices(b1_feat_points, bubble_sampled_pcs[0])
        bubble2_mask = find_indices(b2_feat_points, bubble_sampled_pcs[1])
    else:
        bubble_sampled_pcs = [b1_feat_points[bubble1_mask], b2_feat_points[bubble1_mask]]

    bubble_sampled_pcs = [project_points(convert_pc_optical_color_to_link_frame(pc), t_bubble_to_robot)
                          for pc, t_bubble_to_robot in zip(bubble_sampled_pcs, t_bubbles_to_robot)]
    bubble_sampled_pcs = np.stack(bubble_sampled_pcs, axis=0)

    # for sanity check, visualize and save the flow
    # flow = bubble_sampled_pcs[0][..., 3:].reshape(4, 5, 2)
    # flow = resize_image(flow, 200, 250)
    # flow_vis = visualize_raw_flow(flow, scale=4)
    # cv2.imwrite(f'planning/log_tactile/flow_vis_{time.time()}.png', flow_vis)
    # import pdb; pdb.set_trace()

    # ref_ee_pos = np.array([0.4841749, 0.1179733, 0.22854297 + 0.091])
    # ref_index = 120
    # translation = ee_pos - ref_ee_pos
    # data_dict = load_dictionary_from_hdf5('asset/v18_box3_seq_12.h5')
    # ref_bubbles_pts = data_dict['bubble_pcs'][ref_index][..., :3] + translation
    # inhand_obj_pc = data_dict['inhand_object_pcs'][ref_index] + translation
    # inhand_obj_pc_o3d = xyz_to_pc(inhand_obj_pc)
    # inhand_obj_pc_o3d = inhand_obj_pc_o3d.paint_uniform_color([1, 0, 0])

    # replace observed point cloud with 3d model
    rod_model_path = '/home/albert/Data/rod_3x3x25_fixed_Aug1.ply'
    rod_model_points = get_rod_points_from_model(rod_model_path, ee_pos)
    # rod_model_points = xyz_to_pc(rod_model_points)
    # rod_model_points = rod_model_points.paint_uniform_color([0, 0, 1])

    # for debugging
    # workspace = get_origin_ee_point_cloud(ee_pos)
    # o3d.visualization.draw_geometries(list(workspace) + [inhand_obj_pc_o3d, rod_model_points])
    # print(inhand_obj_pc_o3d.get_center() - rod_model_points.get_center())

    # as a sanity check, ref_bubbles_pts should be close to bubble_sampled_pcs
    # passed! difference smaller than 0.006
    # print('diff', (ref_bubbles_pts.reshape(-1, 2).mean(0) - bubble_sampled_pcs[..., :3].reshape(-1, 2).mean(0) ))
    # bubbles = [xyz_to_pc(pc) for pc in bubble_sampled_pcs] + [xyz_to_pc(pc) for pc in ref_bubbles_pts]
    # o3d.visualization.draw_geometries(bubbles)

    frame_dict = {
        'bubble_pcs': bubble_sampled_pcs.astype(np.float32),
        'inhand_object_pcs': rod_model_points.astype(np.float32),
        'forces': np.stack([b1_force, b2_force], axis=0).astype(np.float32)
    }

    return frame_dict


sync_topics = {
    # '/cam_0/color/image_raw': 'sensor_msgs/msg/Image',
    # '/cam_0/depth/image_rect_raw': 'sensor_msgs/msg/Image',
    '/cam_1/color/image_raw': 'sensor_msgs/msg/Image',
    '/cam_1/depth/image_rect_raw': 'sensor_msgs/msg/Image',
    '/cam_2/color/image_raw': 'sensor_msgs/msg/Image',
    '/cam_2/depth/image_rect_raw': 'sensor_msgs/msg/Image',
    '/cam_3/color/image_raw': 'sensor_msgs/msg/Image',
    '/cam_3/depth/image_rect_raw': 'sensor_msgs/msg/Image',

    '/bubble_1/depth/image_rect_raw': 'sensor_msgs/msg/Image',
    '/bubble_2/depth/image_rect_raw': 'sensor_msgs/msg/Image',
    # '/ee_states': 'custom_msg/msg/StampedFloat32MultiArray',

    '/bubble_1/raw_flow': 'custom_msg/msg/StampedFloat32MultiArray',
    '/bubble_2/raw_flow': 'custom_msg/msg/StampedFloat32MultiArray',
    '/bubble_1/force': 'custom_msg/msg/StampedFloat32MultiArray',
    '/bubble_2/force': 'custom_msg/msg/StampedFloat32MultiArray',
    # '/joint_states': 'sensor_msgs/msg/JointState'
}
topic_msg_dict = {k: None for k in sync_topics.keys()}


def callback(*args):
    # print('callback invoked')
    for msg, (topic, type_name) in zip(args, sync_topics.items()):
        if 'Image' in type_name:
            msg = decode_img_msg(msg)
        elif 'StampedFloat32MultiArray' in type_name:
            msg = msg.data.data
        elif 'JointState' in type_name:
            msg = msg.position
        elif 'point_cloud' in type_name:
            msg = decode_pointcloud_msg(msg)
        topic_msg_dict[topic] = msg


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



pushing_ee_height = 0.325      # 0.32 might touch the table
moving_ee_height = 0.4


def main(args):
    global pushing_ee_height      # fixed height for moving end effector

    # obtain planning config path
    parser = argparse.ArgumentParser(description="dynamics")
    parser.add_argument(
        "-c",
        "--config",
        default=os.path.join("planning", "planning_config.json"),
        type=str,
        help="config file path (default: dynamics_config.json)",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="0",
        type=str,
        help="device to use for training / testing (default: cuda:0)",
    )
    parser.add_argument(
        "--run_robot",
        action="store_true",
        help="run script with real robot",
    )
    planning_config = ConfigParser.from_args(parser)
    run_robot = parser.parse_args().run_robot

    # load dynamics model and config
    pretrained_path = planning_config["pretrained_path"]
    model = DynamicsPredictor.load_from_checkpoint(pretrained_path, map_location='cuda:0').to(device)
    model_config = ConfigParser(dict())
    model_config.update_from_json("/home/albert/github/robopack/v13_latent_lstm_tactile_after.json")
    assert model_config["test_batch_size"] == 1, "test batch size must be 1"
    print(f'Dynamics model and config loaded')

    rclpy.init(args=None)
    planner_node = rclpy.create_node('mpc_planner')

    # move EE to initial pose
    if run_robot:
        robot_interface = init_franka_interface()
        ee_home_pose = np.array([[0.35], [0], [0.65]])  # back off and stay high to make space
        position_only_gripper_move_to(
            robot_interface, ee_home_pose, grasp=False, num_steps=70
        )
        # confirm with keyboard that scene is set up
        input('Press enter to continue when scene setup is ready: ')

    # start subscribing to various topics
    # same as the data collection pipeline
    sync_subscriptions = []
    for k, v in sync_topics.items():
        sync_subscriptions.append(Subscriber(planner_node, globals()[v.split("/")[-1]], k))
        check_topic_status(planner_node, k)

    synchronizer = ApproximateTimeSynchronizer(
        sync_subscriptions, queue_size=200, slop=0.2, allow_headerless=False
    )
    synchronizer.registerCallback(callback)
    camera_node = CameraSubscriber()
    if run_robot:
        robot_node = RobotStatePublisherNode(robot_interface)
    executor = MultiThreadedExecutor()
    executor.add_node(planner_node)
    executor.add_node(camera_node)
    if run_robot:
        executor.add_node(robot_node)
    background_thread = threading.Thread(target=lambda: executor.spin())
    background_thread.start()
    time.sleep(5)  # wait for the image to be received

    for topic, msg in topic_msg_dict.items():
        assert msg is not None, f'{topic} is None'
    print(f'messages have been received')

    # hard code rod location
    # grab the rod from at the given location
    # observe the three back dots on the table
    if run_robot:
        position_only_gripper_move_to(robot_interface,
                                      [0.504314349575648, 0.16348319530917643, 0.2963322137274104 + 0.15],
                                      grasp=False, num_steps=50)
        position_only_gripper_move_to(robot_interface,
                                      [0.504314349575648, 0.16348319530917643, 0.2963322137274104],
                                      grasp=False, num_steps=50)
        print("Closing gripper...")
        close_gripper(robot_interface)
        print("Done closing gripper :)")

        # move to a fixed starting point
        pushing_init_pos = planning_config["initial_ee_pos"]
        pushing_init_pos[-1] = moving_ee_height + 0.3     # lower down height
        ee_states, _ = robot_node.get_ee_states()
        move_to_safely(robot_interface, ee_states, pushing_init_pos, grasp=True)

    start = time.time()
    cam0_pc = decode_pointcloud_msg(camera_node.get_last_reading()[0])
    print("Decode pointcloud message in ", time.time() - start)
    # model_data_dict = reconstruct_single_object_and_save(cam0_pc, cam1_pc, model, predictor, 'box')
    # object_pc = xyz_colors_to_pc(model_data_dict['object_pc'])
    object_pc = get_box_point_cloud(cam0_pc)

    print("Box pc in ", time.time() - start)

    print("Box point cloud received")
    # o3d.visualization.draw_geometries([object_pc])
    print("visualize in ", time.time() - start)

    if run_robot:
        pushing_init_pos = planning_config["pushing_initialization_pos"]
        pushing_init_pos[-1] = pushing_ee_height  # lower down height
        ee_states, _ = robot_node.get_ee_states()
        move_to_safely(robot_interface, ee_states, pushing_init_pos, grasp=True)

    # Next, we start planning

    # first build a sampler
    action_dim = planning_config["action_dim"]
    initial_sampler = CorrelatedNoiseSampler(a_dim=action_dim,
                                             beta=planning_config["initial_sampler_params"]["beta"],
                                             horizon=planning_config["initial_sampler_params"]["horizon"],
                                             num_repeat=planning_config["initial_sampler_params"]["num_repeat"]
                                             )
    replan_sampler = CorrelatedNoiseSampler(a_dim=action_dim,
                                            beta=planning_config["replan_sampler_params"]["beta"],
                                            horizon=planning_config["replan_sampler_params"]["horizon"],
                                            num_repeat=planning_config["replan_sampler_params"]["num_repeat"])

    # then build a planner
    from planning.cost_functions import pointcloud_cost_function as objective

    mu = np.array(planning_config["initial_sampler_params"]["mean"])

    initial_planner = MPPIOptimizer(
        sampler=initial_sampler,
        model=model,
        objective=objective,
        a_dim=action_dim,
        horizon=planning_config["initial_sampler_params"]["horizon"],
        num_samples=planning_config["initial_sampler_params"]["num_samples"],
        gamma=planning_config["initial_sampler_params"]["gamma"],
        num_iters=planning_config["initial_sampler_params"]["num_iterations"],
        init_std=np.array(planning_config["initial_sampler_params"]["std"])
    )

    replan_planner = MPPIOptimizer(
        sampler=replan_sampler,
        model=model,
        objective=objective,
        a_dim=action_dim,
        horizon=planning_config["replan_sampler_params"]["horizon"],
        num_samples=planning_config["replan_sampler_params"]["num_samples"],
        gamma=planning_config["replan_sampler_params"]["gamma"],
        num_iters=planning_config["replan_sampler_params"]["num_iterations"],
        init_std=np.array(planning_config["replan_sampler_params"]["std"])
    )

    # start the planning loop
    # the loop executes for num_execution_iterations, in which the robot
    # executes num_actions_to_execute_per_step actions
    # in the action plan given by the planer

    # take visual observation ONCE
    # For now, we use the heuristic-based perception pipeline
    # If needed, should replace this with the first step in tracking system
    # cam0_pc = decode_pointcloud_msg(camera_node['/cam_0/color/image_raw'][0])
    # model_data_dict = reconstruct_single_object_and_save(cam0_pc, cam1_pc, model, predictor, 'box')
    # object_pc = xyz_colors_to_pc(model_data_dict['object_pc'])
    # object_pc = get_box_point_cloud(cam0_pc)
    # o3d.visualization.draw_geometries([object_pc])

    all_real_bubble_points = []

    # get the soft bubble and rod points
    if run_robot:
        ee_states, ee_orient = robot_node.get_ee_states()
        joint_pos, joint_vel, gripper_state = robot_node.get_joint_states()
        gripper_state = np.array([gripper_state])
        ee_states, ee_orient = np.array(ee_states), np.array(ee_orient)
    else:
        ee_states = np.array([0.5, -0.3, 0.325])
        ee_orient = np.array([0.0, 1.0, 0.0, 0.0])
        gripper_state = np.array([0.0])

    first_frame = get_first_frame(ee_states, ee_orient, gripper_state.tolist()[-1],
                                  topic_msg_dict['/bubble_1/raw_flow'],
                                  topic_msg_dict['/bubble_2/raw_flow'],
                                  topic_msg_dict['/bubble_1/force'],
                                  topic_msg_dict['/bubble_2/force'],
                                  topic_msg_dict['/bubble_1/depth/image_rect_raw'],
                                  topic_msg_dict['/bubble_2/depth/image_rect_raw']
                                  )

    # visualize_first_frame(first_frame, ee_states)
    first_frame["object_pcs"] = [np.asarray(object_pc.points).astype(np.float32)]
    first_frame["object_cls"] = [-1]

    # prepare the initial frame
    from dynamics.dataset import construct_graph_from_video, downsample_points_state_dict_seq
    state_window = [first_frame]  # contains only the first frame
    state_window = downsample_points_state_dict_seq(state_window, model_config)
    batch = construct_graph_from_video(model_config, state_window)

    # actually make it into a batch
    batch = pyg.data.Batch.from_data_list([batch]).cuda()

    # the following is copied from test_planning.py
    B = batch.num_graphs
    N = batch.num_nodes // B
    seq_len = batch.pos.view(B, N, -1).shape[2] // action_dim
    observed_his_len = planning_config["history_len"]

    node_pos = batch.pos.view(B, N, seq_len, -1)
    tac_feat_bubbles = model.autoencoder.encode_structured(batch)
    node_feature = batch.x.view(B, N, -1)
    action_hist = node_feature[:, :, model.type_feat_len + model.vis_feat_len:].view(B, N, seq_len - 1, action_dim)

    # get the history needed
    object_type_feat = node_feature[..., :model.type_feat_len]  # should be zero
    vision_feature = torch.zeros(B, N, model.vis_feat_len).to(object_type_feat.device)

    # start planning
    counter = 0
    best_real_score = 10000
    num_actions_to_execute_per_step = planning_config["num_actions_to_execute_per_step"]
    num_execution_iterations = planning_config["num_execution_iterations"]
    # goal = np.array(planning_config["goal"])
    goal = get_canonical_box_point_cloud(planning_config["goal"]["position"], planning_config["goal"]["orientation"])
    log_dir = os.path.join("planning", str(time.time()))

    # the plan-execution loop
    while True:
        print("\n\nStarting planning...")
        node_pos = node_pos[:, :, :observed_his_len, :]
        tac_feat_bubbles = tac_feat_bubbles[:, :observed_his_len, :, :]
        tactile_feat_objects = torch.zeros(B, tac_feat_bubbles.shape[1],
                                           model.n_object_points,
                                           model.config["ae_enc_dim"]).to(model.device)
        tactile_feat_all_points = torch.cat([tactile_feat_objects, tac_feat_bubbles], dim=2)
        action_hist = action_hist[:, :, :(observed_his_len - 1), :]

        # Generate the initial plan
        print(object_type_feat.shape, vision_feature.shape, node_pos.shape, tac_feat_bubbles.shape,
              tactile_feat_all_points.shape)

        planner = initial_planner if counter == 0 else replan_planner
        if planner == initial_planner:
            print("Using initial planner")
        else:
            print("Using replan planner")

        # create log dir under "planning" folder with the current time as the name
        mu, action, best_action_prediction = planner.plan(
            t=counter,
            log_dir=log_dir,
            observation_batch=(object_type_feat,
                               vision_feature,
                               node_pos,
                               tac_feat_bubbles,
                               tactile_feat_all_points),
            action_history=action_hist,
            goal=goal,
            visualize_top_k=planning_config["visualize_top_k"],
            init_mean=mu,
            return_best=True,
        )

        # print("Found plan:", action)
        # execute the best action
        # merge adjacent duplicate actions into one for easy actuation
        actions = action[:num_actions_to_execute_per_step]
        # merged_actions = merge_same_consecutive_arrays(actions)

        # obtain trajectory
        # current_pos = None
        # while current_pos is None:
        #     _, current_pos = robot_interface.last_eef_rot_and_pos
        current_pos = ee_states
        desired_ee_positions = np.cumsum(actions, axis=0) + current_pos.squeeze()

        # force z value of the desired ee position to be the same as the pushing_ee_height
        desired_ee_positions[:, -1] = pushing_ee_height

        for iteration, (desired_ee_pos, action, curr_pred_state_dict) \
                in enumerate(zip(desired_ee_positions, actions, best_action_prediction)):
            # print("Taking action ", action)
            # tune num_steps to trade off between actuation frequency and actuation accuracy
            # position_only_gripper_move_by(robot_interface, action, num_steps=50, allowance=0.005)
            # ee_states, _ = robot_node.get_ee_states()
            if run_robot:
                position_only_gripper_move_to(robot_interface, desired_ee_pos[:, None], grasp=True, num_steps=50,
                                              allowance=0.005)
                print("Moving to ", desired_ee_pos)
            else:
                print("Not running robot, but attempted to move to ", desired_ee_pos)

            # update the state history based on model prediction and feedback
            curr_pred_state_dict = {k: torch.from_numpy(v).cuda() for k, v in curr_pred_state_dict.items()}

            # collect robot and camera reading before raising end effector
            if run_robot:
                ee_states, ee_orient = robot_node.get_ee_states()
                joint_pos, joint_vel, gripper_state = robot_node.get_joint_states()
                gripper_state = np.array([gripper_state])
                ee_states, ee_orient = np.array(ee_states), np.array(ee_orient)
            else:
                # this is actually incorrect, since the end effector should have moved
                ee_states = np.array([0.5, -0.3, 0.325])
                ee_orient = np.array([0.0, 1.0, 0.0, 0.0])
                gripper_state = np.array([0.02])

            # get the current camera observation, before EE moves
            curr_topic_msg_dict = {k: v for k, v in topic_msg_dict.items()}

            # we need a frame to extract softbubble points and tactile reading
            curr_frame = get_first_frame(ee_states, ee_orient, gripper_state.tolist()[-1],
                                         curr_topic_msg_dict['/bubble_1/raw_flow'],
                                         curr_topic_msg_dict['/bubble_2/raw_flow'],
                                         curr_topic_msg_dict['/bubble_1/force'],
                                         curr_topic_msg_dict['/bubble_2/force'],
                                         curr_topic_msg_dict['/bubble_1/depth/image_rect_raw'],
                                         curr_topic_msg_dict['/bubble_2/depth/image_rect_raw']
                                         )
            # visualize_first_frame(first_frame, ee_states)

            # update object points based on visual feedback for the last step
            object_points = curr_pred_state_dict['object_obs']
            if iteration == num_actions_to_execute_per_step - 1:

                # raise end effector to avoid occlusion by robot
                before_back_off_ee_xyz = ee_states
                if run_robot:
                    # extrapolate the line from box to current ee pose to find the back off point
                    back_off_ee_xyz = find_point_on_line(object_points.mean(0), ee_states, 0.05)
                    position_only_gripper_move_to(robot_interface, [back_off_ee_xyz[0], back_off_ee_xyz[1],
                                                                    very_high_z],
                                                  grasp=True, num_steps=50)

                # read point cloud from camera
                cam0_pc = decode_pointcloud_msg(camera_node.get_last_reading()[0])
                object_pc = get_box_point_cloud(cam0_pc)
                print(f'object points updated based on feedback ')

                # put end effector back to get ready for the next iteration
                if run_robot:
                    position_only_gripper_move_to(robot_interface,
                                                  [before_back_off_ee_xyz[0], before_back_off_ee_xyz[1],
                                                   pushing_ee_height],
                                                  grasp=True, num_steps=50)
                    position_only_gripper_move_to(robot_interface,
                                                  [ee_states[0], ee_states[1], pushing_ee_height],
                                                  grasp=True, num_steps=75)

                if planning_config["use_visual_feedback"]:
                    curr_frame["object_pcs"] = [np.asarray(object_pc.points).astype(np.float32)]
                    state_window = downsample_points_state_dict_seq([curr_frame], model_config)
                    curr_frame = state_window[0]
                    # update the prediction state dict
                    object_points[:, :3] = torch.from_numpy(curr_frame["object_pcs"][0]).cuda()
                    object_points[:, 3:] = 0
                    curr_pred_state_dict['object_obs'] = object_points

                # compute real cost
                # real_cost = -np.linalg.norm(object_points[:, :3].mean(0).cpu().numpy() - goal, axis=-1)
                real_cost = -1 * objective({"object_obs": object_points[:, :3].cpu().numpy()[None, None]}, goal, last_state_only=True).item()
                best_real_score = min(best_real_score, real_cost)
                print(f'\tThe real cost is {real_cost:.3f}')

                if np.abs(real_cost) < 0.03:
                    print(f'real cost very low, goal reached, stop. ')
                    break
            else:
                curr_frame["object_pcs"] = [object_points.cpu().numpy().astype(np.float32)]
            curr_frame["object_cls"] = [-1]

            # sanity check real bubble and predicted bubble, which should be the same
            predicted_bubble = curr_pred_state_dict['bubble'].view(2, 20, -1).cpu().numpy().astype(np.float32)[..., :3]
            real_bubble = curr_frame['bubble_pcs'].astype(np.float32)[..., :3]

            all_real_bubble_points.append(real_bubble)

            diff = np.median(np.abs(predicted_bubble - real_bubble).mean(axis=-1).flatten())
            print(f'diff between real and predicted bubble: {diff:.3f}')
            # import pdb; pdb.set_trace()
            # print(f'diff between real and predicted bubble: {np.abs(predicted_bubble - real_bubble).mean(axis=-1)}')

            if diff > 0.05:
                print('\tdiff too large, please check why')
                predicted_bubble_o3d = xyz_to_pc(predicted_bubble.reshape(-1, 3))
                predicted_bubble_o3d.paint_uniform_color([1, 0, 0])
                real_bubble_o3d = xyz_to_pc(real_bubble.reshape(-1, 3))
                real_bubble_o3d.paint_uniform_color([0, 1, 0])
                o3d.visualization.draw_geometries([predicted_bubble_o3d, real_bubble_o3d])

            dummy_batch = construct_graph_from_video(model_config, [curr_frame])
            dummy_batch = pyg.data.Batch.from_data_list([dummy_batch]).cuda()
            curr_tac_feat_bubbles_real = model.autoencoder.encode_structured(dummy_batch)     # shape torch.Size([1, 1, 40, 5])

            # update bubble points based on feedback
            curr_pred_state_dict['bubble'][..., :3] = torch.from_numpy(real_bubble.reshape(-1, 3)).cuda()

            # update particle history
            curr_pos = torch.cat([curr_pred_state_dict['object_obs'],
                                  curr_pred_state_dict['inhand'],
                                  curr_pred_state_dict['bubble']], dim=0).unsqueeze(0).unsqueeze(2)
            node_pos = torch.cat([node_pos, curr_pos[..., :3]], dim=2)  # shape (B, N, T, 3)

            # note: the rod points are not updated. we are assuming the points remain the same
            # as the previous frame, which should be true in most of the cases

            # update tactile encoding based on feedback
            # curr_tac_feat_all_points_pred = curr_pos[..., 3:].permute(0, 2, 1, 3)    # shape (B, T, N, 3)
            # curr_tac_feat_all_points_pred[:, :, :40] = 0
            # curr_tac_feat_bubbles_pred = curr_tac_feat_all_points_pred[:, :, 40:]
            # curr_tac_feat_all_points_pred_real = torch.cat([curr_tac_feat_all_points_pred[:, :, :40],
            #                                                 curr_tac_feat_bubbles_real], dim=2)  # shape (B, T, N, 3)
            tac_feat_bubbles = torch.cat([tac_feat_bubbles, curr_tac_feat_bubbles_real], dim=1)  # shape (B, T, N, 3)
            # tactile_feat_all_points = torch.cat([tactile_feat_all_points, curr_tac_feat_all_points_pred_real],
            #                                     dim=1)  # shape (B, T, N, 3)

            # update action history
            action = torch.from_numpy(action).unsqueeze(0).unsqueeze(0).unsqueeze(0).cuda()
            action = action.repeat(1, N, 1, 1)
            action_hist = torch.cat([action_hist, action], dim=2)  # shape (B, N, T, 3)


            print(f'Current tactile history length: {tac_feat_bubbles.shape[1]}')
            print(f'Best real score so far: {best_real_score:.3f}')

        print(f"Executed and updated state history, counter = {counter}")

        # update counter
        counter += 1
        # push mu forward to initialize the planner in next iteration
        mu = mu[num_actions_to_execute_per_step:]

        # check stop condition
        if counter * num_actions_to_execute_per_step > num_execution_iterations:
            print(f'num_execution_iterations = {num_execution_iterations} reached. Stop.')
            print(f'best real score: {best_real_score:.3f}')
            break

    # after execution, compute metrics


if __name__ == '__main__':
    main(None)

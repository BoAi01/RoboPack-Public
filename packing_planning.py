import os
import sys
import time
import numpy as np
import torch
import pickle
import multiprocessing

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

sys.path.append('/home/albert/github/robopack')

from utils_deoxys import init_franka_interface
from deoxys.utils.transform_utils import mat2euler

import threading

from control.motion_utils import *
from perception.utils_sam import *
from perception.utils_cv import *
from perception.utils_ros import *
from perception.utils_pc import *
from perception.camera_subscriber import CameraSubscriber, CameraImageSubscriber
from perception.utils_gripper import get_origin_ee_point_cloud, get_bubble_cameras_to_robot, get_rod_points_from_model

from config.task_config import *

from control.robot_state_publisher import RobotStatePublisherNode

from dynamics.config_parser import ConfigParser
from dynamics.dataset import construct_graph_from_video, downsample_points_state_dict_seq

from planning.samplers import PackingStraightLinePushSampler
from planning.planner import PackingPushingPlanner, VisualizationLoggingThread
from planning.cost_functions import packing_cost_function_object as objective

from perception.onrobot_tracker import OnRobotTracker

import rclpy
from rclpy.executors import MultiThreadedExecutor
# import sensor_msgs.msg Image later to prevent it from being aliased as the PIL Image class -- Important!!
# Otherwise results in very ambiguous error "AttributeError: type object 'module' has no attribute '_TYPE_SUPPORT'"
from custom_msg.msg import StampedFloat32MultiArray
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
    position_only_gripper_move_to_keep_yaw(robot_interface, current_pos, **args)
    position_only_gripper_move_to_keep_yaw(robot_interface, target_high_pos, **args)
    position_only_gripper_move_to_keep_yaw(robot_interface, target_pos, **args)



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

    frame_dict = {
        'bubble_pcs': bubble_sampled_pcs.astype(np.float32),
        'forces': np.stack([b1_force, b2_force], axis=0).astype(np.float32),
        'ee_pos': ee_pos,
    }

    return frame_dict


sync_topics = {
    '/bubble_1/depth/image_rect_raw': 'sensor_msgs/msg/Image',
    '/bubble_2/depth/image_rect_raw': 'sensor_msgs/msg/Image',
    '/ee_states': 'custom_msg/msg/StampedFloat32MultiArray',

    '/bubble_1/raw_flow': 'custom_msg/msg/StampedFloat32MultiArray',
    '/bubble_2/raw_flow': 'custom_msg/msg/StampedFloat32MultiArray',
    '/bubble_1/force': 'custom_msg/msg/StampedFloat32MultiArray',
    '/bubble_2/force': 'custom_msg/msg/StampedFloat32MultiArray',
    '/joint_states': 'sensor_msgs/msg/JointState'
}

topic_msg_dict = {k: None for k in sync_topics.keys()}

recording_callback = False
callback_records = {}


def start_recording_callback():
    global recording_callback, callback_records
    recording_callback = True
    callback_records = {k: [] for k in sync_topics.keys()}


def stop_recording_callback():
    global recording_callback
    recording_callback = False


def match_logged_data_to_requested_action(desired_ee_pos_list, topic_data):
    # given a list of desired ee positions, find the corresponding logged data based on
    # the closest ee position ("/ee_states")
    # desired_ee_pos_list: list of desired ee positions
    # topic_data_list: list of logged data, each element is a dictionary of logged data
    # return: a list of logged data, each element is a dictionary of logged data
    #         the length of the list is the same as the length of desired_ee_pos_list
    #         the logged data is the one that is closest to the desired ee position
    #         if there are multiple logged data that are equally close to the desired ee position,
    #         return the one with the smallest index
    print("Matching logged data to desired ee positions...")
    print("There are {} desired ee positions.".format(len(desired_ee_pos_list)))
    print("There are {} logged data.".format(len(topic_data['/ee_states'])))
    print("If the number of logged data is much smaller than the number of desired ee positions, "
          "consider increasing the number of logged data.")
    # get the ee positions from the logged data
    logged_ee_pos_list = np.array(topic_data['/ee_states'])[:, :2]
    # find the closest logged ee position for each desired ee position
    min_errors = []
    min_dist_indices = []
    matched_topic_data_list = []
    for i, desired_ee_pos in enumerate(desired_ee_pos_list):
        desired_ee_pos = np.array(desired_ee_pos)
        logged_ee_pos_list = np.array(logged_ee_pos_list)
        dist = np.linalg.norm(logged_ee_pos_list - desired_ee_pos[:2], axis=1)
        min_dist_index = np.argmin(dist)
        # check if there is another logged ee position that is equally close to the desired ee position (within epsilon)
        # epsilon = 0.002
        # new_mindist_idx = min_dist_index
        # for idx in range(min_dist_index+1, len(dist)):
        #     if dist[idx] - dist[min_dist_index] < epsilon:
        #         new_mindist_idx = idx
        #     else:
        #         break
        # min_dist_indices.append(new_mindist_idx)

        # take index which correspond to the last len(desired_ee_pos_list) logged data. If len(desired_ee_pos_list) is longer than the number of logged data, then the last logged data will be repeated
        if len(dist) > len(desired_ee_pos_list):
            min_dist_index = len(dist) - len(desired_ee_pos_list) + i
        else:
            if i >= len(dist):
                min_dist_index = len(dist) - 1
            else:
                min_dist_index = i
        min_dist_indices.append(min_dist_index)

        min_dist = dist[min_dist_index]
        min_errors.append(min_dist)
        # get dict of logged data
        topic_data_at_idx = {k: v[min_dist_index] for k, v in topic_data.items()}
        matched_topic_data_list.append(topic_data_at_idx)
    print("Average EE error in matched observations:", np.mean(min_errors))
    print("Min dist indices", min_dist_indices)
    print("Number of unique matched observations:", len(set(min_dist_indices)))
    return matched_topic_data_list


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
        if recording_callback:
            callback_records[topic].append(msg)


pushing_ee_height = 0.308      # 0.32 might touch the table
moving_ee_height = 0.4
fixed_z_bag = 0.308
bag_grasp_amount = -0.25

def get_initial_inhand_object_points(object_pcs):
    # identify the inhand object based on the one with the largest y value
    # return the indices of the inhand object and the object points
    # object_pcs: list of point clouds
    # return: inhand_indices, object_points
    print("Getting initial inhand object points...")
    print("There are {} objects.".format(len(object_pcs)))
    # get the y values of the object points
    y_values = [np.mean(np.asarray(pc.points)[:, 1]) for pc in object_pcs]
    # find the object with the largest y value
    inhand_index = np.argmax(y_values)
    inhand_indices = inhand_index
    object_points = np.asarray(object_pcs[inhand_index].points)
    print("Inhand object index:", inhand_index)
    return inhand_indices, object_points

def make_point_cloud(points):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    return pc

def main(args):
    global pushing_ee_height      # fixed height for moving end effector

    # obtain planning config path
    parser = argparse.ArgumentParser(description="dynamics")
    parser.add_argument(
        "-c",
        "--config",
        default=os.path.join("planning", "planning_config_packing.json"),
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
    parser.add_argument(
        "-l",
        "--log_dir",
        default="deployment_log",
        type=str,
        help="directory for logging experiments",
    )

    parser.add_argument(
        "--debug_no_perception",
        action="store_true",
        help="don't do perception to save debugging time",
    )

    planning_config = ConfigParser.from_args(parser)

    logging_queue = multiprocessing.Queue()
    logging_process = VisualizationLoggingThread(
        parameters_queue=logging_queue,
    )
    logging_process.start()

    args = parser.parse_args()
    run_robot = args.run_robot

    # set up experiment log
    exp_log_dir = os.path.join(args.log_dir, time.strftime("%Y-%m-%d"), time.strftime("%H-%M-%S"))
    os.makedirs(exp_log_dir, exist_ok=False)

    rclpy.init(args=None)
    planner_node = rclpy.create_node('mpc_planner')

    # move EE to initial pose
    robot_interface = init_franka_interface()

    if run_robot:
        open_gripper(robot_interface)
        reset_robot_z(robot_interface)
        ee_home_pose = np.array([[0.35], [0], [0.65]])  # back off and stay high to make space
        position_only_gripper_move_to(
            robot_interface, ee_home_pose, grasp=False, num_steps=70
        )
        # confirm with keyboard that scene is set up
        # input('Press enter to continue when scene setup is ready: ')

    # start subscribing to various topics
    # same as the data collection pipeline
    sync_subscriptions = []
    for k, v in sync_topics.items():
        sync_subscriptions.append(Subscriber(planner_node, globals()[v.split("/")[-1]], k))
        check_topic_status(planner_node, k)

    synchronizer = ApproximateTimeSynchronizer(
        sync_subscriptions, queue_size=200, slop=0.14, allow_headerless=False
    )
    synchronizer.registerCallback(callback)
    camera_node = CameraImageSubscriber()
    robot_node = RobotStatePublisherNode(robot_interface)
    executor = MultiThreadedExecutor()
    executor.add_node(planner_node)
    executor.add_node(camera_node)
    executor.add_node(robot_node)
    background_thread = threading.Thread(target=lambda: executor.spin())
    background_thread.start()

    if not args.debug_no_perception:
        tracker = OnRobotTracker(camera_node, verbose=True, save_gpu_vmem=planning_config["model"] != "pymunk_baseline",
                                 setting="packing")
    else:
        time.sleep(3)

    for topic, msg in topic_msg_dict.items():
        assert msg is not None, f'{topic} is None'
    print(f'messages have been received')

    object_progress = []

    # load dynamics model and config
    if planning_config["model"] == "pymunk_baseline":
        from netcompdy.envs.scripts.box_pushing_sim_dynamics import BoxPushingPymunkDynamics
        model = BoxPushingPymunkDynamics()
        model_config = ConfigParser(dict())
        # we load the config from our dynamics models, but this is just used for things like number of points per obj
        model_config.update_from_json(planning_config["dynamics_config_path"])
        print("Loaded pymunk dynamics")
    elif planning_config["model"] == "dpinet_baseline":
        from dynamics.models.dynamics_ae import DynamicsPredictor
        pretrained_path = planning_config["pretrained_path"]
        model = DynamicsPredictor.load_from_checkpoint(pretrained_path, map_location='cuda:0').to(device)
        model_config = ConfigParser(dict())
        model_config.update_from_json(planning_config["dynamics_config_path"])
        assert model_config["test_batch_size"] == 1, "test batch size must be 1"
    else:
        from dynamics.models.estimator_predictor_obj_latent_lstm_multi import DynamicsPredictor
        pretrained_path = planning_config["pretrained_path"]
        model = DynamicsPredictor.load_from_checkpoint(pretrained_path, map_location='cuda:0').to(device)
        model_config = ConfigParser(dict())
        model_config.update_from_json(planning_config["dynamics_config_path"])
        assert model_config["test_batch_size"] == 1, "test batch size must be 1"
    print(f'Dynamics model and config loaded')

    # set up tracker

    # hard code rod location
    # grab the rod from at the given location
    # observe the three back dots on the table
    if args.debug_no_perception:
        if not os.path.exists('object_pcs_0.pkl'):
            object_pcs = tracker.get_all_object_points()
            with open('object_pcs_0.pkl', 'wb') as f:
                # dump the points for each object_pcs to a pickle
                pickle.dump([np.asarray(pc.points) for pc in object_pcs], f)
        else:
            print("object_pcs_0.pkl already exists, skipping writing to pickle")
            # load it
            with open('object_pcs_0.pkl', 'rb') as f:
                object_pcs = pickle.load(f)
            # convert object_pcs into o3d point clouds
            object_pcs = [make_point_cloud(pc) for pc in object_pcs]
    else:
        object_pcs = tracker.get_all_object_points()
        with open('object_pcs_0.pkl', 'wb') as f:
            # dump the points for each object_pcs to a pickle
            pickle.dump([np.asarray(pc.points) for pc in object_pcs], f)
    print("Got object pcs", len(object_pcs))
    if run_robot:
        position_only_gripper_move_to(robot_interface,
                                      [0.4, 0.16348319530917643, 0.2963322137274104 + 0.25],
                                      grasp=False, num_steps=100)
        time.sleep(1)
        reset_robot_z(robot_interface, set_position=-np.pi/2+0.25)
        input("Place the rod on the table and press enter to continue")

        # for i in range(15):
        #     robot_interface.control(
        #         controller_type='OSC_YAW',
        #         action=[0.0, 0.0, 0.0, 0.0, 0.0, 3.14/(15)*20, -1.0],
        #         controller_cfg=get_default_controller_config('OSC_YAW'),
        #     )
        #     time.sleep(0.2)

        position_only_gripper_move_to_keep_yaw(robot_interface,
                                      [0.511314349575648, 0.16348319530917643, 0.2963322137274104 + 0.25],
                                      grasp=False, num_steps=100)

        position_only_gripper_move_to_keep_yaw(robot_interface,
                                      [0.511314349575648, 0.16348319530917643, 0.2403322137274104],
                                      grasp=False, num_steps=100)
        # save the orientation in euler angle
        current_rot, current_pos = robot_interface.last_eef_rot_and_pos
        initial_rot_euler = mat2euler(current_rot)
        hold_yaw = initial_rot_euler[2]
        close_gripper(robot_interface, amount=bag_grasp_amount)

        time.sleep(3)
        position_only_gripper_move_to_keep_yaw(robot_interface,
                                      [0.511314349575648, 0.16348319530917643, fixed_z_bag],
                                      grasp=True, num_steps=50, target_yaw=hold_yaw)

        # write the object pcs to a pickle, if they are not already there
        if args.debug_no_perception:
            if not os.path.exists('object_pcs_1.pkl'):
                object_pcs_new = tracker.get_all_object_points()
                with open('object_pcs_1.pkl', 'wb') as f:
                    # dump the points for each object_pcs to a pickle
                    pickle.dump([np.asarray(pc.points) for pc in object_pcs_new], f)
            else:
                print("object_pcs_1.pkl already exists, skipping writing to pickle")
                # load it
                with open('object_pcs_1.pkl', 'rb') as f:
                    object_pcs_new = pickle.load(f)
                # convert object_pcs into o3d point clouds
                object_pcs_new = [make_point_cloud(pc) for pc in object_pcs_new]
        else:
            object_pcs_new = tracker.get_all_object_points()
            with open('object_pcs_1.pkl', 'wb') as f:
                # dump the points for each object_pcs to a pickle
                pickle.dump([np.asarray(pc.points) for pc in object_pcs_new], f)
        inhand_indices, inhand_object_points = get_initial_inhand_object_points(object_pcs_new)
        object_pcs[inhand_indices] = object_pcs_new[inhand_indices]
        ee_pos_for_initial_inhand = np.array(robot_node.get_ee_states()[0])

        # move to a fixed starting point
        pushing_init_pos = planning_config["initial_ee_pos"]
        pushing_init_pos[-1] = moving_ee_height     # lower down height
        ee_states, _ = robot_node.get_ee_states()
        move_to_safely(robot_interface, ee_states, pushing_init_pos, grasp=True, target_yaw=hold_yaw)

    if run_robot:
        pushing_init_pos = planning_config["pushing_initialization_pos"]
        ee_states, _ = robot_node.get_ee_states()
        move_to_safely(robot_interface, ee_states, pushing_init_pos, grasp=True, target_yaw=hold_yaw)

    start = time.time()
    # Next, we start planning
    # first build a sampler
    action_dim = planning_config["action_dim"]
    initial_sampler = PackingStraightLinePushSampler(num_actions=planning_config["initial_sampler_params"]["num_actions"], allzero_allowed=False)
    replan_sampler = PackingStraightLinePushSampler(num_actions=planning_config["replan_sampler_params"]["num_actions"], allzero_allowed=True)


    # then build a planner
    initial_planner = PackingPushingPlanner(
        sampler=initial_sampler,
        model=model,
        objective=objective,
        horizon=1,
        num_samples=planning_config["initial_sampler_params"]["num_samples"],
        logging_thread=logging_queue,
    )

    replan_planner = PackingPushingPlanner(
        sampler=replan_sampler,
        model=model,
        objective=objective,
        horizon=1,
        num_samples=planning_config["replan_sampler_params"]["num_samples"],
        logging_thread=logging_queue,
    )

    # start the planning loop
    # the loop executes for num_execution_iterations, in which the robot
    # executes num_actions_to_execute_per_step actions
    # in the action plan given by the planer

    all_real_bubble_points = []
    all_real_bubble_feats = []

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
    object_pcs[inhand_indices] = make_point_cloud(inhand_object_points + (ee_states - ee_pos_for_initial_inhand)[:3])

    first_frame["object_pcs"] = [np.asarray(obj_pc.points).astype(np.float32) for obj_pc in object_pcs]
    if len(first_frame["object_pcs"]) < 6:
        print(f"Only found {len(first_frame['object_pcs'])} objects, padding to 6.")
        first_frame["object_pcs"] += [np.zeros((50, 3), dtype=np.float32) for _ in range(6 - len(first_frame["object_pcs"]))]
    first_frame["object_cls"] = [-1]
    # take the last object as the inhand object randomly
    first_frame["inhand_object_pcs"] = first_frame["object_pcs"].pop()

    # prepare the initial frame
    state_window = [first_frame]  # contains only the first frame
    state_window, _, object_masks, inhand_object_mask = downsample_points_state_dict_seq(state_window, model_config, return_mask=True)
    batch = construct_graph_from_video(model_config, state_window)

    # actually make it into a batch
    batch = pyg.data.Batch.from_data_list([batch]).cuda()

    # the following is copied from test_planning.py
    B = batch.num_graphs
    N = batch.num_nodes // B
    seq_len = batch.pos.view(B, N, -1).shape[2] // action_dim
    observed_his_len = planning_config["history_len"]

    node_pos = batch.pos.view(B, N, seq_len, -1)
    all_real_bubble_feats.append(batch.clone().cpu())

    if planning_config["model"] == "pymunk_baseline":
        tac_feat_bubbles = torch.zeros(B, 1, 40, 5).to(model.device)
    else:
        tac_feat_bubbles = model.autoencoder.encode_structured(batch)

    node_feature = batch.x.view(B, N, -1)
    action_hist = node_feature[:, :, model.type_feat_len + model.vis_feat_len:].view(B, N, seq_len - 1, action_dim)

    # get the history needed
    object_type_feat = node_feature[..., :model.type_feat_len]  # should be zero
    vision_feature = torch.zeros(B, N, model.vis_feat_len).to(object_type_feat.device)

    inhand_object_pc, table_object_pcs = PackingPushingPlanner.extract_object_pcs(node_pos[0, :, 0])  # the third axis is time. Use the first frame
    goal_points = PackingPushingPlanner.infer_bounding_box_from_pc(table_object_pcs)

    # start planning
    counter = 0
    best_real_score = 10000
    num_actions_to_execute_per_step = replan_sampler.num_actions + 1
    max_execution_steps = planning_config["max_execution_steps"]

    # log_dir should be planning/<date>/<time>
    starttime = time.strftime("%H-%M-%S")
    log_dir = os.path.join(exp_log_dir, "planning", "planning_logs")
    phys_params_log_dir = os.path.join(exp_log_dir, "planning", "phys_params")
    os.makedirs(log_dir, exist_ok=False)
    os.makedirs(phys_params_log_dir, exist_ok=True)

    mu = None       # to be iteratively updated

    # save first_frame as a pickle file in the log_dir
    with open(os.path.join(log_dir, 'first_frame.pkl'), 'wb') as f:
        pickle.dump(first_frame, f)

    # the plan-execution loop
    while True:
        print("\n\nStarting planning...")
        node_pos = node_pos[:, :, -observed_his_len:, :]
        tac_feat_bubbles = tac_feat_bubbles[:, -observed_his_len:, :, :]
        tactile_feat_objects = torch.zeros(B, tac_feat_bubbles.shape[1],
                                           model.n_object_points,
                                           model_config["ae_enc_dim"]).to(model.device)
        tactile_feat_all_points = torch.cat([tactile_feat_objects, tac_feat_bubbles], dim=2)
        action_hist = action_hist[:, :, -(observed_his_len - 1):, :]

        # Generate the initial plan
        # print(object_type_feat.shape, vision_feature.shape, node_pos.shape, tac_feat_bubbles.shape,
        #       tactile_feat_all_points.shape)

        planner = initial_planner if counter == 0 else replan_planner
        if planner == initial_planner:
            print("Using initial planner")
        else:
            print("Using replan planner")

        # create log dir under "planning" folder with the current time as the name
        pre_plan_mu = mu    # for saving
        action, best_action_prediction = planner.plan(
            t=counter,
            log_dir=log_dir,
            observation_batch=(object_type_feat,
                               vision_feature,
                               node_pos,
                               tac_feat_bubbles,
                               tactile_feat_all_points),
            action_history=action_hist,
            visualize_top_k=planning_config["visualize_top_k"],
            goal_points=goal_points
        )

        planning_time_node_pos = node_pos.cpu().numpy().copy()
        # print("planning time node pos", node_pos)

        # print("Found plan:", action)
        # execute the best action
        # merge adjacent duplicate actions into one for easy actuation, if deemed suitable
        # actions = action[:num_actions_to_execute_per_step]
        TO_TAKE = 45
        actions = action[:TO_TAKE]

        # obtain ee trajectory
        current_pos = ee_states
        desired_ee_positions = np.cumsum(actions, axis=0) + current_pos.squeeze()

        # force z value of the desired ee position to be the same as the pushing_ee_height
        desired_ee_positions[:, -1] = pushing_ee_height

        # save key variables to local disk
        # perhaps we can save all inputs and outputs of the planner for offline reproduction
        save_dict = {
            'plan_idx': counter,
            'log_dir': log_dir,
            'observation_batch': (object_type_feat,
                                  vision_feature,
                                  node_pos,
                                  tac_feat_bubbles,
                                  tactile_feat_all_points),
            'action_history': action_hist,
            'pre_planning_mu': pre_plan_mu,
            'post_planning_mu': mu,
            'actions': actions,
            'best_action_prediction': best_action_prediction,
            'desired_ee_positions': desired_ee_positions,
            'planning_config': planning_config
        }
        with open(os.path.join(exp_log_dir, f"plan_{counter}.pkl"), 'wb') as file:
            pickle.dump(save_dict, file)
            print(f'\tplanner configuration {counter} saved to disk')

        camera_node.start_recording_rgb()

        # move to high position
        if run_robot:
            high_pos = desired_ee_positions[0].copy()
            high_pos[-1] = high_pos[-1]
            position_only_gripper_move_to_keep_yaw(robot_interface, high_pos[:, None], grasp=True, num_steps=200,
                                          allowance=0.005, target_yaw=hold_yaw)
            position_only_gripper_move_to_keep_yaw(robot_interface, desired_ee_positions[0][:, None], grasp=True, num_steps=200,
                                          allowance=0.0025, target_yaw=hold_yaw)
            time.sleep(1)
            num_recorded_frames = camera_node.num_recorded_frames()
            if len(np.nonzero(actions)) > 0:
                start_recording_callback()
                position_only_gripper_move_to_waypoints_keep_yaw(robot_interface, desired_ee_positions[1:][..., None], grasp=True, num_steps=1000,
                                              allowance=0.005, max_speed=0.3, min_speed=0.3, target_yaw=hold_yaw)
                time.sleep(1)
                stop_recording_callback()

        # update the state history using the observations
        real_observations = match_logged_data_to_requested_action(desired_ee_positions, callback_records)

        for iteration, (real_obs, curr_pred_state_dict) in enumerate(zip(real_observations, best_action_prediction)):
            ee_states, ee_orient = real_obs['/ee_states'][:3], real_obs['/ee_states'][3:],
            ee_states, ee_orient = np.array(ee_states), np.array(ee_orient)
            gripper_state = real_obs['/joint_states'][-1]
            gripper_state = np.array([gripper_state])
            curr_frame = get_first_frame(ee_states, ee_orient, gripper_state.tolist()[-1],
                                            real_obs['/bubble_1/raw_flow'],
                                            real_obs['/bubble_2/raw_flow'],
                                            real_obs['/bubble_1/force'],
                                            real_obs['/bubble_2/force'],
                                            real_obs['/bubble_1/depth/image_rect_raw'],
                                            real_obs['/bubble_2/depth/image_rect_raw']
                                            )
            curr_pred_state_dict = {k: torch.from_numpy(v).cuda() for k, v in curr_pred_state_dict.items()}
            object_points = curr_pred_state_dict['object_obs']
            if iteration == len(actions) - 1:
                if run_robot:
                    # before taking visual observation,
                    # raise end effector to avoid occlusion by robot
                    # position_only_gripper_move_to(robot_interface, [ee_states[0], ee_states[1],
                    #                                                 very_high_z-0.05],
                    #                               grasp=True, num_steps=150)
                    time.sleep(1)
                    camera_node.stop_recording_rgb()
                    if planning_config["intermediate_tracking_steps"] > 0:
                        tracker.perform_intermediate_tracking_steps(camera_node.get_intermediate_frames(num_recorded_frames,
                                                                                                        planning_config["intermediate_tracking_steps"]))

                    object_pcs[inhand_indices] = make_point_cloud(
                        inhand_object_points + (np.array(robot_node.get_ee_states()[0]) - ee_pos_for_initial_inhand)[:3])
                    # print the center location of the inhand ob ject
                    current_packing_object_approx_center = np.mean(np.asarray(object_pcs[inhand_indices].points), axis=0)
                    print("current inhand approx center:", current_packing_object_approx_center)

                    if planning_config["use_visual_feedback"]:
                        # read point cloud from camera

                        if args.debug_no_perception:
                            if not os.path.exists('object_pcs_0.pkl'):
                                object_pcs = tracker.get_all_object_points()
                                with open('object_pcs_0.pkl', 'wb') as f:
                                    # dump the points for each object_pcs to a pickle
                                    pickle.dump([np.asarray(pc.points) for pc in object_pcs], f)
                            else:
                                print("object_pcs_0.pkl already exists, skipping writing to pickle")
                                # load it
                                with open('object_pcs_0.pkl', 'rb') as f:
                                    object_pcs = pickle.load(f)
                                # convert object_pcs into o3d point clouds
                                object_pcs = [make_point_cloud(pc) for pc in object_pcs]
                        else:
                            object_pcs = tracker.get_all_object_points()
                        curr_frame["object_pcs"] = [np.asarray(obj_pc.points).astype(np.float32) for obj_pc in
                                                     object_pcs]
                        print(f'object points updated based on feedback ')
                        if len(curr_frame["object_pcs"]) < 6:
                            print("Only found {} objects, padding to 6.")
                            curr_frame["object_pcs"] += [np.zeros((50, 3), dtype=np.float32) for _ in
                                                          range(6 - len(first_frame["object_pcs"]))]
                        curr_frame["object_cls"] = [-1]
                        # take the last object as the inhand object randomly
                        curr_frame["inhand_object_pcs"] = curr_frame["object_pcs"].pop()
                        state_window = downsample_points_state_dict_seq([curr_frame], model_config,
                                                                        object_masks=object_masks,
                                                                        inhand_object_mask=inhand_object_mask)
                        curr_frame = state_window[0]
                        # update the prediction state dict
                        object_points[:, :3] = torch.from_numpy(np.concatenate([p for p in curr_frame["object_pcs"]] + [curr_frame["inhand_object_pcs"]])).cuda()
                        object_points[:, 3:] = 0
                        curr_frame["bubble_pcs"] = curr_frame["bubble_pcs"][:, 0]
                        curr_pred_state_dict['object_obs'] = object_points
                    else:
                        curr_frame["object_pcs"] = object_points.cpu().numpy().astype(np.float32)
                        # split into 6 objects of 20 points each
                        curr_frame["object_pcs"] = np.split(curr_frame["object_pcs"], 6, axis=0)
                        curr_frame["inhand_object_pcs"] = curr_frame["object_pcs"].pop()
                        object_points_from_observation = object_points
            else:
                curr_frame["object_pcs"] = object_points.cpu().numpy().astype(np.float32)
                # split into 6 objects of 20 points each
                curr_frame["object_pcs"] = np.split(curr_frame["object_pcs"], 6, axis=0)
                curr_frame["inhand_object_pcs"] = curr_frame["object_pcs"].pop()

            curr_frame["object_cls"] = [-1]
            predicted_bubble = curr_pred_state_dict['bubble'].view(2, 20, -1).cpu().numpy().astype(np.float32)[..., :3]
            real_bubble = curr_frame['bubble_pcs'].astype(np.float32)[..., :3]

            all_real_bubble_points.append(real_bubble)
            diff = np.median(np.abs(predicted_bubble - real_bubble)[..., :-1].mean(axis=-1).flatten())
            # print(f'diff between real and predicted bubble in x and y axes: {diff:.3f}')
            # import pdb; pdb.set_trace()
            # print(f'diff between real and predicted bubble: {np.abs(predicted_bubble - real_bubble).mean(axis=-1)}')

            if run_robot and diff > 0.05:
                print('\tdiff too large, please check why')
                predicted_bubble_o3d = xyz_to_pc(predicted_bubble.reshape(-1, 3))
                predicted_bubble_o3d.paint_uniform_color([1, 0, 0])
                real_bubble_o3d = xyz_to_pc(real_bubble.reshape(-1, 3))
                real_bubble_o3d.paint_uniform_color([0, 1, 0])
                o3d.visualization.draw_geometries([predicted_bubble_o3d, real_bubble_o3d])
                breakpoint()

            dummy_batch = construct_graph_from_video(model_config, [curr_frame])
            dummy_batch = pyg.data.Batch.from_data_list([dummy_batch]).cuda()
            all_real_bubble_feats.append(dummy_batch.clone().cpu())
            if planning_config["model"] == "pymunk_baseline":
                curr_tac_feat_bubbles_real = torch.zeros(1, 1, 40, 5).to(model.device)
            else:
                curr_tac_feat_bubbles_real = model.autoencoder.encode_structured(
                    dummy_batch)  # shape torch.Size([1, 1, 40, 5])

            # update bubble points based on feedback
            curr_pred_state_dict['bubble'][..., :3] = torch.from_numpy(real_bubble.reshape(-1, 3)).cuda()
            # update inhand points based on feedback
            # curr_pred_state_dict['inhand'] = torch.from_numpy(curr_frame['inhand_object_pcs']).cuda()
            # update particle history
            curr_pos = torch.cat([curr_pred_state_dict['object_obs'],
                                         curr_pred_state_dict['bubble']], dim=0).unsqueeze(0).unsqueeze(2)
            node_pos = torch.cat([node_pos, curr_pos[..., :3]], dim=2)  # shape (B, N, T, 3)
            tac_feat_bubbles = torch.cat([tac_feat_bubbles, curr_tac_feat_bubbles_real], dim=1)  # shape (B, T, N, 3)
            # update action history
            action = actions[iteration]
            # if iteration == 0:
            #     action = actions[iteration]
            # else:
            #     action = (node_pos[0, :, -1][20:40] - node_pos[0, :, -2][20:40]).mean(axis=0).cpu().numpy()
            action = torch.from_numpy(action).unsqueeze(0).unsqueeze(0).unsqueeze(0).cuda()
            action = action.repeat(1, N, 1, 1)
            action_hist = torch.cat([action_hist, action], dim=2)  # shape (B, N, T, 3)

            print(f'Current tactile history length: {tac_feat_bubbles.shape[1]}')

        # if (current_packing_object_approx_center[1] < -0.07 and current_packing_object_approx_center[0] > 0.45) or \
        #         (current_packing_object_approx_center[1] < -0.06 and current_packing_object_approx_center[0] < 0.45):
        curr_ee_state = robot_node.get_ee_states()[0]
        print("Curr ee state:", curr_ee_state)
        if (curr_ee_state[1] < -0.06 and curr_ee_state[0] < 0.45) or (curr_ee_state[1] < -0.06 and curr_ee_state[0] > 0.45):
            print("Current packing object approx center is less than threshold, try to pack!")
            pack = input("Pack? \n ")
            if pack.lower() == "y":
                camera_node.start_recording_rgb()
                # try to drop the object
                curr_ee_pos = np.array(robot_node.get_ee_states()[0])[:3]
                # lower by 4cm
                # curr_ee_pos[-1] -= 0.06
                curr_ee_pos[-1] -= 0.04
                position_only_gripper_move_to_keep_yaw(robot_interface,
                                                  list(curr_ee_pos),
                                                  grasp=True, num_steps=100, target_yaw=hold_yaw)
                # open gripper
                open_gripper(robot_interface)
                time.sleep(1)
                # make the robot move up out of the way
                curr_ee_pos[2] += 0.24
                position_only_gripper_move_to_keep_yaw(robot_interface,
                                                  list(curr_ee_pos),
                                                  grasp=False, num_steps=100, target_yaw=hold_yaw)
                camera_node.stop_recording_rgb()
                counter = 1000000

        real_cost = 100
        # real_cost = -1 * objective({"object_obs": object_points_from_observation[:, :3].cpu().numpy()[None, None]}, goal,
        #                            last_state_only=True).item()
        best_real_score = min(best_real_score, real_cost)
        print(f'\tThe real cost is {real_cost:.3f}!!!!!')
        if (best_real_score == real_cost):
            print(f'\tbest real score updated to {best_real_score:.3f}!!!!!')
        print(f'Best real score so far: {best_real_score:.3f}')

        if np.abs(real_cost) <= 0.01:
            print(f'real cost very low, goal reached, stop. ')
            counter = 1000000

        object_progress.append(
            {
                "object_points": object_points_from_observation[:, :3].cpu().numpy(),
                "real_score": real_cost,
                "best_real_cost": best_real_score
            }
        )

        # save object progress to exp_log_dir
        with open(os.path.join(exp_log_dir, "object_progress.pkl"), 'wb') as file:
            pickle.dump(object_progress, file)
            print(f'\tprogress saved to disk')

        print(f"Executed and updated state history, counter = {counter}")
        if not (planning_config["model"] == "pymunk_baseline" or planning_config["model"] == "dpinet_baseline"):
            np.save(f'{phys_params_log_dir}/physics_params_{starttime}.npy', model.pred_physics_list, allow_pickle=True)
        # save bubble feats as pickle file
        with open(f'{phys_params_log_dir}/bubble_feats_{starttime}.pkl', 'wb') as f:
            pickle.dump(all_real_bubble_feats, f)
         # check stop condition

        # update counter
        # if counter % 5 == 0:
        #     camera_node.log_recorded_frames(exp_log_dir)
        counter += 1

        if counter * num_actions_to_execute_per_step > max_execution_steps:
            print(f'num_execution_iterations = {max_execution_steps} reached. Stop.')
            print(f'best real score: {best_real_score:.3f}')
            camera_node.log_recorded_frames(exp_log_dir)
            print("------------------------------------------------------------------")
            print("----------------------- D  O  N  E  :  ) -------------------------")
            print("------------------------------------------------------------------")
            break
        if counter > planning_config["max_plan_executions"]:
            print(f'num_execution_iterations = {planning_config["max_plan_executions"]} reached. Stop.')
            print(f'best real score: {best_real_score:.3f}')
            camera_node.log_recorded_frames(exp_log_dir)
            print("------------------------------------------------------------------")
            print("----------------------- D  O  N  E  :  ) -------------------------")
            print("------------------------------------------------------------------")
            break

if __name__ == '__main__':
    main(None)

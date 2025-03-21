import sys
from utils_deoxys import init_franka_interface
sys.path.append('/home/albert/github/robopack/ros2_numpy')

import ros2_numpy

import pdb
import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor
from deoxys.experimental.motion_utils import position_only_gripper_move_to

from perception.camera_subscriber import CameraSubscriber
from perception.utils_sam import *
from perception.utils_cv import *
from perception.utils_ros import *
from utils_general import *
from control.input_utils import input2action

from control.robot_state_publisher import RobotStatePublisherNode

from deoxys import config_root
from deoxys.utils import YamlConfig
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.log_utils import get_deoxys_example_logger


data_root = "/media/albert/ExternalHDD/bubble_data/v4_0711"
logger = get_deoxys_example_logger()

from config.task_config import *


def reconstruct_single_object_and_save(cam0_pc, cam1_pc, model, predictor):
    cam0_rgb, cam0_depth = point_cloud_to_rgbd(cam0_pc, intrinsic_matrices['cam_0'], 720, 1280)
    cam1_rgb, cam1_depth = point_cloud_to_rgbd(cam1_pc, intrinsic_matrices['cam_1'], 720, 1280)

    cam0_masks, boxes_filt, pred_phrases = predict(model, predictor, cam0_rgb, "objects")
    cam1_masks, boxes_filt, pred_phrases = predict(model, predictor, cam1_rgb, "objects")

    cam0_masks = remove_redundant_images(cam0_masks.squeeze(1), threshold=100)
    cam1_masks = remove_redundant_images(cam1_masks.squeeze(1), threshold=100)

    cam0_pcs = get_point_cloud_of_every_object(cam0_masks, cam0_rgb, cam0_depth, intrinsics['cam_0'],
                                               extrinsics['cam_0'])
    cam1_pcs = get_point_cloud_of_every_object(cam1_masks, cam1_rgb, cam1_depth, intrinsics['cam_1'],
                                               extrinsics['cam_1'])
    object_pcs = merge_multiview_object_pcs(cam0_pcs, cam1_pcs)

    # get the largest object
    # max_cloud = object_pcs[0]
    # for pc in object_pcs:
    #     if np.asarray(pc.points).shape[0] < np.asarray(max_cloud.points).shape[0]:
    #         max_cloud = pc

    # object_pcs = [max_cloud]
    o3d.visualization.draw_geometries(object_pcs)
    # pdb.set_trace()

    denoised_sampled_pcs = []
    for i, object_pc in enumerate(object_pcs):
        object_pc += fill_point_cloud_by_downward_projection(object_pc)
        # object_pc2 = denoise_pc_by_stats(object_pc1, denoise_depth=2)
        # denoised_sampled_pc = sample_points_from_pc(object_pc, n_points=N_POINTS_PER_OBJ, ratio=None,
        #                                             mesh_fix=True)
        object_pc = farthest_point_sampling_dgl_pc(object_pc, N_POINTS_PER_OBJ)
        denoised_sampled_pcs.append(object_pc)

    # assert len(denoised_sampled_pcs) == 1
    o3d.visualization.draw_geometries(denoised_sampled_pcs)
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


def main(args):
    # start interface
    robot_interface = init_franka_interface()
    ee_home_pose = np.array([[0.35], [0], [0.50]])    # back off and stay high to make space
    position_only_gripper_move_to(
        robot_interface, ee_home_pose
    )

    # start publishing robot state in the background
    rclpy.init(args=None)
    camera_node = CameraSubscriber()      # listen for two cameras
    robot_node = RobotStatePublisherNode(robot_interface)
    executor = MultiThreadedExecutor()
    executor.add_node(camera_node)
    executor.add_node(robot_node)
    background_thread = threading.Thread(target=lambda: executor.spin())
    background_thread.start()
    time.sleep(2)       # wait for the image to be received

    assert camera_node.get_last_reading() is not None, 'no messages received from the cameras'

    user_input = None
    while user_input != 'c':
        user_input = input("Place in-hand object on the table. "
                           "Enter c to start scanning: ")

    object_data_dir = create_sequential_folder(data_root)
    cam0_pc, cam1_pc = [decode_pointcloud_msg(msg) for msg in camera_node.get_last_reading()[:2]]
    model, predictor = load_models()
    model_data_path = os.path.join(object_data_dir, 'inhand_obj.h5')
    model_data_dict = reconstruct_single_object_and_save(cam0_pc, cam1_pc, model, predictor)

    # start teleop
    user_input = None
    while user_input != 'c':
        user_input = input("Next, teleop to grab the box. Press c when ready: ")

    device = SpaceMouse(vendor_id=vendor_id, product_id=product_id)
    device.start_control()

    controller_cfg = YamlConfig(
        os.path.join(config_root, controller_cfg_file)
    ).as_easydict()

    while True:
        action, grasp = input2action(
            device=device,
            controller_type=controller_type,
        )
        if action is None:      # right bottom on spacemouse
            break

        # set unused orientation dims to 0
        action[3:6] = 0.0
        logger.info(action)

        robot_interface.control(
            controller_type=controller_type,
            action=action,
            controller_cfg=controller_cfg,
        )

    # record ee data
    from calibration.publish_joint_state_ros2 import get_joint_states, get_ee_states
    ee_pos, ee_rot = get_ee_states(robot_interface)
    model_data_dict['ee_states'] = np.array(ee_pos + ee_rot)
    joint_pos, joint_vel, gripper_state = get_joint_states(robot_interface)
    position = joint_pos + [gripper_state]
    model_data_dict['joint_states'] = np.array(position)
    save_dictionary_to_hdf5(model_data_dict, model_data_path)

    # start tele operation
    user_input = None
    while user_input != 'c':
        user_input = input("Ready for teleop. \n"
                           f"Please place other objects onto the table,"
                           f" start recording to {object_data_dir} and enter c: ")

    device._reset_state = 0
    device._enabled = True
    while True:
        action, grasp = input2action(
            device=device,
            controller_type=controller_type,
        )
        if action is None:
            break

        # set unused orientation dims to 0
        action[3:6] = 0.0
        logger.info(action)

        robot_interface.control(
            controller_type=controller_type,
            action=action,
            controller_cfg=controller_cfg,
        )


if __name__ == "__main__":
    main(None)

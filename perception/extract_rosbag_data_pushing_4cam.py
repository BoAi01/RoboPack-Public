#!/usr/bin/env python3

import pdb
import sys

import numpy as np
import open3d.cuda.pybind.visualization.rendering
import open3d.visualization
import rosbags

sys.path.append('/home/albert/github/robopack')
sys.path.append('/home/albert/github/robopack/ros2_numpy')

from config.task_config import N_POINTS_PER_OBJ, extrinsics, intrinsics_depth, intrinsic_matrices

import tqdm
import pdb
import time

from rclpy.serialization import deserialize_message
from rosbags.rosbag2 import Reader

from perception.utils_ros import *
from perception.utils_gripper import *
from perception.utils_cv import *
from perception.utils_sam import *
from utils_general import *

# the below needs to be imported at last, not sure why
from sensor_msgs.msg import PointCloud2, Image, JointState
from std_msgs.msg import Float32MultiArray


"""
The scripts extracts the data from ros bag files. Each trajectory is one h5 file located at
 data_vxxx_parsed/seq_x/rosbagx.h5, containing one dictionary that has the following key-value format:

object_pcs: particles of the object with shape (T, N, NUM_POINTS, 6), where T is the number of time steps, 
    N is the number of objects, NUM_POINTS is the number of points per object, and 6 indicates xyz + rgb.
bubble_pcs: particles of the two soft bubbles with shape (T, 2, NUM_POINTS, 6), the same as the above, 
    except that N = 2 (left and right bubbles).
ee_states: state of the end effector with shape (T, 7), where 7 indicates xyz + wxyz (note it's not xyzw).
joint_states: state of the 7 DOF joints and additionally, the finger distance, with shape (T, 8).
forces: force distribution of sensed by the two soft bubbles with shape (T, 2, 7).
inhand_object_pcs: point cloud of the in-hand object with shape (T, NUM_POINTS). 
flows: flow information extracted by the two soft bubbles with shape (T, 2, 240, 320, 3), where 2 indicates 
    the left and right bubbles, and (240, 320, 3) is the flow image shape. 
    
Note that all point clouds are in the robot's coordinate system. 
"""


topics_to_collect = {
    '/bubble_1/depth/color/points': 'sensor_msgs/msg/PointCloud2',
    '/bubble_2/depth/color/points': 'sensor_msgs/msg/PointCloud2',
    '/cam_0/depth/color/points': 'sensor_msgs/msg/PointCloud2',
    '/cam_1/depth/color/points': 'sensor_msgs/msg/PointCloud2',
    '/cam_2/depth/color/points': 'sensor_msgs/msg/PointCloud2',
    '/cam_3/depth/color/points': 'sensor_msgs/msg/PointCloud2',
    # '/bubble_1/flow': 'sensor_msgs/msg/Image',
    '/bubble_1/raw_flow': 'std_msgs/msg/Float32MultiArray',
    '/bubble_1/force': 'std_msgs/msg/Float32MultiArray',
    # '/bubble_2/flow': 'sensor_msgs/msg/Image',
    '/bubble_2/raw_flow': 'std_msgs/msg/Float32MultiArray',
    '/bubble_2/force': 'std_msgs/msg/Float32MultiArray',
    '/bubble_C0A594A050583234322E3120FF0D2108/pressure': 'std_msgs/msg/Float32',
    '/bubble_AF6E04F550555733362E3120FF0F3217/pressure': 'std_msgs/msg/Float32',
    '/ee_states': 'std_msgs/msg/Float32MultiArray',
    '/joint_states': 'sensor_msgs/msg/JointState'
}

topics_in_h5 = {
    '/bubble_C0A594A050583234322E3120FF0D2108/pressure': 'std_msgs/msg/Float32',
    '/bubble_AF6E04F550555733362E3120FF0F3217/pressure': 'std_msgs/msg/Float32',
}

topic_to_class = {
    'sensor_msgs/msg/PointCloud2': PointCloud2,
    'sensor_msgs/msg/Image': Image,
    'std_msgs/msg/Float32MultiArray': Float32MultiArray,
    'sensor_msgs/msg/JointState': JointState
}

# def estimate_inhand_object_pose(obj_pc, ee_pose, finger_dist):
#     """
#     Returns the transformation matrix from the object to the end effector in the robot frame.
#     :param obj_pc: object point cloud
#     :param ee_pose:
#     :param finger_dist:
#     :return:
#     """
#     obj_pos = obj_pc.get_center()
#     obj_wxyz = np.array([1, 0, 0, 0])  # no rotation
#     t_obj_to_robot = pos_quat_to_matrix(obj_pos, obj_wxyz)
#     t_ee_to_robot = pos_quat_to_matrix(ee_pose[:3], ee_pose[3:])
#     t_obj_to_ee = np.linalg.inv(t_ee_to_robot) @ t_obj_to_robot   # robot to obj
#     pass


def read_rosbag(path, every_n_frames, max_frame_count):
    # dics to save the messages
    data_dict = {}
    count_dict = {}

    # first read data from h5 dict
    other_topics = np.load(os.path.join(path, 'other_topics.npy'), allow_pickle=True).item()
    for topic, msg_list in other_topics.items():
        if not topic in data_dict:
            data_dict[topic] = []
        for i, msg in enumerate(msg_list):
            if i % every_n_frames != 0:
                continue
            if i >= max_frame_count * every_n_frames:
                continue
            data_dict[topic].append(msg)

    # create reader instance and open for reading
    with Reader(path) as reader:
        # create a progress bar
        # progress_bar = tqdm.tqdm(total=sum(1 for _ in reader.messages()), unit="message")

        # iterate over messages
        for connection, timestamp, rawdata in tqdm.tqdm(reader.messages()):
            # filter out unused messages
            if connection.topic not in topics_to_collect:
                continue

            # create key in dict
            # if not connection.topic in msgs_dict:
            #     msgs_dict[connection.topic] = []
            if not connection.topic in data_dict:
                data_dict[connection.topic] = []
            if not connection.topic in count_dict:
                count_dict[connection.topic] = 0

            count_dict[connection.topic] += 1
            # skip if this is not the right n-th message, where n is a multiple of every_n_frames
            if count_dict[connection.topic] % every_n_frames != 0:
                continue

            # deserialize message
            msg = deserialize_message(rawdata, topic_to_class[connection.msgtype])

            data = None
            # decode message
            if 'Image' in connection.msgtype:
                if 'color' in connection.topic:
                    data = decode_img_msg(msg)
                else:  # depth
                    data = decode_img_msg(msg, clip_value=2000)
            elif 'PointCloud2' in connection.msgtype:
                data = decode_pointcloud_msg(msg)
            elif 'Float32MultiArray' in connection.msgtype:
                data = decode_multiarray_msg(msg)
            elif 'JointState' in connection.msgtype:
                data = decode_joint_state_msg(msg)
            else:
                raise NotImplementedError(f'unknown message type {connection.msgtype}')

            data_dict[connection.topic].append(data)
            # msgs_dict[connection.topic].append(msg)

            # progress bar
            # progress_bar.update()

            is_exit = True
            for topic, count in count_dict.items():
                is_exit = is_exit & (count >= max_frame_count * every_n_frames)    # 1000 frames would take up RAM

            if is_exit:
                return data_dict

    return data_dict


# model, predictor = load_models()


def is_pc_center_in_workspace(pc):
    center = pc.get_center()
    return (center <= np.array([xrange[1], yrange[1], zrange[1]])).all() \
        and (center >= np.array([xrange[0], yrange[0], zrange[0]])).all()


# def get_box_point_cloud(*raw_pcs):
#     min_height = 0.005
#     max_height = 0.05
#
#     pcs = []
#     for i, pc in enumerate(raw_pcs):
#         pc = project_pc(convert_pc_optical_color_to_link_frame(pc, i), extrinsics[f'cam_{i}'])
#         pc = extract_workspace_pc(pc, xrange, yrange, zrange)
#         pcs.append(pc)
#
#     object_pc = filter_point_cloud_by_height(pcs[0] + pcs[1] + pcs[2] + pcs[3], min_height, max_height)
#
#     # o3d.visualization.draw_geometries([object_pc])
#     # import pdb
#     # pdb.set_trace()
#
#     return fill_point_cloud_by_downward_projection(object_pc) + object_pc


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


rod_color_mean = np.array([0.0353799, 0.19466609, 0.30409592])
rod_color_std = np.array([0.03177297, 0.01785385, 0.01702595])


def segment_blue_rod_points(pc):


    # Extract the colors from the point cloud
    colors = np.asarray(pc.colors)

    # Calculate the lower and upper bounds for each channel (R, G, B)
    lower_bounds = rod_color_mean - 12 * rod_color_std
    upper_bounds = rod_color_mean + 12 * rod_color_std

    # Create boolean masks for each channel to filter the points
    red_mask = np.logical_and(colors[:, 0] >= lower_bounds[0], colors[:, 0] <= upper_bounds[0])
    green_mask = np.logical_and(colors[:, 1] >= lower_bounds[1], colors[:, 1] <= upper_bounds[1])
    blue_mask = np.logical_and(colors[:, 2] >= lower_bounds[2], colors[:, 2] <= upper_bounds[2])

    # Combine the masks to get the final mask for filtering
    combined_mask = np.logical_and.reduce((red_mask, green_mask, blue_mask))

    # Filter the points based on the mask
    rod = pc.select_by_index(np.where(combined_mask)[0])
    rest = pc.select_by_index(np.where(~combined_mask)[0])

    return rod, rest


def main(args):
    """
    For each folder named seq_x in the source path, read the rosbag files,
    reconstruct the scene, sample the point clouds, and save the training data
    into a h5 file.
    :param args: user-input args
    :return: None
    """

    for i, seq_folder in enumerate(os.listdir(args.source_path)):
        if 'seq_3' in seq_folder or 'seq_2' in seq_folder:
            print(f'skipping {seq_folder}')
            continue

        seq_folder_path = os.path.join(args.source_path, seq_folder)
        print(f'[{i+1}/{len(os.listdir(args.source_path))}] start reading {seq_folder_path}')

        if not os.path.isdir(seq_folder_path):
            continue

        # read in-hand object data
        in_hand_obj_dict = load_dictionary_from_hdf5(os.path.join(seq_folder_path, 'inhand_obj.h5'))
        in_hand_pc = xyz_colors_to_pc(in_hand_obj_dict['sampled_points'])

        if in_hand_obj_dict['ee_states'][2] > 0.35:     # meaning it might have an error
            warnings.warn(f"ee state is {in_hand_obj_dict['ee_states'][:3]}, which might contain error. Setting this"
                          f"to 0.30 instead")
            in_hand_obj_dict['ee_states'][:3] = in_hand_pc.get_center()
            in_hand_obj_dict['ee_states'][2] = 0.297

        lower_adjustment = 0.015        # make the rod lower, manual adjustment
        estimated_rod_center = in_hand_pc.get_center()[:2].tolist() + [0.125 - lower_adjustment]

        t_init_ee_to_robot = pos_quat_to_matrix(in_hand_obj_dict['ee_states'][:3],
                                                in_hand_obj_dict['ee_states'][3:])
        t_obj_to_robot = pos_quat_to_matrix(estimated_rod_center, [1, 0, 0, 0])
        t_obj_to_init_ee = np.linalg.inv(t_init_ee_to_robot) @ t_obj_to_robot
        t_obj_to_init_ee[1, -1] = 0  # assume no movement along y axis
        # centered_in_hand_pc = project_pc(in_hand_pc, np.linalg.inv(t_obj_to_robot))

        # replace observed point cloud with 3d model
        rod_model_path = '/home/albert/Data/rod_3x3x25_fixed_Aug1.ply'
        rod_model = o3d.io.read_triangle_mesh(rod_model_path)
        # rod_model.transform(t_init_ee_to_robot)
        # rod_model.transform(pos_quat_to_matrix([0.015, 0, -(in_hand_obj_dict['ee_states'][2] - 0.125)],
        #                                        [1, 0, 0, 0]))
        rod_model_points = sample_points_from_mesh(rod_model, N_POINTS_PER_OBJ)
        centered_in_hand_model_points = rod_model_points.paint_uniform_color(rod_color_mean)
        # centered_in_hand_model_points = project_pc(rod_model_points, np.linalg.inv(t_obj_to_robot))

        # for debugging
        # if args.debug:
        #     plane, ee, origin = get_origin_ee_point_cloud(in_hand_obj_dict['ee_states'][:3])
        #     o3d.visualization.draw_geometries([plane, ee, origin, centered_in_hand_model_points, centered_in_hand_pc])
        #     import pdb
        #     pdb.set_trace()

        # replace the observed pc with the mesh pc
        centered_in_hand_pc = centered_in_hand_model_points

        for j, rosbag_folder in enumerate(os.listdir(seq_folder_path)):
            if 'rosbag' not in rosbag_folder:
                continue
            # if not ('7' in rosbag_folder  or '8' in rosbag_folder):
            #     continue
            os.makedirs(os.path.join(args.target_path, seq_folder), exist_ok=True)
            h5_path = os.path.join(args.target_path, seq_folder, f'{rosbag_folder}.h5')
            if os.path.exists(h5_path):
                print(f'skipping {h5_path} because it already exists')
                continue

            rosbag_folder_path = os.path.join(seq_folder_path, rosbag_folder)
            print(f'\t[{i}/{len(os.listdir(seq_folder_path))}] start reading rosbag {rosbag_folder_path}')

            try:
                data_dict = read_rosbag(rosbag_folder_path, args.every_n_frames, args.max_nframe)
            except rosbags.rosbag2.errors.ReaderError as e:
                print(f'error occurred reading rosbag: {str(e)}')
                print(f'skipping this one... ')
                continue

            # check missing topics
            missing_topics = []
            for topic in topics_to_collect:
                if not topic in data_dict:
                    missing_topics.append(topic)
            assert len(missing_topics) == 0, f"missing topics: {missing_topics}"

            # adjust asynchronization between topics
            # move robot states messages backwards
            # synchronization_offset = 0
            # for topic, data in data_dict.items():
            #     if not 'states' in topic:
            #         data_dict[topic] = data[synchronization_offset:]
            #     else:
            #         data_dict[topic] = data[:-synchronization_offset]

            lens = []
            for topic, data in data_dict.items():
                lens.append(len(data))
            # print(f'max length - min length = {max(lens) - min(lens)}')
            print(lens)
            try:
                assert max(lens) > min(lens) * 1.2, f"there is some synchronization problem in rosbag"
            except:
                # import pdb
                # pdb.set_trace()
                warnings.warn('there might be sync problem')

            # read data every some frames
            min_len = min(lens)
            save_data_dict = {
                'object_pcs': [],
                'bubble_pcs': [],
                'inhand_object_pcs': [],
                'forces': [],
                # 'flows': [],
                'pressure': [],
                'ee_states': [],
                'joint_states': []
            }
            saved_n_frame = 0
            for idx in tqdm.tqdm(range(0, min_len)):        # 8, 9
                # if idx < 15:
                #     continue
                cur_time = time.time()

                # robot-related info
                ee_pose = data_dict['/ee_states'][idx]
                ee_pos, ee_orient = ee_pose[:3], ee_pose[3:]
                gripper_dist = data_dict['/joint_states'][idx][-1]
                t_bubbles_to_robot = get_bubble_cameras_to_robot(ee_pos, ee_orient, gripper_dist)

                # tactile sensor reading
                bubble1_pc = data_dict['/bubble_1/depth/color/points'][idx]
                bubble2_pc = data_dict['/bubble_2/depth/color/points'][idx]

                bubble_1_rgb, bubble_1_depth = point_cloud_to_rgbd(bubble1_pc, intrinsic_matrices['bubble_1'],
                                                                   480, 640)
                bubble_2_rgb, bubble_2_depth = point_cloud_to_rgbd(bubble2_pc, intrinsic_matrices['bubble_2'],
                                                                   480, 640)

                bubble_1_flow = data_dict['/bubble_1/raw_flow'][idx].reshape(240, 320, 2)
                bubble_2_flow = data_dict['/bubble_2/raw_flow'][idx].reshape(240, 320, 2)

                bubble_1_flow = cv2.resize(bubble_1_flow, (640, 480))
                bubble_2_flow = cv2.resize(bubble_2_flow, (640, 480))
                #
                # vis_flow = visualize_raw_flow(bubble_1_flow).astype(np.uint8)
                #
                # plt.imsave('images/vis_flow.jpg', vis_flow / 255.0)
                #
                # bubble_1_flow = visualize_raw_flow(bubble_1_flow).astype(np.uint8)
                # bubble_2_flow = visualize_raw_flow(bubble_2_flow).astype(np.uint8)
                #
                # # import pdb
                # pc = construct_pointcloud_from_rgbd(vis_flow, bubble_1_depth, intrinsics_depth['bubble_1'])
                # pc = remove_distant_points_pc(pc, 0.15)
                # open3d.visualization.draw_geometries([pc])
                # #
                # # pdb.set_trace()

                b1_rgb_flow = np.concatenate([bubble_1_rgb, bubble_1_flow], axis=2)
                b2_rgb_flow = np.concatenate([bubble_2_rgb, bubble_2_flow], axis=2)

                b1_feat_points = rgbd_feat_to_point_cloud(b1_rgb_flow, bubble_1_depth, intrinsic_matrices['bubble_1'])
                b2_feat_points = rgbd_feat_to_point_cloud(b2_rgb_flow, bubble_2_depth, intrinsic_matrices['bubble_2'])

                bubble_pcs = [b1_feat_points, b2_feat_points]
                bubble_pcs = [remove_distant_points(pc, 0.15) for pc in bubble_pcs]
                bubble_pcs = [project_points(convert_pc_optical_color_to_link_frame(pc), t_bubble_to_robot)
                              for pc, t_bubble_to_robot in zip(bubble_pcs, t_bubbles_to_robot)]
                bubble_pcs = [random_select_rows(pc, 10000) for pc in bubble_pcs]
                bubble_pcs = [denoise_by_cluster(pc, 0.01, 10, 1) for pc in bubble_pcs]
                bubble_sampled_pcs = [farthest_point_sampling_dgl(pc, n_points=N_POINTS_PER_OBJ)
                                      for pc in bubble_pcs]

                # open3d.visualization.draw(bubble_pcs)
                # import pdb
                # pdb.set_trace()

                # TODO: convert bubble pc to link frame, and then the workspace?

                bubble_sampled_pcs_o3d = [xyz_colors_to_pc(np.concatenate([rec_pc[:, :3],
                                                                           np.zeros((rec_pc.shape[0], 3))], axis=-1))
                                          for rec_pc in bubble_sampled_pcs]

                # bubble_sampled_pcs_o3d = [xyz_colors_to_pc(np.concatenate([pc[:, :3], pc[:, 6:]], axis=-1))
                #                           for pc in bubble_pcs]

                # # for debugging
                # from utils_cv import visualize_raw_flow
                # vis1 = visualize_raw_flow(bubble_1_flow) / 255
                # plt.imsave('vis1.jpg', vis1)
                # rec_pc = bubble_sampled_pcs[0]
                # rec_pc_pc = xyz_colors_to_pc(np.concatenate([rec_pc[:, :3], np.zeros((rec_pc.shape[0], 3))], axis=-1))
                # o3d.visualization.draw_geometries([rec_pc_pc])

                # bubble_pcs = [bubble1_pc, bubble2_pc]
                # bubble_pcs = [remove_distant_points_pc(pc, 0.15) for pc in bubble_pcs]  # remove points more than 20cm away
                #
                # bubble_filled_pcs = [pc + fill_point_cloud_by_downward_projection(pc) for pc in bubble_pcs]
                # bubble_filled_projected_pcs = [project_pc(convert_pc_optical_to_link_frame(pc), t_bubble_to_robot)
                #                                for pc, t_bubble_to_robot in zip(bubble_filled_pcs, t_bubbles_to_robot)]
                # bubble_filled_projected_pcs = [denoise_pc_by_stats(pc, denoise_depth=2)
                #                                for pc in bubble_filled_projected_pcs]
                # bubble_sampled_pcs = [farthest_point_sampling_dgl_pc(pc, n_points=N_POINTS_PER_OBJ)
                #                       for pc in bubble_filled_projected_pcs]

                # if args.debug:
                #     o3d.visualization.draw_geometries(bubble_filled_projected_pcs)

                # visual sensor reading
                cam_pcs = [data_dict[f'/cam_{i}/depth/color/points'][idx] for i in range(4)]

                pcs = []
                for i, pc in enumerate(cam_pcs):
                    pc = project_pc(convert_pc_optical_color_to_link_frame(pc, i), extrinsics[f'cam_{i}'])
                    pc = extract_workspace_pc(pc, xrange, yrange, zrange)
                    pcs.append(pc)

                box_points = filter_point_cloud_by_height(pcs[0] + pcs[1] + pcs[2] + pcs[3], 0.015, 0.05)
                _, box_points = segment_blue_rod_points(box_points)
                rod_points, _ = segment_blue_rod_points(pcs[0] + pcs[1] + pcs[2] + pcs[3])

                # o3d.visualization.draw_geometries([box_points])

                box_points = denoise_by_cluster(box_points, 0.01, 50, 1)
                box_points = fill_point_cloud_by_downward_projection(box_points, bottom_z=0.005)
                box_points = denoise_pc_by_stats(box_points, denoise_depth=2)
                box_points = farthest_point_sampling_dgl_pc(box_points, N_POINTS_PER_OBJ)

                # add the in-hand object representation
                t_ee_to_robot_now = pos_quat_to_matrix(ee_pos, ee_orient)
                in_hand_pc_now = project_pc(centered_in_hand_pc, t_ee_to_robot_now @ t_obj_to_init_ee)

                if args.debug:
                    o3d.visualization.draw_geometries([box_points, in_hand_pc_now] + bubble_sampled_pcs_o3d + pcs)
                    # o3d.visualization.draw_geometries(cam0_pcs + cam1_pcs)
                    pdb.set_trace()

                try:
                    # if len(denoised_sampled_pcs) == 0:
                    #     base_name = f'{seq_folder}_{rosbag_folder}'
                    #     plt.imsave(f'debugging/{base_name}_cam0_mask.png', np.clip(get_mask_image(cam0_masks), 0, 1))
                    #     plt.imsave(f'debugging/{base_name}_cam1_mask.png', np.clip(get_mask_image(cam1_masks), 0, 1))
                    #     plt.imsave(f'debugging/{base_name}_cam0_rgb.png', cam0_rgb)
                    #     plt.imsave(f'debugging/{base_name}_cam1_rgb.png', cam1_rgb)
                    #     # pdb.set_trace()
                    #     continue

                    # store data
                    denoised_sampled_pcs = [box_points]
                    object_pcs_array = np.stack([get_pc_xyz_color_array(pc)
                                                 for pc in denoised_sampled_pcs], axis=0)
                    if len(save_data_dict['object_pcs']) > 0:
                        if not (object_pcs_array.shape == save_data_dict['object_pcs'][-1].shape):
                            print(f'skipped object_pcs_array.shape == {object_pcs_array.shape}')
                            continue   # skip the data point

                    save_data_dict['object_pcs'].append(np.stack([get_pc_xyz_color_array(pc)
                                                                  for pc in denoised_sampled_pcs], axis=0))
                    save_data_dict['bubble_pcs'].append(np.stack(bubble_sampled_pcs, axis=0))
                    save_data_dict['inhand_object_pcs'].append(get_pc_xyz_color_array(in_hand_pc_now))
                    save_data_dict['forces'].append(
                        np.stack([data_dict['/bubble_1/force'][idx], data_dict['/bubble_2/force'][idx]], axis=0)
                    )
                    # save_data_dict['flows'].append(
                    #     np.stack([data_dict['/bubble_1/raw_flow'][idx], data_dict['/bubble_2/raw_flow'][idx]], axis=0)
                    # )
                    save_data_dict['pressure'].append(
                        np.stack(
                            [
                                data_dict['/bubble_AF6E04F550555733362E3120FF0F3217/pressure'][idx],
                                data_dict['/bubble_C0A594A050583234322E3120FF0D2108/pressure'][idx]
                            ], axis=0
                        )
                    )
                    save_data_dict['ee_states'].append(data_dict['/ee_states'][idx])
                    save_data_dict['joint_states'].append(data_dict['/joint_states'][idx])

                except Exception as e:
                    print(e)
                    pdb.set_trace()

                saved_n_frame += 1

            # save data
            file = h5py.File(h5_path, 'w')
            for key, value in save_data_dict.items():
                try:
                    np.stack(value, axis=0)
                except Exception as e:
                    print(e)
                    pdb.set_trace()
                file.create_dataset(key, data=np.stack(value, axis=0))
            file.close()
            print(f'Data saved successfully at {h5_path}, saved {saved_n_frame}/{min_len} frames. ')
            del data_dict


# install torch 2.0 for SAM to work: pip3 install torch torchvision torchaudio
# install torch 113+cuda116 for dynamics model to work:
# pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, default='/media/albert/ExternalHDD/bubble_data/v9_0808',
                        help="Path to directory containing the seq_x folders")
    parser.add_argument("--every_n_frames", type=int, default=1, help="Sample a frame every x frames")
    parser.add_argument("--debug", type=int, default=0, help="Debug mode or not")
    parser.add_argument("--target_path", type=str,
                        default='/media/albert/ExternalHDD/bubble_data/v9_0808_parsed_2',
                        help="Directory to store parsed data")
    parser.add_argument("--num_objects", type=int, default=1, help="number of objects on the table")
    parser.add_argument("--max_nframe", type=int, default=130, help="max number of frames in one h5 file")  # 100 frames take (19-x)G RAM
    args = parser.parse_args()

    main(args)

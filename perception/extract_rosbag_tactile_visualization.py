#!/usr/bin/env python3

import sys

import numpy as np
import open3d.cuda.pybind.visualization.rendering
import open3d.visualization
import rosbags

sys.path.append('/home/albert/github/robopack')
sys.path.append('/home/albert/github/robopack/ros2_numpy')

from utils.visualizer import play_and_save_video
from config.task_config import N_POINTS_PER_OBJ, extrinsics, intrinsics_depth, intrinsic_matrices

import tqdm

from rclpy.serialization import deserialize_message
from rosbags.rosbag2 import Reader

from perception.utils_ros import *
from perception.utils_cv import *
from utils_general import *

# the below needs to be imported at last, not sure why
from sensor_msgs.msg import PointCloud2, Image, JointState
from std_msgs.msg import Float32MultiArray


"""
The scripts extracts the tactile reading from rosbag file and save visualizations of the tactile
    readings to the local disk.
    
    data from ros bag files. Each trajectory is one h5 file located at
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
    '/bubble_1/color/image_rect_raw': 'sensor_msgs/msg/Image',
    '/bubble_2/color/image_rect_raw': 'sensor_msgs/msg/Image',
    '/bubble_1/flow': 'sensor_msgs/msg/Image',
    '/bubble_2/flow': 'sensor_msgs/msg/Image',
}

topic_to_class = {
    'sensor_msgs/msg/PointCloud2': PointCloud2,
    'sensor_msgs/msg/Image': Image,
    'std_msgs/msg/Float32MultiArray': Float32MultiArray,
    'sensor_msgs/msg/JointState': JointState
}


def read_rosbag(path, every_n_frames, max_frame_count):
    # dics to save the messages
    data_dict = {}
    count_dict = {}

    # create reader instance and open for reading
    with Reader(path) as reader:
        # create a progress bar
        # progress_bar = tqdm.tqdm(total=sum(1 for _ in reader.messages()), unit="message")

        # iterate over messages
        for connection, timestamp, rawdata in tqdm.tqdm(reader.messages()):
            # filter out unused messages
            print(connection.topic, connection.msgtype)
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


def main(args):
    """
    For each folder named seq_x in the source path, read the rosbag files,
    reconstruct the scene, sample the point clouds, and save the training data
    into a h5 file.
    :param args: user-input args
    :return: None
    """

    data_dict = read_rosbag(args.source_path, args.every_n_frames, args.max_nframe)

    # check missing topics
    missing_topics = []
    for topic in topics_to_collect:
        if not topic in data_dict:
            missing_topics.append(topic)
    assert len(missing_topics) == 0, f"missing topics: {missing_topics}"

    data_len = len(data_dict['/bubble_1/flow'])

    # process the frames
    # actually this step is redundant if the recorded signals are already the visualized images
    visualization_dic = {
        'rgb_arrows_overlaid_b1': [],
        'rgb_arrows_overlaid_b2': [],
        'rgb_b1': [],
        'rgb_b2': []
    }
    for idx in range(data_len):
        for bubble_idx in range(1, 3):
            # bubble_vis_path = f'{args.target_path}/bubble_{bubble_idx}/step{idx}.jpg'
            # os.makedirs(os.path.dirname(bubble_vis_path), exist_ok=True)

            rgb_arrows_overlaid = data_dict[f'/bubble_{bubble_idx}/flow'][idx].reshape(240, 320, 3)
            visualization_dic[f'rgb_arrows_overlaid_b{bubble_idx}'].append(rgb_arrows_overlaid)

            rgb = data_dict[f'/bubble_{bubble_idx}/color/image_rect_raw'][idx].reshape(240, 320, 3)
            visualization_dic[f'rgb_arrows_overlaid_b{bubble_idx}'].append(rgb)

            # # generate visualizations from raw vectors
            # raw_vectors = data_dict[f'/bubble_{bubble_idx}/color/image_rect_raw'][idx]
            #
            # b1_flow = data_dict['/bubble_1/flow'][idx].reshape(240, 320, 3)
            # b2_flow = data_dict['/bubble_2/flow'][idx].reshape(240, 320, 3)
            # plt.imsave('tac_vis_images/b1_vis_flow.jpg', b1_flow / 255.0)
            # plt.imsave('tac_vis_images/b2_vis_flow.jpg', b2_flow / 255.0)
            #
            # b1_flow = cv2.resize(b1_flow, (640, 480))
            # b2_flow = cv2.resize(b2_flow, (640, 480))
            #
            # b1_flow_vis = visualize_raw_flow(b1_flow).astype(np.uint8)
            # b2_flow_vis = visualize_raw_flow(b2_flow).astype(np.uint8)
            #
            # plt.imsave('tac_vis_images/b1_vis_flow.jpg', b1_flow_vis / 255.0)
            # plt.imsave('tac_vis_images/b2_vis_flow.jpg', b1_flow_vis / 255.0)
            #
            # bubble_1_flow = visualize_raw_flow(bubble_1_flow).astype(np.uint8)
            # bubble_2_flow = visualize_raw_flow(bubble_2_flow).astype(np.uint8)

    # save the frames as videos
    vis_path = os.path.join(args.target_path, os.path.basename(args.source_path))
    os.makedirs(vis_path, exist_ok=True)
    for key, img_list in visualization_dic.items():
        play_and_save_video(img_list, os.path.join(vis_path, f'{key}.mp4'), fps=15, pause_time_last_frame=0)

        # b1_flow = data_dict['/bubble_1/flow'][idx].reshape(240, 320, 3)
        # b2_flow = data_dict['/bubble_2/flow'][idx].reshape(240, 320, 3)
        # plt.imsave('tac_vis_images/b1_vis_flow.jpg', b1_flow / 255.0)
        # plt.imsave('tac_vis_images/b2_vis_flow.jpg', b2_flow / 255.0)
        # breakpoint()
        #
        # b1_flow = cv2.resize(b1_flow, (640, 480))
        # b2_flow = cv2.resize(b2_flow, (640, 480))
        #
        # b1_flow_vis = visualize_raw_flow(b1_flow).astype(np.uint8)
        # b2_flow_vis = visualize_raw_flow(b2_flow).astype(np.uint8)
        #
        # plt.imsave('tac_vis_images/b1_vis_flow.jpg', b1_flow_vis / 255.0)
        # plt.imsave('tac_vis_images/b2_vis_flow.jpg', b1_flow_vis / 255.0)

        # bubble_1_flow = visualize_raw_flow(bubble_1_flow).astype(np.uint8)
        # bubble_2_flow = visualize_raw_flow(bubble_2_flow).astype(np.uint8)
        #
        # # import pdb
        # pc = construct_pointcloud_from_rgbd(vis_flow, bubble_1_depth, intrinsics_depth['bubble_1'])
        # pc = remove_distant_points_pc(pc, 0.15)
        # open3d.visualization.draw_geometries([pc])
        # #
        # # pdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str,
                        default='/home/albert/github/punyo/rosbag2_2024_02_05-19_44_23',
                        help="Path to the rosbag folder")
    parser.add_argument("--every_n_frames", type=int, default=1, help="Sample a frame every x frames")
    parser.add_argument("--target_path", type=str,
                        default='tac_vis_images',
                        help="Directory to store parsed data")
    parser.add_argument("--max_nframe", type=int, default=1e5, help="max number of frames in one h5 file")
    args = parser.parse_args()

    main(args)

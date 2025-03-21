import sys
sys.path.append('/home/albert/github/robopack')

import open3d as o3d
import numpy as np
import argparse
import os
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image, JointState
from message_filters import ApproximateTimeSynchronizer, Subscriber
from std_msgs.msg import Float32MultiArray
import matplotlib.pyplot as plt

from perception.utils_ros import decode_img_msg
from utils_general import read_extrinsics

from datetime import datetime

import cv2


# Get the current date and time
current_datetime = datetime.now()

# Convert the datetime object to a safe folder name format
date_time_folder = current_datetime.strftime('%Y-%m-%d_%H-%M-%S')


root_dir = f'/media/albert/ExternalHDD/track_data/sep6/{date_time_folder}'
folder_name = ['camera_0', 'camera_1', 'camera_2', 'camera_3']
global_count = 0

time_last = time.time()
def callback(*args):
    global global_count
    msgs = [decode_img_msg(x) for x in args]
    for i, msg in enumerate(msgs):
        rgb_depth = 'rgb' if i < 4 else 'depth'
        folder_to_save = os.path.join(root_dir, folder_name[i % 4])
        os.makedirs(folder_to_save, exist_ok=True)
        file_path = os.path.join(folder_to_save, f'{rgb_depth}_{global_count}.png')
        if i < 4:
            image_bgr = cv2.cvtColor(msg, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_path, image_bgr)
        else:
            cv2.imwrite(file_path, msg)

    global_count += 1
    global time_last
    print(f'Callback fps: {1/(time.time() - time_last)}')
    time_last = time.time()


def main(args):
    rclpy.init(args=None)
    node = rclpy.create_node('realsense_to_open3d_node2')

    rgb1 = Subscriber(node, Image, '/cam_0/color/image_raw')
    rgb2 = Subscriber(node, Image, '/cam_1/color/image_raw')
    rgb3 = Subscriber(node, Image, '/cam_2/color/image_raw')
    rgb4 = Subscriber(node, Image, '/cam_3/color/image_raw')
    #
    depth1 = Subscriber(node, Image, '/cam_0/depth/image_rect_raw')
    depth2 = Subscriber(node, Image, '/cam_1/depth/image_rect_raw')
    depth3 = Subscriber(node, Image, '/cam_2/depth/image_rect_raw')
    depth4 = Subscriber(node, Image, '/cam_3/depth/image_rect_raw')

    synchronizer = ApproximateTimeSynchronizer(
        [rgb1, rgb2, rgb3, rgb4, depth1, depth2, depth3, depth4],
        queue_size=200, slop=0.2
    )
    synchronizer.registerCallback(callback)

    # rgb3.registerCallback(callback)
    # rclpy.spin(node)

    # rclpy.init(args=None)
    # node = rclpy.create_node('realsense_to_open3d_node2')
    #
    # sub1 = Subscriber(node, PointCloud2, '/cam_0/depth/color/points')
    # sub2 = Subscriber(node, PointCloud2, '/cam_1/depth/color/points')
    # sub3 = Subscriber(node, Image, '/cam_0/color/image_raw')
    # sub4 = Subscriber(node, Image, '/cam_1/color/image_raw')
    # sub5 = Subscriber(node, Image, '/cam_0/depth/image_rect_raw')
    # sub6 = Subscriber(node, Image, '/cam_1/depth/image_rect_raw')
    #
    # synchronizer = ApproximateTimeSynchronizer(
    #     [sub1, sub2, sub3, sub4, sub5, sub6], queue_size=100, slop=0.1
    # )
    # synchronizer.registerCallback(callback)

    try:
        print('start spinning')
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
        print('successfully shut down')

    # read and save extrinsic
    extrinsics = read_extrinsics('/home/albert/github/robopack/config/sensor/4cameras_pose_robot_v7.yml')
    from config.task_config import intrinsic_matrices
    for key, value in extrinsics.items():
        cam_index = int(key[-1])
        intrinsics = intrinsic_matrices[key]
        cam_folder = folder_name[cam_index]
        os.makedirs(cam_folder, exist_ok=True)
        np.save(os.path.join(root_dir, cam_folder, 'camera_extrinsics.npy'), value)
        np.save(os.path.join(root_dir, cam_folder, 'camera_intrinsics.npy'),
                [intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]])
    print(f'data saved to {root_dir}')


datafolder = None
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to process data from a specified folder.')
    parser.add_argument('--data_folder', type=str, help='Path to the data folder')
    args = parser.parse_args()
    datafolder = args.data_folder
    main(args)

import sys
import numpy as np
sys.path.append('/home/albert/github/robopack')

import rclpy
from rclpy.time import Time

import os
import time

from utils_general import AverageMeter
from perception.utils_ros import decode_img_msg, check_topic_status, ApproximateTimeSynchronizer
from datetime import datetime
import cv2
from message_filters import Subscriber
from std_msgs.msg import Float32MultiArray, Float32
from custom_msg.msg import StampedFloat32MultiArray

from utils_general import find_max_seq_folder, read_extrinsics
from utils.autosave_list import AutosaveList
from sensor_msgs.msg import PointCloud2, Image, JointState

# date_time_folder = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
seq_folder_path = find_max_seq_folder('/home/albert/Dataset/1_31_packing_fig2/')
data_folder = seq_folder_path  # os.path.join(seq_folder_path, date_time_folder)
print('saving to ', data_folder)

# WARNING: ROS2 only supports synchronizing message of the same type.
sync_topics = {
    '/cam_0/color/image_raw': 'sensor_msgs/msg/Image',
    '/cam_0/depth/image_rect_raw': 'sensor_msgs/msg/Image',
    '/cam_1/color/image_raw': 'sensor_msgs/msg/Image',
    '/cam_1/depth/image_rect_raw': 'sensor_msgs/msg/Image',
    '/cam_2/color/image_raw': 'sensor_msgs/msg/Image',
    '/cam_2/depth/image_rect_raw': 'sensor_msgs/msg/Image',
    '/cam_3/color/image_raw': 'sensor_msgs/msg/Image',
    '/cam_3/depth/image_rect_raw': 'sensor_msgs/msg/Image',

    # Don't really know why... but the following topics cannot be synchronized
    '/bubble_1/depth/image_rect_raw': 'sensor_msgs/msg/Image',
    '/bubble_1/color/image_rect_raw': 'sensor_msgs/msg/Image',
    '/bubble_2/depth/image_rect_raw': 'sensor_msgs/msg/Image',
    '/bubble_2/color/image_rect_raw': 'sensor_msgs/msg/Image',
    '/ee_states': 'custom_msg/msg/StampedFloat32MultiArray',

    '/bubble_1/raw_flow': 'custom_msg/msg/StampedFloat32MultiArray',
    '/bubble_2/raw_flow': 'custom_msg/msg/StampedFloat32MultiArray',
    '/bubble_1/force': 'custom_msg/msg/StampedFloat32MultiArray',
    # # '/bubble_C0A594A050583234322E3120FF0D2108/pressure': 'std_msgs/msg/Float32',
    '/bubble_2/force': 'custom_msg/msg/StampedFloat32MultiArray',
    # '/bubble_AF6E04F550555733362E3120FF0F3217/pressure': 'std_msgs/msg/Float32',
    '/joint_states': 'sensor_msgs/msg/JointState'
}

unsync_topics = { }

topics_save_to_h5 = {k: [] for k, v in unsync_topics.items()}

autosave_lists = {}
for topic, type_name in sync_topics.items():
    if topic == '/joint_states' or topic == '/ee_states':
        file_prefix = topic[1:]
    else:
        file_prefix = topic.split('/')[2]
    msg_folder_name = topic.split('/')[1]
    autosave_lists[topic] = AutosaveList(os.path.join(data_folder, msg_folder_name), file_prefix, 100)

last_call_back_time = time.time()
has_waited = False

frequency_counter = AverageMeter()
time_diff = AverageMeter()
global_count = 0
time_last = time.time()


def callback(*args):
    """
    Saves the data to disk
    :return: None
    """
    global time_last, global_count

    print('synchronized callback invoked')
    current_time = time.time()

    global last_call_back_time, has_waited

    # if current_time - last_call_back_time > 3:
    #     has_waited = True

    # # wait for a few frames to pass before recording data to ensure synchronization
    # if current_time - last_call_back_time < 3 and not has_waited:
    #     print(f'waiting... ')
    #     return

    # msgs = list(args)

    # debugging
    min_time, max_time = float('inf'), float('-inf')
    for msg in args:
        msg_time = Time.from_msg(msg.header.stamp).nanoseconds
        #msg_time = msg.header.stamp.nanosec
        if msg_time > max_time:
            max_time = msg_time
        if msg_time < min_time:
            min_time = msg_time
    time_diff.update((max_time - min_time)/1e9)
    print(f'time diff across topics: last = {time_diff.val:.3f}, avg = {time_diff.avg}')

    # check data collection frequency
    # if has_waited:
    frequency_counter.update((time.time() - last_call_back_time))
    last_call_back_time = time.time()
    print(f'average frequency = {1 / frequency_counter.avg:.3f} hz')

    for msg, (topic, type_name) in zip(args, sync_topics.items()):
        if 'Image' in type_name:
            msg = decode_img_msg(msg)
        elif 'StampedFloat32MultiArray' in type_name:
            msg = msg.data.data
        elif 'JointState' in type_name:
            msg = msg.position
        autosave_lists[topic].append(msg)

    # process unsynchronized messages
    assert len(unsync_topics) == 0

    global_count += 1
    if global_count % 100 == 0:
        print("------------------------------")
        print(f'GLOBAL COUNT = {global_count}')
        print("------------------------------")

    print(f'callback ends here')


def record():
    rclpy.init(args=None)
    node = rclpy.create_node('record_data')

    # qos profile needed for
    qos_profile = rclpy.qos.QoSProfile(
        history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
        depth=10,  # Change the depth value as needed, it determines how many past messages to keep
        reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT
    )

    sync_subscriptions = []
    for k, v in sync_topics.items():
        print(k, v.split("/")[-1])
        sync_subscriptions.append(Subscriber(node, globals()[v.split("/")[-1]], k))
        check_topic_status(node, k)

    # subs = [Subscriber(k, globals()[v.split("/")[-1]]) for k, v in topics_to_type.items()]
    synchronizer = ApproximateTimeSynchronizer(
        sync_subscriptions, queue_size=200, slop=0.14, allow_headerless=False
    )
    synchronizer.registerCallback(callback)

    unsync_subscriptions = []
    for k, v in unsync_topics.items():
        # subscriber = Subscriber(node, globals()[v.split("/")[-1]], k)
        # print(v.split("/")[-1], f'callback{k.replace("/", "_")}')
        subscription = node.create_subscription(
            globals()[v.split("/")[-1]],  # type
            k,  # topic
            globals()[f'callback{k.replace("/", "_")}'],  # callback function name
            qos_profile
        )
        unsync_subscriptions.append(subscription)
        check_topic_status(node, k)

    # writer must be closed, otherwise bag not readable
    try:
        print('start spinning node. Please make sure callback is being invoked')
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        print('starting shutting down, please do NOT exit!!!')
        rclpy.shutdown()

        # clear autosave lists
        for topic, autosave_list in autosave_lists.items():
            autosave_list.close()

        from utils_general import save_dictionary_to_hdf5
        np.save(os.path.join(data_folder, 'other_topics.npy'), topics_save_to_h5)

        # # save lists
        # for topic, data in autosave_lists.items():
        #     if len(data) > 0:
        #         np.save(os.path.join(data_folder, topic.split("/")[1]), data)

        # read and save extrinsic
        extrinsics = read_extrinsics('/home/albert/github/robopack/config/sensor/4cameras_pose_robot_v9.yml')
        from config.task_config import intrinsic_matrices
        for cam_name, intrinsics in intrinsic_matrices.items():
            cam_folder = os.path.join(data_folder, cam_name)
            os.makedirs(cam_folder, exist_ok=True)
            np.save(os.path.join(cam_folder, 'camera_intrinsics.npy'),
                    [intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]])

            if cam_name in extrinsics:
                np.save(os.path.join(cam_folder, 'camera_extrinsics.npy'), extrinsics[cam_name])
            else:
                assert 'bubble' in cam_name, 'there should not be cameras other than bubble without extrinsics'

        print(f'All data saved to {data_folder}')


if __name__ == '__main__':
    record()

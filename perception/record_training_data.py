import sys
import warnings
import numpy as np
sys.path.append('/home/albert/github/robopack')

import rclpy
import logging
import os
import time

from utils_general import AverageMeter
from perception.utils_ros import decode_img_msg
from datetime import datetime
import cv2

from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import PointCloud2, Image, JointState
from std_msgs.msg import Float32MultiArray, Float32
from custom_ee_msg.msg import StampedFloat32MultiArray
# from rosbags.rosbag2 import Writer
from rclpy.serialization import serialize_message

from utils_general import find_max_seq_folder, create_sequential_folder, Queue, read_extrinsics
from utils.autosave_list import AutosaveList


# date_time_folder = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
seq_folder_path = find_max_seq_folder('/media/albert/ExternalHDD/bubble_data/test')
data_folder = seq_folder_path  # os.path.join(seq_folder_path, date_time_folder)
print('saving to ', data_folder)

# WARNING: ROS2 only supports synchronizing message of the same type.
sync_topics = {
    # '/cam_0/color/image_raw': 'sensor_msgs/msg/Image',
    # '/cam_0/depth/image_rect_raw': 'sensor_msgs/msg/Image',
    '/cam_1/color/image_raw': 'sensor_msgs/msg/Image',
    '/cam_1/depth/image_rect_raw': 'sensor_msgs/msg/Image',
    '/cam_2/color/image_raw': 'sensor_msgs/msg/Image',
    '/cam_2/depth/image_rect_raw': 'sensor_msgs/msg/Image',
    '/cam_3/color/image_raw': 'sensor_msgs/msg/Image',
    '/cam_3/depth/image_rect_raw': 'sensor_msgs/msg/Image',

    # Don't really know why... but the following topics cannot be synchronized
    # '/bubble_1/color/image_rect_raw': 'sensor_msgs/msg/Image',
    '/bubble_1/depth/image_rect_raw': 'sensor_msgs/msg/Image',
    #
    # '/bubble_2/color/image_rect_raw': 'sensor_msgs/msg/Image',
    '/bubble_2/depth/image_rect_raw': 'sensor_msgs/msg/Image',
}

unsync_topics = {
    '/bubble_1/raw_flow': 'std_msgs/msg/Float32MultiArray',
    '/bubble_2/raw_flow': 'std_msgs/msg/Float32MultiArray',
    '/bubble_1/force': 'std_msgs/msg/Float32MultiArray',
    # '/bubble_C0A594A050583234322E3120FF0D2108/pressure': 'std_msgs/msg/Float32',
    '/bubble_2/force': 'std_msgs/msg/Float32MultiArray',
    # '/bubble_AF6E04F550555733362E3120FF0F3217/pressure': 'std_msgs/msg/Float32',
    '/ee_states': 'std_msgs/msg/Float32MultiArray',
    '/joint_states': 'sensor_msgs/msg/JointState'
}


topics_save_to_h5 = {k: [] for k, v in unsync_topics.items()}

autosave_lists = {}
for topic, type_name in sync_topics.items():
    file_prefix = 'rgb' if 'color' in type_name else 'depth'
    msg_folder_name = topic.split('/')[1]
    autosave_lists[topic] = AutosaveList(os.path.join(data_folder, msg_folder_name), file_prefix, 100)


# To be honest, nobody should bind the variable and time stamp in such a stupid way
# but there is no simpler option given the missing functionalities of ROS 2
max_size = 30
bubble_1_depth_image_rect_raw = Queue(max_size, [(None, time.time())])
bubble_2_depth_image_rect_raw = Queue(max_size, [(None, time.time())])
bubble_1_flow = Queue(max_size, [(None, time.time())])
bubble_1_raw_flow = Queue(max_size, [(None, time.time())])
bubble_1_force = Queue(max_size, [(None, time.time())])
bubble_C0A594A050583234322E3120FF0D2108_pressure = Queue(max_size, [(None, time.time())])
bubble_2_flow = Queue(max_size, [(None, time.time())])
bubble_2_raw_flow = Queue(max_size, [(None, time.time())])
bubble_2_force = Queue(max_size, [(None, time.time())])
bubble_AF6E04F550555733362E3120FF0F3217_pressure = Queue(max_size, [(None, time.time())])
ee_states = Queue(max_size, [(None, time.time())])
joint_states = Queue(max_size, [(None, time.time())])


# Hand-implemented callback functions for each topic
# Not really ideal, but it works

def callback_bubble_1_depth_image_rect_raw(msg):
    global bubble_1_depth_image_rect_raw
    bubble_1_depth_image_rect_raw.add((msg, time.time()))


def callback_bubble_2_depth_image_rect_raw(msg):
    global bubble_2_depth_image_rect_raw
    bubble_2_depth_image_rect_raw.add((msg, time.time()))


def callback_bubble_1_flow(msg):
    global bubble_1_flow
    bubble_1_flow.add((msg, time.time()))


def callback_bubble_1_raw_flow(msg):
    global bubble_1_raw_flow
    bubble_1_raw_flow.add((msg, time.time()))


def callback_bubble_1_force(msg):
    global bubble_1_force
    bubble_1_force.add((msg, time.time()))


def callback_bubble_C0A594A050583234322E3120FF0D2108_pressure(msg):
    global bubble_C0A594A050583234322E3120FF0D2108_pressure
    bubble_C0A594A050583234322E3120FF0D2108_pressure.add((msg, time.time()))


def callback_bubble_2_flow(msg):
    global bubble_2_flow
    bubble_2_flow.add((msg, time.time()))


def callback_bubble_2_raw_flow(msg):
    global bubble_2_raw_flow
    bubble_2_raw_flow.add((msg, time.time()))


def callback_bubble_2_force(msg):
    global bubble_2_force
    bubble_2_force.add((msg, time.time()))


def callback_bubble_AF6E04F550555733362E3120FF0F3217_pressure(msg):
    global bubble_AF6E04F550555733362E3120FF0F3217_pressure
    bubble_AF6E04F550555733362E3120FF0F3217_pressure.add((msg, time.time()))


def callback_ee_states(msg):
    global ee_states
    ee_states.add((msg, time.time()))


def callback_joint_states(msg):
    global joint_states
    joint_states.add((msg, time.time()))


def check_topic_status(node, topic):
    publishers_info = node.get_publishers_info_by_topic(topic)
    if publishers_info:
        logging.debug(f"Topic {topic} is active. Publishers: {publishers_info}")
    else:
        logging.warning(f"No active publishers found for topic {topic}")


last_call_back_time = time.time()
has_waited = False


def find_most_recent_msg_in_queue(queue, ref_time):
    most_recent_msg, most_recent_t = None, float('-inf')
    for msg, t in queue.queue:
        if abs(ref_time - t) < (ref_time - most_recent_t):
            most_recent_msg = msg
            most_recent_t = t
    # delay = abs(most_recent_t - ref_time)
    delay = abs(most_recent_t - ref_time)
    return most_recent_msg, delay


frequency_counter = AverageMeter()
delay_counter = AverageMeter()
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
    if current_time - last_call_back_time > 3:
        has_waited = True

    # wait for a few frames to pass before recording data to ensure synchronization
    if current_time - last_call_back_time < 3 and not has_waited:
        print(f'waiting... ')
        return

    # msgs = list(args)

    # debugging
    # min_time, max_time = float('inf'), float('-inf')
    # for msg in msgs:
    #     try:
    #         msg_time = msg.header.stamp.sec
    #         if msg_time > max_time:
    #             max_time = msg_time
    #         if msg_time < min_time:
    #             min_time = msg_time
    #     except NameError:
    #         pass
    # print(f'time diff across topics = {max_time - min_time:.3f}')

    # check data collection frequency
    if has_waited:
        frequency_counter.update(1/(time.time() - last_call_back_time))
        print(f'average frequency = {frequency_counter.avg:.3f} hz')

    last_call_back_time = time.time()

    # process synchronized messages
    msgs = [decode_img_msg(x) for x in args]
    # msg_folder_names = [x.split('/')[1] for x in sync_topics.keys()]
    type_names = [x.split('/')[2] for x in sync_topics.keys()]

    # for msg, msg_folder_name, type_name in zip(msgs, msg_folder_names, type_names):
    #     file_name = f"{'rgb' if 'color' in type_name else 'depth'}_{global_count}.png"
    #     folder_to_save = os.path.join(data_folder, msg_folder_name)
    #     os.makedirs(folder_to_save, exist_ok=True)
    #     file_path = os.path.join(folder_to_save, file_name)
    #     if 'color' in type_name:
    #         image_bgr = cv2.cvtColor(msg, cv2.COLOR_RGB2BGR)
    #         cv2.imwrite(file_path, image_bgr)
    #     else:
    #         cv2.imwrite(file_path, msg)

    for msg, topic, type_name in zip(msgs, sync_topics.keys(), type_names):
        # if 'color' in type_name:
        #     msg = cv2.cvtColor(msg, cv2.COLOR_RGB2BGR)
        autosave_lists[topic].append(msg)

    # process unsynchronized messages
    for topic, type_name in unsync_topics.items():
        queue = globals()[topic.replace("/", "_")[1:]]
        msg, delay = find_most_recent_msg_in_queue(queue, current_time)

        delay_counter.update(delay)
        print(f'cumulative average delay: {delay_counter.avg:.3f}')

        if delay > 0.2:
            warnings.warn(f"delay for topic {topic} is {delay:.3f} seconds")

        if 'Float32MultiArray' in type_name:
            topics_save_to_h5[topic].append(msg.data)
        elif 'JointState' in type_name:
            topics_save_to_h5[topic].append(msg.position)
        elif 'Image' in type_name:
            data = decode_img_msg(msg)
            topics_save_to_h5[topic].append(data)

    global_count += 1


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
        sync_subscriptions.append(Subscriber(node, globals()[v.split("/")[-1]], k))
        check_topic_status(node, k)

    # subs = [Subscriber(k, globals()[v.split("/")[-1]]) for k, v in topics_to_type.items()]
    synchronizer = ApproximateTimeSynchronizer(
        sync_subscriptions, queue_size=200, slop=0.2, allow_headerless=False
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
    extrinsics = read_extrinsics('/home/albert/github/robopack/config/sensor/4cameras_pose_robot_v8.yml')
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

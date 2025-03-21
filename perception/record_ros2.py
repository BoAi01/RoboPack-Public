import sys
import warnings

import numpy as np

sys.path.append('/home/albert/github/robopack')

import rclpy
import logging
import os
import time

from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import PointCloud2, Image, JointState
from std_msgs.msg import Float32MultiArray, Float32
from rosbags.rosbag2 import Writer
from rclpy.serialization import serialize_message

from utils_general import find_max_seq_folder, create_sequential_folder, Queue

rosbag_path = find_max_seq_folder('/media/albert/ExternalHDD/bubble_data/v12_0906')
print(f'bag file will be at {rosbag_path}')
writer_path = os.path.join(rosbag_path, create_sequential_folder(rosbag_path, 'rosbag', create=False))
writer = Writer(writer_path)
print(f'write writes to {writer_path}')
writer.open()

# WARNING: ROS2 only supports synchronizing message of the same type.
sync_topics = {
    '/bubble_1/depth/color/points': 'sensor_msgs/msg/PointCloud2',
    '/bubble_2/depth/color/points': 'sensor_msgs/msg/PointCloud2',
    '/cam_0/depth/color/points': 'sensor_msgs/msg/PointCloud2',
    '/cam_1/depth/color/points': 'sensor_msgs/msg/PointCloud2',
    '/cam_2/depth/color/points': 'sensor_msgs/msg/PointCloud2',
    '/cam_3/depth/color/points': 'sensor_msgs/msg/PointCloud2',
    # '/cam_0/color/image_raw': 'sensor_msgs/msg/Image',
    # '/cam_0/depth/image_rect_raw': 'sensor_msgs/msg/Image',
    # '/cam_1/color/image_raw': 'sensor_msgs/msg/Image',
    # '/cam_1/depth/image_rect_raw': 'sensor_msgs/msg/Image',
    # '/cam_2/color/image_raw': 'sensor_msgs/msg/Image',
    # '/cam_2/depth/image_rect_raw': 'sensor_msgs/msg/Image',
    # '/cam_3/color/image_raw': 'sensor_msgs/msg/Image',
    # '/cam_3/depth/image_rect_raw': 'sensor_msgs/msg/Image',
}

unsync_topics = {
    # '/bubble_1/flow': 'sensor_msgs/msg/Image',
    # '/bubble_2/flow': 'sensor_msgs/msg/Image',
    '/bubble_1/raw_flow': 'std_msgs/msg/Float32MultiArray',
    '/bubble_2/raw_flow': 'std_msgs/msg/Float32MultiArray',
    '/bubble_1/force': 'std_msgs/msg/Float32MultiArray',
    '/bubble_C0A594A050583234322E3120FF0D2108/pressure': 'std_msgs/msg/Float32',
    '/bubble_2/force': 'std_msgs/msg/Float32MultiArray',
    '/bubble_AF6E04F550555733362E3120FF0F3217/pressure': 'std_msgs/msg/Float32',
    '/ee_states': 'std_msgs/msg/Float32MultiArray',
    '/joint_states': 'sensor_msgs/msg/JointState'
}

topics_save_to_h5 = {
    '/bubble_C0A594A050583234322E3120FF0D2108/pressure': [],
    '/bubble_AF6E04F550555733362E3120FF0F3217/pressure': [],
}

sync_connections = [writer.add_connection(k, v) for k, v in sync_topics.items()]
sync_msg_types = list(sync_topics.values())
unsync_connections = [writer.add_connection(k, v) for k, v in unsync_topics.items()]
unsync_msg_types = list(unsync_topics.values())

# To be honest, nobody should bind the variable and time stamp in such a stupid way
# but there is no simpler option given the missing functionalities of ROS 2
max_size = 20
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
    delay = abs(most_recent_t - ref_time)
    return most_recent_msg, delay


from utils_general import AverageMeter
frequency_counter = AverageMeter()
def callback(*args):
    """
    Saves the data to disk
    :return: None
    """
    print('synchronized callback invoked')
    current_time = time.time()

    global last_call_back_time, has_waited
    if current_time - last_call_back_time > 3:
        has_waited = True

    # wait for a few frames to pass before recording data to ensure synchronization
    if current_time - last_call_back_time < 3 and not has_waited:
        print(f'waiting... ')
        return

    msgs = list(args)

    # debugging
    min_time, max_time = float('inf'), float('-inf')
    for msg in msgs:
        try:
            msg_time = msg.header.stamp.sec
            if msg_time > max_time:
                max_time = msg_time
            if msg_time < min_time:
                min_time = msg_time
        except NameError:
            pass
    print(f'time diff across topics = {max_time - min_time:.3f}')

    # check data collection frequency
    if has_waited:
        frequency_counter.update(1/(time.time() - last_call_back_time))
        print(f'average frequency = {frequency_counter.avg:.3f} hz')

    print(f'data collecting at {1 / (time.time() - last_call_back_time)} hz')
    last_call_back_time = time.time()

    time_now = (min_time + max_time) / 2

    # save messages to disk
    for msg, connection, msg_type in zip(msgs, sync_connections, sync_msg_types):
        writer.write(connection, time_now, serialize_message(msg))

    for topic, connection, msg_type in zip(unsync_topics.keys(), unsync_connections, unsync_msg_types):
        queue = globals()[topic.replace("/", "_")[1:]]
        msg, delay = find_most_recent_msg_in_queue(queue, current_time)
        # print(topic, delay)
        if delay > 0.2:
            warnings.warn(f'delay of topic {topic} is {delay}')
        # assert msg is not None, f'{topic.replace("/", "_")[1:]} is None. Node active?'
        # assert time.time() - timestamp < 2, \
        #     f"variable of topic {topic} has not been updated for more than {time.time() - timestamp} secs!!"
        if topic not in topics_save_to_h5:
            # print('error', topic)
            try:
                writer.write(connection, time_now, serialize_message(msg))
            except AttributeError as e:
                print(e)
                import pdb
                pdb.set_trace()
        else:
            # print(topic, msg)
            topics_save_to_h5[topic].append(msg.data)

    return


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
        sync_subscriptions, queue_size=500, slop=0.3, allow_headerless=False
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
        print('starting shutting down, please do not exit')
        rclpy.shutdown()
        writer.close()

        from utils_general import save_dictionary_to_hdf5
        np.save(os.path.join(writer_path, 'other_topics.npy'), topics_save_to_h5)

        print('successfully shut down')


if __name__ == '__main__':
    record()

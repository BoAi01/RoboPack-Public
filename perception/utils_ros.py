import logging
import sys

sys.path.append('/home/albert/github/robopack/ros2_numpy')

import struct
import numpy as np
import ros2_numpy
from perception.utils_pc import *


def decode_pointcloud_msg(pcd_msg, keep_thres=None):
    """
    Extracts PointCloud from deserialized message
    """
    cloud_rec = ros2_numpy.numpify(pcd_msg)

    cloud_array = cloud_rec.view('<f4').reshape(cloud_rec.shape + (-1,))
    points = cloud_array[:, :3]

    cloud_rgb_bytes = cloud_array[:, -1].tobytes()
    cloud_bgr = np.frombuffer(cloud_rgb_bytes, dtype=np.uint8).reshape(-1, 4) / 255
    cloud_rgb = cloud_bgr[:, ::-1]

    # extract PC
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(cloud_rgb[:, 1:])

    # remove too distant points
    if keep_thres is not None:
        pcd = remove_distant_points_pc(pcd, keep_thres)

    return pcd


def decode_multiarray_msg(arr_msg):
    return np.array(arr_msg.data)


def decode_joint_state_msg(joint_msg):
    return np.array(joint_msg.position)


from cv_bridge import CvBridge
bridge = CvBridge()
def decode_img_msg(img_msg, clip_value=None):
    """
    Extracts Image data from deserialized message, which could be RGB or depth messages
    """
    img = bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
    if clip_value is not None:
        img = np.clip(img, 0, clip_value)
    return img


def decode_byte_float32array(byte_array):
    # Calculate the length of each float value (4 bytes for float32)
    float_size = 4

    # Calculate the number of float values in the byte array
    num_floats = len(byte_array) // float_size

    # Use struct.unpack to decode the byte array into a normal array of floats
    # Specify the byte order explicitly as little-endian ('<')
    normal_array = struct.unpack('<' + 'f' * num_floats, byte_array)

    return list(normal_array)


def ros2_pc_to_open3d(rs_pc):
    # extract points from pc raw data
    cloud_rec = ros2_numpy.numpify(rs_pc)
    cloud_array = cloud_rec.view('<f4').reshape(cloud_rec.shape + (-1,))
    points = cloud_array[:, :3]

    # RGB
    cloud_rgb_bytes = cloud_array[:, -1].tobytes()
    cloud_bgr = np.frombuffer(cloud_rgb_bytes, dtype=np.uint8).reshape(-1, 4) / 255
    cloud_rgb = cloud_bgr[:, ::-1]

    # extract PC
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(cloud_rgb[:, 1:])

    return pcd


from message_filters import TimeSynchronizer, Subscriber, ROSClock, Duration, Time, LoggingSeverity
class ApproximateTimeSynchronizer(TimeSynchronizer):

    """
    Approximately synchronizes messages by their timestamps.

    :class:`ApproximateTimeSynchronizer` synchronizes incoming message filters
    by the timestamps contained in their messages' headers. The API is the same
    as TimeSynchronizer except for an extra `slop` parameter in the constructor
    that defines the delay (in seconds) with which messages can be synchronized.
    The ``allow_headerless`` option specifies whether to allow storing
    headerless messages with current ROS time instead of timestamp. You should
    avoid this as much as you can, since the delays are unpredictable.
    """

    def __init__(self, fs, queue_size, slop, allow_headerless=False):
        TimeSynchronizer.__init__(self, fs, queue_size)
        self.slop = Duration(seconds=slop)
        self.allow_headerless = allow_headerless
        self.last = None

    def add(self, msg, my_queue, my_queue_index=None):
        if not hasattr(msg, 'header') or not hasattr(msg.header, 'stamp'):
            if not self.allow_headerless:
                msg_filters_logger = rclpy.logging.get_logger('message_filters_approx')
                msg_filters_logger.set_level(LoggingSeverity.INFO)
                msg_filters_logger.warn("can not use message filters for "
                              "messages without timestamp infomation when "
                              "'allow_headerless' is disabled. auto assign "
                              "ROSTIME to headerless messages once enabling "
                              "constructor option of 'allow_headerless'.")
                return

            stamp = ROSClock().now()
        else:
            stamp = msg.header.stamp
            if not hasattr(stamp, 'nanoseconds'):
                stamp = Time.from_msg(stamp)
            # print(stamp)
        self.lock.acquire()
        my_queue[stamp.nanoseconds] = msg
        while len(my_queue) > self.queue_size:
            del my_queue[min(my_queue)]
        # self.queues = [topic_0 {stamp: msg}, topic_1 {stamp: msg}, ...]
        # old: now, we just search through all queues since we are much more efficient
        #if my_queue_index is None:
        #    search_queues = self.queues
        #else:
        #    search_queues = self.queues[:my_queue_index] + \
        #        self.queues[my_queue_index+1:]
        search_queues = self.queues
        # sort and leave only reasonable stamps for synchronization
        stamps = []
        for i, queue in enumerate(search_queues):
            topic_stamps = []
            for s in queue:
                stamp_delta = Duration(nanoseconds=abs(s - stamp.nanoseconds))
                if stamp_delta > self.slop:
                    continue  # far over the slop
                topic_stamps.append(((Time(nanoseconds=s,
                                   clock_type=stamp.clock_type)), stamp_delta))
            if not topic_stamps:
                # no feasible messages for this topic, give up
                # if self.last and ROSClock().now() - self.last > Duration(seconds=1):
                    # print("no msgs for topic", i)
                self.lock.release()
                return
            # sort by message receive time
            topic_stamps = sorted(topic_stamps, key=lambda x: x[0].nanoseconds)
            stamps.append(topic_stamps)

        # Old synchronization logic
        #print(len(list(itertools.product(*[list(zip(*s))[0] for s in stamps]))))
        #for vv in itertools.product(*[list(zip(*s))[0] for s in stamps]):
        #    vv = list(vv)
        #    # insert the new message
        #    if my_queue_index is not None:
        #        vv.insert(my_queue_index, stamp)
        #    qt = list(zip(self.queues, vv))
        #    if ( ((max(vv) - min(vv)) < self.slop) and
        #        (len([1 for q,t in qt if t.nanoseconds not in q]) == 0) ):
        #        msgs = [q[t.nanoseconds] for q,t in qt]
        #        self.signalMessage(*msgs)
        #        for q,t in qt:
        #            del q[t.nanoseconds]
        #        break  # fast finish after the synchronization

        # get the sorted lists of message receive times for each topic
        time_lists = [list(zip(*s))[0] for s in stamps]
        # find the first set of indices that has all stamps within self.slop
        indices = find_messages_within_epsilon(time_lists, self.slop)
        if indices:
            sync_times = [time_lists[i][idx] for i, idx in enumerate(indices)]
            #print((max(sync_times) - min(sync_times)).nanoseconds / 1e9)
            #print([t.nanoseconds / 1e9 for t in sync_times])
            #print([t.nanoseconds / 1e9 for t in sync_times])
            #print(indices)
            msgs = [self.queues[i][t.nanoseconds] for i, t in enumerate(sync_times)]
            min_time, max_time = float('inf'), float('-inf')
            for msg in msgs:
                msg_time = Time.from_msg(msg.header.stamp).nanoseconds
                if msg_time > max_time:
                    max_time = msg_time
                if msg_time < min_time:
                    min_time = msg_time
            #print([Time.from_msg(msg.header.stamp).nanoseconds/1e9 for msg in msgs])
            #print((max_time - min_time)/ 1e9)
            self.signalMessage(*msgs)
            for q, t in zip(self.queues, sync_times):
                del q[t.nanoseconds]
            self.last = ROSClock().now()

        self.lock.release()
        #print("Return index", my_queue_index)


def find_messages_within_epsilon(arrays, epsilon):
    events = []

    # Create a list of events for each array
    for i, array in enumerate(arrays):
        for j, timestamp in enumerate(array):
            events.append((timestamp, i, j))

    # Sort the events by timestamp
    events.sort()

    n = len(arrays)  # Number of lists
    count = [0] * n  # Count of events from each array within epsilon

    left_pointer = 0
    result_indices = [-1] * n  # Initialize with -1 to indicate no matching element found yet

    for right_pointer in range(len(events)):
        _, array_index, element_index = events[right_pointer]
        count[array_index] += 1

        # Remove events from the left side of the window
        while events[right_pointer][0] - events[left_pointer][0] > epsilon:
            count[events[left_pointer][1]] -= 1
            left_pointer += 1

        # Check if all arrays have events within epsilon
        if all(count[i] > 0 for i in range(n)):
            # Update the result indices for each array
            result_indices[array_index] = element_index

            # Check if all arrays have matching elements
            if all(result_index != -1 for result_index in result_indices):
                return result_indices

    return None  # No solution found


def check_topic_status(node, topic):
    publishers_info = node.get_publishers_info_by_topic(topic)
    if publishers_info:
        logging.debug(f"Topic {topic} is active. Publishers: {publishers_info}")
    else:
        logging.warning(f"No active publishers found for topic {topic}")

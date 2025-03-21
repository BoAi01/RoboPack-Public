import sys
sys.path.append('/home/albert/github/robopack')

import numpy as np
import ros2_numpy

from std_msgs.msg import Float32MultiArray
from rclpy.node import Node
import rclpy
from sensor_msgs.msg import Image, PointCloud2
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.serialization import deserialize_message, serialize_message


class CameraSubscriber(Node):
    """

    """
    def __init__(self):
        super().__init__('realsense_to_open3d_node2')
        print('Listening to 4 cameras')

        self.subscription1 = Subscriber(
            self,
            PointCloud2,
            '/cam_0/depth/color/points',
        )

        self.subscription2 = Subscriber(
            self,
            PointCloud2,
            '/cam_1/depth/color/points',
        )

        self.subscription3 = Subscriber(
            self,
            PointCloud2,
            '/cam_2/depth/color/points',
        )

        self.subscription4 = Subscriber(
            self,
            PointCloud2,
            '/cam_3/depth/color/points',
        )

        # self.subscription5 = Subscriber(
        #     self,
        #     Image,
        #     '/cam_0/depth/image_rect_raw',
        # )
        #
        # self.subscription6 = Subscriber(
        #     self,
        #     Image,
        #     '/cam_1/depth/image_rect_raw',
        # )

        self.synchronizer = ApproximateTimeSynchronizer(
            [self.subscription1, self.subscription2, self.subscription3, self.subscription4],
            queue_size=100,
            slop=0.2
        )

        self.synchronizer.registerCallback(self.callback)
        self.last_reading = None

    def callback(self, *args):
        if self.last_reading is None:
            print('From subscriber: first image received')
        self.last_reading = args

    def get_last_reading(self):
        return self.last_reading

from perception.utils_ros import decode_img_msg
from moviepy.editor import ImageSequenceClip

def save_moviepy_gif(obs_list, name, fps=5):
    clip = ImageSequenceClip(obs_list, fps=fps)
    if name[-4:] != ".gif":
        clip.write_gif(f"{name}.gif", fps=fps)
    else:
        clip.write_gif(name, fps=fps)

def save_moviepy_mp4(obs_list, name, fps=5):
    clip = ImageSequenceClip(obs_list, fps=fps)
    if name[-4:] != ".mp4":
        clip.write_videofile(f"{name}.mp4", fps=fps)
    else:
        clip.write_videofile(name, fps=fps)

class CameraImageSubscriber(Node):
    """
    """

    def callback(self, topic_name, msg):
        if not self.last_readings:
            print('From subscriber: first image received')
        self.last_readings[topic_name] = decode_img_msg(msg)
        # if self.is_recording:
        #     print("Recording caught topic", topic_name)
        # record every 10 frames to save memory
        if self.is_recording:
            if topic_name not in self.recorded_frames.keys():
                self.recorded_frames[topic_name] = []
            if topic_name not in self.recorded_frame_counter.keys():
                self.recorded_frame_counter[topic_name] = 0
            self.recorded_frame_counter[topic_name] += 1
            if self.recorded_frame_counter[topic_name] % 10 == 0:
                self.recorded_frames[topic_name].append(decode_img_msg(msg))

    def start_recording_rgb(self):
        self.is_recording = True

    def stop_recording_rgb(self):
        self.is_recording = False

    def num_recorded_frames(self):
        num_recorded_frames = {}
        for topic_name in self.recorded_frames.keys():
            num_recorded_frames[topic_name] = len(self.recorded_frames[topic_name])
        return num_recorded_frames

    def get_intermediate_frames(self, start_num_recorded_frames_dct, n):
        # return n evenly spaced frames between start_num_recorded_frames and the current number of recorded frames for each topic
        intermediate_frames = {}
        for topic_name in self.recorded_frames.keys():
            start_num_recorded_frames = start_num_recorded_frames_dct[topic_name]
            num_recorded_frames = len(self.recorded_frames[topic_name])
            print("Topic name", topic_name, "start_num_recorded_frames", start_num_recorded_frames, "num_recorded_frames", num_recorded_frames)
            # take n evenly spaced frames between start_num_recorded_frames and num_recorded_frames
            intermediate_frames[topic_name] = []
            for i in range(n):
                idx = int(start_num_recorded_frames + (i + 1) * (num_recorded_frames - start_num_recorded_frames) / (n + 1) * 1/2)
                intermediate_frames[topic_name].append(self.recorded_frames[topic_name][idx])
        return intermediate_frames


    def log_recorded_frames(self, folder):
        print("Logging recorded frames!!")
        for topic_name in self.recorded_frames.keys():
            if 'rgb' in topic_name:
                # write an mp4 file containing the recorded frames
                save_moviepy_mp4(self.recorded_frames[topic_name], f"{folder}/{topic_name[0:5]}.mp4", fps=15)

    def __init__(self):
        super().__init__('realsense_camera_subscriber')

        self.cam0_rgb = Subscriber(
            self,
            Image,
            '/cam_0/color/image_raw',
        )

        self.cam0_depth = Subscriber(
            self,
            Image,
            '/cam_0/depth/image_rect_raw',
        )

        self.cam1_rgb = Subscriber(
            self,
            Image,
            '/cam_1/color/image_raw',
        )

        self.cam1_depth = Subscriber(
            self,
            Image,
            '/cam_1/depth/image_rect_raw',
        )

        self.cam2_rgb = Subscriber(
            self,
            Image,
            '/cam_2/color/image_raw',
        )

        self.cam2_depth = Subscriber(
            self,
            Image,
            '/cam_2/depth/image_rect_raw',
        )

        self.cam3_rgb = Subscriber(
            self,
            Image,
            '/cam_3/color/image_raw',
        )

        self.cam3_depth = Subscriber(
            self,
            Image,
            '/cam_3/depth/image_rect_raw',
        )

        self.last_readings = {}
        self.recorded_frames = {}
        self.recorded_frame_counter = {}
        self.is_recording = False

        self.cam0_rgb.registerCallback(lambda x: self.callback("cam_0_rgb", x))
        self.cam1_rgb.registerCallback(lambda x: self.callback("cam_1_rgb", x))
        self.cam2_rgb.registerCallback(lambda x: self.callback("cam_2_rgb", x))
        self.cam3_rgb.registerCallback(lambda x: self.callback("cam_3_rgb", x))

        self.cam0_depth.registerCallback(lambda x: self.callback("cam_0_depth", x))
        self.cam1_depth.registerCallback(lambda x: self.callback("cam_1_depth", x))
        self.cam2_depth.registerCallback(lambda x: self.callback("cam_2_depth", x))
        self.cam3_depth.registerCallback(lambda x: self.callback("cam_3_depth", x))

        # self.subscription2 = Subscriber(
        #     self,
        #     PointCloud2,
        #     '/cam_1/depth/color/points',
        # )
        #
        # self.subscription3 = Subscriber(
        #     self,
        #     Image,
        #     '/cam_0/color/image_raw',
        # )
        #
        # self.subscription4 = Subscriber(
        #     self,
        #     Image,
        #     '/cam_1/color/image_raw',
        # )
        #
        # self.subscription5 = Subscriber(
        #     self,
        #     Image,
        #     '/cam_0/depth/image_rect_raw',
        # )
        #
        # self.subscription6 = Subscriber(
        #     self,
        #     Image,
        #     '/cam_1/depth/image_rect_raw',
        # )

        # self.synchronizer = ApproximateTimeSynchronizer(
        #     [self.subscription1],
        #     queue_size=100,
        #     slop=0.2
        # )

    def get_last_reading(self):
        return self.last_readings

if __name__ == "__main__":
    # test recording and writing videos
    import time
    rclpy.init()
    camera_subscriber = CameraImageSubscriber()
    camera_subscriber.start_recording_rgb()
    rclpy.spin_once(camera_subscriber, timeout_sec=10)
    time.sleep(10)
    camera_subscriber.stop_recording_rgb()
    camera_subscriber.log_recorded_frames("test")
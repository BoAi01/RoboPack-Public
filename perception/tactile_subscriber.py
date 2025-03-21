

import sys
import pdb

sys.path.append('/home/albert/github/robopack')


import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np

import cv2
import threading
from perception.utils_cv import visualize_raw_flow


class TactileSubscriber(Node):
    def __init__(self):
        super().__init__('multi_topic_subscriber')

        self.subscriber_1 = self.create_subscription(
            Float32MultiArray,
            '/bubble_1/raw_flow',
            self.callback_bubble_1,
            10)

        self.subscriber_2 = self.create_subscription(
            Float32MultiArray,
            '/bubble_2/raw_flow',
            self.callback_bubble_2,
            10)

        self.timer = self.create_timer(0.2, self.timer_callback)  # 2-second interval
        self.flow_arrows_b1 = None
        self.flow_arrows_b2 = None

    def convert_msg_to_np_array(self, msg):
        return np.array(msg.data, dtype=np.float32).reshape(240, 320, 2)

    def callback_bubble_1(self, msg):
        data_array = self.convert_msg_to_np_array(msg)
        self.flow_arrows_b1 = data_array
        # self.get_logger().info(f'Received data from bubble 1: shape {data_array.shape}')
        raw_flow_arrows = visualize_raw_flow(data_array)
        cv2.imshow("Bubble1-received flow", raw_flow_arrows)
        cv2.waitKey(1)

    def callback_bubble_2(self, msg):
        data_array = self.convert_msg_to_np_array(msg)
        self.flow_arrows_b2 = data_array
        # self.get_logger().info(f'Received data from bubble 1: shape {data_array.shape}')
        raw_flow_arrows = visualize_raw_flow(data_array)
        cv2.imshow("Bubble2-received flow", raw_flow_arrows)
        cv2.waitKey(1)

    def get_mean_diff(self):
        force_diff = self.flow_arrows_b1 - self.flow_arrows_b2
        return force_diff.reshape(-1, 2).mean(0)

    def timer_callback(self):
        # self.get_logger().info('Timer callback triggered')
        # Add your custom logic here that needs to be executed at regular intervals

        if self.flow_arrows_b1 is not None and self.flow_arrows_b2 is not None:
            force_diff = self.flow_arrows_b1 - self.flow_arrows_b2
            # self.get_logger().info(f'force_diff mean = {self.flow_arrows_b1.reshape(-1, 2).mean(0)}, '
            #                        f'{self.flow_arrows_b2.reshape(-1, 2).mean(0)}')

            # def classify_tactile(b1_tactile, b2_tactile, threshold=1):
            #     assert threshold > 0
            #
            #     b1_mean, b2_mean = b1_tactile.reshape(-1, 2).mean(0), b2_tactile.reshape(-1, 2).mean(0)
            #     diff = b1_mean - b2_mean
            #
            #     if b1_mean[0] < -3 and b2_mean[0] < -3:
            #         return "Encountering obstacle"
            #
            #     if diff[0] > threshold:
            #         return "move to left"
            #     elif diff[0] < -threshold:
            #         return "move to the right"
            #     else:
            #         return "fine"

            # self.get_logger().info(f'force_diff mean = {classify_tactile(self.flow_arrows_b1, self.flow_arrows_b2)}')

            diff_vis = visualize_raw_flow(force_diff)
            cv2.imshow("bubble1 - bubble2", diff_vis)
            cv2.waitKey(1)

    def get_readings(self):
        readings = [self.flow_arrows_b1, self.flow_arrows_b2]
        # assert None not in readings
        return readings


def main(args=None):
    rclpy.init(args=args)
    node = TactileSubscriber()

    ros_thread = threading.Thread(target=lambda: rclpy.spin(node))
    ros_thread.start()
    ros_thread.join()
    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()

import sys
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
from message_filters import Subscriber, TimeSynchronizer, ApproximateTimeSynchronizer


class ImageSubscriberNode(Node):
    NUM_CAMERAS = 2

    def __init__(self, camera_names):
        super().__init__('image_subscriber_node')
        assert len(camera_names) == self.NUM_CAMERAS, f"does not support cameras more than {self.NUM_CAMERAS}"
        self.cv_bridge = CvBridge()
        self.subscriber1_rgb = Subscriber(self, Image, f'/{camera_names[0]}/color/image_raw')
        self.subscriber2_rgb = Subscriber(self, Image, f'/{camera_names[1]}/color/image_raw')
        # self.subscriber3_rgb = Subscriber(self, Image, '/camera_top/color/image_raw')
        self.subscriber1_depth = Subscriber(self, Image, f'/{camera_names[0]}/depth/image_rect_raw')
        self.subscriber2_depth = Subscriber(self, Image, f'/{camera_names[1]}/depth/image_rect_raw')
        # self.subscriber3_depth = Subscriber(self, Image, '/camera_top/depth/image_rect_raw')
        self.subscriber1_points = Subscriber(self, PointCloud2, f'/{camera_names[0]}/depth/color/points')
        self.subscriber2_points = Subscriber(self, PointCloud2, f'/{camera_names[1]}/depth/color/points')
        # self.subscriber3_points= Subscriber(self, PointCloud2, '/camera_top/depth/color/points')
        self.synchronizer = ApproximateTimeSynchronizer(
            [self.subscriber1_rgb, self.subscriber2_rgb,
             self.subscriber1_depth, self.subscriber2_depth,
             self.subscriber1_points, self.subscriber2_points], queue_size=10, slop=1)
        self.synchronizer.registerCallback(self.callback)

        self.last_points = []

    def rgb_msg_to_numpy(self, img):
        return self.cv_bridge.imgmsg_to_cv2(img, desired_encoding='bgr8')

    def depth_msg_to_numpy(self, dimg):
        return self.cv_bridge.imgmsg_to_cv2(dimg, desired_encoding='passthrough')

    def callback(self, img1, img2, dimg1, dimg2, points1, points2):
        # Convert ROS images to OpenCV format
        cv_img1 = self.rgb_msg_to_numpy(img1)
        cv_img2 = self.rgb_msg_to_numpy(img2)
        # cv_img3 = self.rgb_msg_to_numpy(img3)
        depth_image1 = self.depth_msg_to_numpy(dimg1)
        depth_image2 = self.depth_msg_to_numpy(dimg2)
        # depth_image3 = self.depth_msg_to_numpy(dimg3)

        # visualize
        # cv2.imshow('Image 1', cv_img1)
        # cv2.imshow('Image 2', cv_img2)
        # cv2.imshow('Image 3', cv_img3)
        # cv2.waitKey(1)

        self.last_points = [points1, points2]

        return points1, points2


def main(args=None):
    if args is None:
        args = sys.argv

    rclpy.init(args=args)
    node = ImageSubscriberNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

import rospy
from threading import Thread
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
from message_filters import Subscriber, ApproximateTimeSynchronizer


class ImageSubscriberNode(object):
    NUM_CAMERAS = 2

    def __init__(self, camera_names):
        rospy.init_node('image_subscriber_node')

        assert len(camera_names) == self.NUM_CAMERAS, f"Does not support cameras more than {self.NUM_CAMERAS}"

        self.cv_bridge = CvBridge()
        self.subscriber1_rgb = Subscriber(f'/{camera_names[0]}/color/image_raw', Image)
        self.subscriber2_rgb = Subscriber(f'/{camera_names[1]}/color/image_raw', Image)
        # self.subscriber3_rgb = Subscriber('/camera_top/color/image_raw', Image)
        self.subscriber1_depth = Subscriber(f'/{camera_names[0]}/depth/image_rect_raw', Image)
        self.subscriber2_depth = Subscriber(f'/{camera_names[1]}/depth/image_rect_raw', Image)
        # self.subscriber3_depth = Subscriber('/camera_top/depth/image_rect_raw', Image)
        self.subscriber1_points = Subscriber(f'/{camera_names[0]}/depth/color/points', PointCloud2)
        self.subscriber2_points = Subscriber(f'/{camera_names[1]}/depth/color/points', PointCloud2)
        # self.subscriber3_points = Subscriber('/camera_top/depth/color/points', PointCloud2)
        self.synchronizer = ApproximateTimeSynchronizer(
            [self.subscriber1_rgb, self.subscriber2_rgb,
             self.subscriber1_depth, self.subscriber2_depth,
             self.subscriber1_points, self.subscriber2_points], queue_size=100, slop=1)
        self.synchronizer.registerCallback(self.callback)

        self.last_callback_inputs = []

    def rgb_msg_to_numpy(self, img):
        return self.cv_bridge.imgmsg_to_cv2(img, desired_encoding='bgr8')

    def depth_msg_to_numpy(self, dimg):
        return self.cv_bridge.imgmsg_to_cv2(dimg, desired_encoding='passthrough')

    def callback(self, img1, img2, dimg1, dimg2, points1, points2):
        print(1)

        self.last_callback_inputs = [img1, img2, dimg1, dimg2, points1, points2]

        # Convert ROS images to OpenCV format
        cv_img1 = self.rgb_msg_to_numpy(img1)
        cv_img2 = self.rgb_msg_to_numpy(img2)
        #         cv_img3 = self.rgb_msg_to_numpy(img3)
        depth_image1 = self.depth_msg_to_numpy(dimg1)
        depth_image2 = self.depth_msg_to_numpy(dimg2)
        #         depth_image3 = self.depth_msg_to_numpy(dimg3)

        # visualize
        # cv2.imshow('Image 1', cv_img1)
        # cv2.imshow('Image 2', cv_img2)
        # cv2.imshow('Image 3', cv_img3)
        # cv2.waitKey(1)

        return points1, points2

    def start(self):
        #         rospy.init_node('listener')
        # thread = Thread(target=rospy.spin)
        # thread.start()
        rospy.spin()

import rospy
from threading import Thread

def spin_ros():
    while not rospy.is_shutdown():
        rospy.spin()

node = ImageSubscriberNode(['cam_0', 'cam_1'])

# Create a new thread and set it to execute spin_ros()
thread = Thread(target=spin_ros)

# Start the thread
thread.start()

# Print "3" in the main thread
print(3)
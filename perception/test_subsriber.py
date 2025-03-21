import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

"""
Subscribes to an image node 
"""
class SubscriberNode(Node):
    def __init__(self, topic):
        super().__init__('subscriber_node')
        self.subscription = self.create_subscription(
            Image,
            topic,
            self.callback,
            10  # Buffer size
        )
        self.received_messages = 0

    def callback(self, msg):
        self.received_messages += 1
        print(f"Received message {self.received_messages}: {msg.data}")
        if self.received_messages >= 5:
            self.subscription.destroy()
            self.get_logger().info("Exiting...")

def main(args=None):
    if args is None:
        args = sys.argv

    if len(args) < 2:
        print("Please provide a topic name to subscribe to.")
        return

    topic = args[1]
    print(f'listening on topic: {topic}')

    rclpy.init(args=args)
    node = SubscriberNode(topic)
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

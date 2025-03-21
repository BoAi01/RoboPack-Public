import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray
from custom_msg.msg import StampedFloat32MultiArray
from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from transforms3d.quaternions import mat2quat


class RobotStatePublisherNode(Node):
    def __init__(self, robot_interface):
        super().__init__('joint_state_publisher')
        self.robot_interface = robot_interface

        self.joint_pos_pub = self.create_publisher(
            JointState, '/joint_states', QoSProfile(depth=10)
        )
        self.joint_names = [f'panda_joint{i}' for i in range(1, 8)]
        self.joint_names += ['panda_finger_joint1', 'panda_finger_joint1']

        self.ee_states_pub = self.create_publisher(
            StampedFloat32MultiArray, '/ee_states', 30
        )

        self.timer = self.create_timer(1/30, self.publish_joint_and_ee_states)

    def get_joint_states(self):
        joint_pos = self.robot_interface.last_q
        joint_vel = np.zeros(7)
        gripper_state = self.robot_interface.last_gripper_q

        if joint_pos is None:
            self.get_logger().info('joint_pos is None')
        if gripper_state is None:
            self.get_logger().info('gripper_state is None')

        return joint_pos.tolist(), joint_vel.tolist(), gripper_state

    def get_ee_states(self):
        ee_rot, ee_pos = self.robot_interface.last_eef_rot_and_pos

        if ee_rot is None or ee_pos is None:
            self.get_logger().info('None in ee state')

        ee_quat = mat2quat(ee_rot)      # vector of length 4
        ee_pos = np.squeeze(ee_pos, 1)  # vector of length 3

        return ee_pos.tolist(), ee_quat.tolist()

    def publish_joint_and_ee_states(self):
        # print('publishing joint and ee states')
        joint_pos, joint_vel, gripper_state = self.get_joint_states()

        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.name = self.joint_names
        # print('gripper state', gripper_state, 'joint_pos', joint_pos)
        joint_state.position = joint_pos + [gripper_state.tolist()]
        # joint_state.velocity = joint_vel + [0.0, 0.0]
        self.joint_pos_pub.publish(joint_state)

        ee_state_msg = Float32MultiArray()
        # ee_state_msg.header.stamp = self.get_clock().now().to_msg()
        ee_pos, ee_rot = self.get_ee_states()
        ee_state_msg.data = ee_pos + ee_rot   # x y z w x y z
        ee_state_msg_stamped = StampedFloat32MultiArray()
        ee_state_msg_stamped.header.stamp = self.get_clock().now().to_msg()
        ee_state_msg_stamped.data = ee_state_msg
        self.ee_states_pub.publish(ee_state_msg_stamped)


def main(args=None):
    rclpy.init(args=args)

    interface_cfg = "charmander.yml"
    robot_interface = FrankaInterface(
        config_root + f"/{interface_cfg}", use_visualizer=False
    )
    time.sleep(3)   # robot interface initialization takes time

    node = RobotStatePublisherNode(robot_interface)

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

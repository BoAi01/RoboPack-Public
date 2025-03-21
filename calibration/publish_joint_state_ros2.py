import sys
sys.path.append('/home/albert/github/robopack')

import time
import numpy as np
import rclpy
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray
from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from transforms3d.quaternions import mat2quat
from utils_deoxys import init_franka_interface


def get_joint_states(robot_interface):
    joint_pos = robot_interface.last_q
    joint_vel = np.zeros(7)
    gripper_state = robot_interface.last_gripper_q

    if joint_pos is None:
        print(f'joint_pos is None')
    if gripper_state is None:
        print(f'gripper_state is None')

    return joint_pos.tolist(), joint_vel.tolist(), gripper_state


def get_ee_states(robot_interface):
    ee_rot, ee_pos = robot_interface.last_eef_rot_and_pos

    if ee_rot is None or ee_pos is None:
        print(f'None in ee state')

    ee_quat = mat2quat(ee_rot)      # vector of length 4
    ee_pos = np.squeeze(ee_pos, 1)      # vector of length 3

    return ee_pos.tolist(), ee_quat.tolist()


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('joint_state_publisher')

    robot_interface = init_franka_interface()

    # joint states
    joint_pos_pub = node.create_publisher(JointState, '/joint_states', QoSProfile(depth=10))
    joint_names = [f'panda_joint{i}' for i in range(1, 8)]
    joint_names += ['panda_finger_joint1', 'panda_finger_joint2']

    # ee states
    ee_states_pub = node.create_publisher(Float32MultiArray, '/ee_states', 10)
    joint_names = ['ee_quat', 'ee_pos']

    print('publishing states to ros2 topics...')
    while rclpy.ok():
        joint_pos, joint_vel, gripper_state = get_joint_states(robot_interface)

        joint_state = JointState()
        joint_state.header.stamp = node.get_clock().now().to_msg()
        joint_state.name = joint_names
        # print(type(gripper_state), joint_pos)
        joint_state.position = joint_pos + [float(gripper_state), float(gripper_state)]
        joint_state.velocity = joint_vel + [0.0, 0.0]
        joint_pos_pub.publish(joint_state)

        ee_state_msg = Float32MultiArray()
        # ee_state_msg.header.stamp = node.get_clock().now().to_msg()
        ee_pos, ee_rot = get_ee_states(robot_interface)
        ee_state_msg.data = ee_pos + ee_rot   # x y z w x y z
        print('ee xyz wxyz: ', ee_pos, ee_rot)
        print('gripper distance: ', gripper_state)
        ee_states_pub.publish(ee_state_msg)

        time.sleep(1 / 30)  # 30 hz

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

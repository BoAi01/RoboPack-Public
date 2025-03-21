"""
Publish joint states of the robot so that rivz could  capture it
Run under native (ROS 1) environment.
"""

import time

import numpy as np
import rospy

from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from sensor_msgs.msg import JointState


def get_joint_states(robot_interface):
    joint_pos = robot_interface.last_q
    joint_vel = np.zeros(7)
    gripper_state = robot_interface.last_gripper_q
    return joint_pos.tolist(), joint_vel.tolist(), gripper_state


def main():
    rospy.init_node('joint_state_publisher', anonymous=True)

    interface_cfg = "charmander.yml"
    robot_interface = FrankaInterface(
        config_root + f"/{interface_cfg}", use_visualizer=False
    )
    time.sleep(3)  # robot interface initialization takes time

    joint_pos_pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
    joint_names = [f'panda_joint{i}' for i in range(1, 8)]
    joint_names += ['panda_finger_joint1', 'panda_finger_joint1']
    rate = rospy.Rate(15)
    while not rospy.is_shutdown():
        rate.sleep()
        joint_pos, joint_vel, gripper_state = get_joint_states(robot_interface)
        joint_state = JointState()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = joint_names
        joint_state.position = joint_pos + [gripper_state*0.5, gripper_state*0.5]
        joint_state.velocity = joint_vel + [0.0, 0.0]
        joint_pos_pub.publish(joint_state)


if __name__ == "__main__":
    main()

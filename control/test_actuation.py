import sys
sys.path.append('/home/albert/github/robopack')

from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.config_utils import (add_robot_config_arguments,
                                       get_default_controller_config)
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.log_utils import get_deoxys_example_logger

import random
from utils_deoxys import init_franka_interface
import numpy as np
from control.motion_utils import *


def sample_xy_in_range(xrange, yrange):
    """
    Randomly samples a coordinate (x, y) in the given range
    :param xrange: range of x
    :param yrange: range of y
    :return: (x, y)
    """
    x = random.uniform(*xrange)
    y = random.uniform(*yrange)
    return [x, y]


num_trials = 10
errs = []


def main():
    robot_interface = init_franka_interface()
    target_ee_pos = np.array([0.5, 0, 0.3])
    position_only_gripper_move_to(
        robot_interface, target_ee_pos, num_steps=100, allowance=0.005  # need 50 steps for accurate actuation
    )
    target_ee_pos = np.array([0.5, -0.2, 0.3])
    position_only_gripper_move_to(
        robot_interface, target_ee_pos, num_steps=100, allowance=0.005  # need 50 steps for accurate actuation
    )
    import pdb; pdb.set_trace()
    for i in range(num_trials):
        # target_ee_pos = np.array(sample_xy_in_range([0.3, 0.6], [-0.2, 0.2]) + [0.3])
        # ee_home_pose = np.array([[0.35], [0], [0.50]])  # back off and stay high to make space
        # target_ee_pos = target_ee_pos + np.array(sample_xy_in_range([-0.02, 0.02], [-0.02, 0.02]) + [0.0])
        # move in a 2cm x 2cm square around the target
        if i == 0:
            target_ee_pos += np.array([0.0, 0.0, 0.02])
        elif i == 1:
            target_ee_pos += np.array([0.0, 0.0, -0.02])
        elif i == 2:
            target_ee_pos += np.array([0.0, 0.02, 0.0])
        elif i == 3:
            target_ee_pos += np.array([0.0, -0.02, 0.0])
        elif i == 4:
            target_ee_pos += np.array([0.02, 0.0, 0.0])
        elif i == 5:
            target_ee_pos += np.array([-0.02, 0.0, 0.0])
        else:
            break
        position_only_gripper_move_to(
            robot_interface, target_ee_pos, num_steps=25, allowance=0.005   # need 50 steps for accurate actuation
        )
        ee_rot, ee_pos = robot_interface.last_eef_rot_and_pos
        ee_pos = ee_pos.squeeze(1)
        print(f'target ee pos: {target_ee_pos}, actual pos: {ee_pos}')

        errs.append(abs(ee_pos - target_ee_pos))

        print(f'err mean = {np.mean(errs, axis=0)}, std = {np.std(errs, axis=0)}')


if __name__ == '__main__':
    main()

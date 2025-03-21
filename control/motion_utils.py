"""
This is an experimental file where we have some standard abstractions for certain manipulation behaviors. This part will be made standard once we've tested.
"""
import time
import warnings
from typing import Union

import numpy as np
from deoxys.utils.transform_utils import mat2euler
from deoxys.utils.config_utils import (get_default_controller_config,
                                       verify_controller_config)

# Joint space motion abstractions


def reset_joints_to(
    robot_interface,
    start_joint_pos: Union[list, np.ndarray],
    controller_cfg: dict = None,
    timeout=7,
):
    assert type(start_joint_pos) is list or type(start_joint_pos) is np.ndarray
    if controller_cfg is None:
        controller_cfg = get_default_controller_config(controller_type="JOINT_POSITION")
    else:
        assert controller_cfg["controller_type"] == "JOINT_POSITION", (
            "This function is only for JOINT POSITION mode. You specified "
            + controller_cfg["controller_type"]
        )
        controller_cfg = verify_controller_config(controller_cfg)
    if type(start_joint_pos) is list:
        action = start_joint_pos + [-1.0]
    else:
        action = start_joint_pos.tolist() + [-1.0]
    start_time = time.time()
    while True:
        if (
            robot_interface.received_states
            and robot_interface.check_nonzero_configuration()
        ):
            if (
                np.max(
                    np.abs(np.array(robot_interface.last_q) - np.array(start_joint_pos))
                )
                < 1e-3
            ):
                break
        robot_interface.control(
            controller_type="JOINT_POSITION",
            action=action,
            controller_cfg=controller_cfg,
        )
        end_time = time.time()

        # Add timeout
        if end_time - start_time > timeout:
            break
    robot_interface.close()
    return True


def joint_interpolation_traj(
    start_q, end_q, num_steps=100, traj_interpolator_type="min_jerk"
):
    assert traj_interpolator_type in ["min_jerk", "linear"]

    traj = []

    if traj_interpolator_type == "min_jerk":
        for i in range(0, num_steps + 1):
            t = float(i) * (1 / num_steps)
            transformed_step_size = 10.0 * (t**3) - 15 * (t**4) + 6 * (t**5)
            traj.append(start_q + transformed_step_size * (end_q - start_q))
        traj = np.array(traj)
    elif traj_interpolator_type == "linear":
        step_size = (end_q - start_q) / float(num_steps)
        grid = np.arange(num_steps).astype(np.float64)
        traj = np.array([start_q + grid[i] * step_size for i in range(num_steps)])

        # add endpoint
        traj = np.concatenate([traj, end_q[None]], axis=0)
    return traj


def follow_joint_traj(
    robot_interface,
    joint_traj: list,
    num_addition_steps=30,
    controller_cfg: dict = None,
):
    """This is a simple function to follow a given trajectory in joint space.

    Args:
        robot_interface (FrankaInterface): _description_
        joint_traj (list): _description_
        num_addition_steps (int, optional): _description_. Defaults to 100.
        controller_cfg (dict, optional): controller configurations. Defaults to None.
    Returns:
        joint_pos_history (list): a list of recorded joint positions
        action_history (list): a list of recorded action commands
    """

    if controller_cfg is None:
        controller_cfg = get_default_controller_config(
            controller_type="JOINT_IMPEDANCE"
        )
    else:
        assert controller_cfg["controller_type"] == "JOINT_IMPEDANCE", (
            "This function is only for JOINT IMPEDANCE mode. You specified "
            + controller_cfg["controller_type"]
        )
        controller_cfg = verify_controller_config(controller_cfg)

    prev_action = np.array([0.0] * 8)
    joint_pos_history = []
    action_history = []

    assert (
        robot_interface.last_q is not None
        and robot_interface.check_nonzero_configuration()
    )

    for target_joint_pos in joint_traj:
        assert len(target_joint_pos) >= 7
        if type(target_joint_pos) is np.ndarray:
            action = target_joint_pos.tolist()
        else:
            action = target_joint_pos
        if len(action) == 7:
            action = action + [-1.0]
        current_joint_pos = np.array(robot_interface.last_q)
        robot_interface.control(
            controller_type="JOINT_IMPEDANCE",
            action=action,
            controller_cfg=controller_cfg,
        )
        joint_pos_history.append(current_joint_pos.flatten().tolist())
        action_history.append(prev_action.tolist())
        prev_action = np.array(action)

    for i in range(num_addition_steps):
        current_joint_pos = np.array(robot_interface.last_q)
        robot_interface.control(
            controller_type="JOINT_IMPEDANCE",
            action=action,
            controller_cfg=controller_cfg,
        )
        joint_pos_history.append(current_joint_pos.flatten().tolist())
        action_history.append(prev_action.tolist())
        prev_action = np.array(action)

    return joint_pos_history, action_history


def reset_robot_z(robot_interface, set_position=None):

    controller_cfg = get_default_controller_config(controller_type="JOINT_POSITION")
    controller_type = "JOINT_POSITION"

    # Golden resetting joints
    reset_joint_positions = [
        0.09162008114028396,
        -0.19826458111314524,
        -0.01990020486871322,
        -2.4732269941140346,
        -0.01307073642274261,
        2.30396583422025,
        0.8480939705504309,
    ]

    # This is for varying initialization of joints a little bit to
    # increase data variation.
    # reset_joint_positions = [
    #     e + np.clip(np.random.randn() * 0.005, -0.005, 0.005)
    #     for e in reset_joint_positions
    # ]
    # get the most recent joint positions
    action = robot_interface.last_q.tolist() + [-1.0]
    # action = reset_joint_positions + [-1.0]
    if set_position:
        action[-2] = reset_joint_positions[-1] + set_position
    else:
        action[-2] = reset_joint_positions[-1]
    while True:
        if len(robot_interface._state_buffer) > 0:

            if (
                    np.max(
                        np.abs(
                            np.array(robot_interface._state_buffer[-1].q)
                            - np.array(action[:-1])
                        )[-1]
                    )
                    < 1e-3
            ):
                break
        robot_interface.control(
            controller_type=controller_type,
            action=action,
            controller_cfg=controller_cfg,
        )
    print("Done resetting")

def position_only_gripper_move_to_keep_yaw(
    robot_interface, target_pos, num_steps=100, controller_cfg: dict = None, grasp=False, max_speed=None, allowance=0.02,
        target_yaw=None,
):
    """_summary_

    Args:
        robot_interface (FrankaInterface): the python interface for robot control
        target_pos (np.array or list): target xyz location
        num_steps (int, optional): number of steps to control. Defaults to 100.
        controller_cfg (dict, optional): controller configurations. Defaults to None.
        grasp (bool, optional): close the gripper if set to True. Defaults to False.
    Return:
        eef_pos_history (list): a list of recorded end effector positions
        action_history (list): a list of recorded action commands
    """
    if max_speed is None:
        max_speed = 0.4
    assert max_speed < 0.6, f'max_speed = {max_speed} is too high, should be less than 0.6'

    if np.array(target_pos).shape == (3,):
        target_pos = np.expand_dims(np.array(target_pos), axis=1)

    if controller_cfg is None:
        controller_cfg = get_default_controller_config(controller_type="OSC_YAW")
    else:
        assert controller_cfg["controller_type"] == "OSC_YAW", (
            "This function is only for OSC_POSITION mode. You specified "
            + controller_cfg["controller_type"]
        )
        controller_cfg = verify_controller_config(controller_cfg)
    eef_pos_history = []
    action_history = []

    current_pos = None
    while current_pos is None:
        _, current_pos = robot_interface.last_eef_rot_and_pos

    prev_action = np.array([0.0] * 6 + [int(grasp) * 2 - 1])
    windup = 0.15

    for i in range(num_steps):
        _, current_pos = robot_interface.last_eef_rot_and_pos
        action = np.array([0.0] * 6 + [int(grasp) * 2 - 1])
        action[:3] = ((target_pos - current_pos).flatten()) * 10
        # bound mean and max value to avoid drastic instant motion or under-actuation
        action[:3] = np.clip(action[:3], -max_speed, max_speed)
        if target_yaw is not None:
            action[5] = (target_yaw - mat2euler(robot_interface.last_eef_rot_and_pos[0])[2]) * 10
        # if np.all(np.abs(action[:3]) < 0.01):
        #     action[:3] = 0.0
        # elif np.all(np.abs(action[:3]) < 0.05):
        #     action[:3] = action[:3] / np.linalg.norm(action[:3]) * 0.05

        # if the eef_pos_history shows that the robot hasn't moved much in the last 5 steps, increase the speed
        if len(eef_pos_history) > 5:
            last_5_steps = np.array(eef_pos_history[-5:])
            if np.all(np.abs(last_5_steps - last_5_steps[0]) < 0.005):
                # print("robot hasn't moved much in the last 5 steps, increasing speed")
                windup *= 1.1
                action[:3] = action[:3] / np.linalg.norm(action[:3]) * windup
            else:
                windup = 0.15
        robot_interface.control(
            controller_type="OSC_YAW", action=action, controller_cfg=controller_cfg
        )
        eef_pos_history.append(current_pos.flatten().tolist())
        action_history.append(prev_action)
        prev_action = np.array(action)

        # adding check
        _, current_pos = robot_interface.last_eef_rot_and_pos
        diff = (target_pos - current_pos).squeeze(1)
        # print("Before sleep diff is: ", diff)

        if (abs(diff) < allowance).all():
            print(f"Early break, reached target position in {i} steps")
            break

    # import time; time.sleep(2)
    # _, current_pos = robot_interface.last_eef_rot_and_pos
    # diff = (target_pos - current_pos).squeeze(1)
    # print("After sleep diff is: ", diff)
    if (abs(diff) > allowance).any():
        print(f'robot might be under-actuating, given error {diff} larger than {allowance}')
        print(f"Robot target position was: {target_pos.squeeze()}")
        print(f"Robot final position was: {current_pos.squeeze()}")

    return eef_pos_history, action_history

def position_only_gripper_move_to(
    robot_interface, target_pos, num_steps=100, controller_cfg: dict = None, grasp=False, max_speed=None, allowance=0.02
):
    """_summary_

    Args:
        robot_interface (FrankaInterface): the python interface for robot control
        target_pos (np.array or list): target xyz location
        num_steps (int, optional): number of steps to control. Defaults to 100.
        controller_cfg (dict, optional): controller configurations. Defaults to None.
        grasp (bool, optional): close the gripper if set to True. Defaults to False.
    Return:
        eef_pos_history (list): a list of recorded end effector positions
        action_history (list): a list of recorded action commands
    """
    if max_speed is None:
        max_speed = 0.4
    assert max_speed < 0.6, f'max_speed = {max_speed} is too high, should be less than 0.6'

    if np.array(target_pos).shape == (3,):
        target_pos = np.expand_dims(np.array(target_pos), axis=1)

    if controller_cfg is None:
        controller_cfg = get_default_controller_config(controller_type="OSC_POSITION")
    else:
        assert controller_cfg["controller_type"] == "OSC_POSITION", (
            "This function is only for OSC_POSITION mode. You specified "
            + controller_cfg["controller_type"]
        )
        controller_cfg = verify_controller_config(controller_cfg)
    eef_pos_history = []
    action_history = []

    current_pos = None
    while current_pos is None:
        _, current_pos = robot_interface.last_eef_rot_and_pos

    prev_action = np.array([0.0] * 6 + [int(grasp) * 2 - 1])
    windup = 0.15

    for i in range(num_steps):
        _, current_pos = robot_interface.last_eef_rot_and_pos
        action = np.array([0.0] * 6 + [int(grasp) * 2 - 1])
        action[:3] = ((target_pos - current_pos).flatten()) * 10
        # bound mean and max value to avoid drastic instant motion or under-actuation
        action[:3] = np.clip(action[:3], -max_speed, max_speed)
        # if np.all(np.abs(action[:3]) < 0.01):
        #     action[:3] = 0.0
        # elif np.all(np.abs(action[:3]) < 0.05):
        #     action[:3] = action[:3] / np.linalg.norm(action[:3]) * 0.05

        # if the eef_pos_history shows that the robot hasn't moved much in the last 5 steps, increase the speed
        if len(eef_pos_history) > 5:
            last_5_steps = np.array(eef_pos_history[-5:])
            if np.all(np.abs(last_5_steps - last_5_steps[0]) < 0.005):
                # print("robot hasn't moved much in the last 5 steps, increasing speed")
                windup *= 1.1
                action[:3] = action[:3] / np.linalg.norm(action[:3]) * windup
            else:
                windup = 0.15
        robot_interface.control(
            controller_type="OSC_POSITION", action=action, controller_cfg=controller_cfg
        )
        eef_pos_history.append(current_pos.flatten().tolist())
        action_history.append(prev_action)
        prev_action = np.array(action)

        # adding check
        _, current_pos = robot_interface.last_eef_rot_and_pos
        diff = (target_pos - current_pos).squeeze(1)
        # print("Before sleep diff is: ", diff)

        if (abs(diff) < allowance).all():
            print(f"Early break, reached target position in {i} steps")
            break

    # import time; time.sleep(2)
    # _, current_pos = robot_interface.last_eef_rot_and_pos
    # diff = (target_pos - current_pos).squeeze(1)
    # print("After sleep diff is: ", diff)
    if (abs(diff) > allowance).any():
        print(f'robot might be under-actuating, given error {diff} larger than {allowance}')
        print(f"Robot target position was: {target_pos.squeeze()}")
        print(f"Robot final position was: {current_pos.squeeze()}")

    return eef_pos_history, action_history


def position_only_gripper_move_to_waypoints(
    robot_interface, target_positions, num_steps=100, controller_cfg: dict = None, grasp=False, max_speed=None, allowance=0.02, min_speed=0.1
):
    """_summary_

    Args:
        robot_interface (FrankaInterface): the python interface for robot control
        target_pos (np.array or list): target xyz location
        num_steps (int, optional): number of steps to control. Defaults to 100.
        controller_cfg (dict, optional): controller configurations. Defaults to None.
        grasp (bool, optional): close the gripper if set to True. Defaults to False.
    Return:
        eef_pos_history (list): a list of recorded end effector positions
        action_history (list): a list of recorded action commands
    """
    if max_speed is None:
        max_speed = 0.4
    assert max_speed < 0.6, f'max_speed = {max_speed} is too high, should be less than 0.6'

    if controller_cfg is None:
        controller_cfg = get_default_controller_config(controller_type="OSC_POSITION")
    else:
        assert controller_cfg["controller_type"] == "OSC_POSITION", (
            "This function is only for OSC_POSITION mode. You specified "
            + controller_cfg["controller_type"]
        )
        controller_cfg = verify_controller_config(controller_cfg)
    eef_pos_history = []
    action_history = []

    current_pos = None
    while current_pos is None:
        _, current_pos = robot_interface.last_eef_rot_and_pos

    prev_action = np.array([0.0] * 6 + [int(grasp) * 2 - 1])
    windup = 0.15
    current_target_idx = 0

    for i in range(num_steps):
        if current_target_idx >= len(target_positions):
            break
        target_pos = target_positions[current_target_idx]
        _, current_pos = robot_interface.last_eef_rot_and_pos
        action = np.array([0.0] * 6 + [int(grasp) * 2 - 1])
        action[:3] = ((target_pos - current_pos).flatten()) * 10
        # bound mean and max value to avoid drastic instant motion or under-actuation
        # if it's not the last target position yet, we want to each action to have absolute value least at min_speed m/s
        if current_target_idx < len(target_positions) - 1:
            # if actions are too small, we want to increase the speed
            if np.linalg.norm(action[:3]) < min_speed:
                action[:3] = action[:3] / np.linalg.norm(action[:3]) * min_speed

        action[:3] = np.clip(action[:3], -max_speed, max_speed)
        # if np.all(np.abs(action[:3]) < 0.01):
        #     action[:3] = 0.0
        # elif np.all(np.abs(action[:3]) < 0.05):
        #     action[:3] = action[:3] / np.linalg.norm(action[:3]) * 0.05

        # if the eef_pos_history shows that the robot hasn't moved much in the last 5 steps, increase the speed
        if len(eef_pos_history) > 5:
            last_5_steps = np.array(eef_pos_history[-5:])
            if np.all(np.abs(last_5_steps - last_5_steps[0]) < 0.005):
                # print("robot hasn't moved much in the last 5 steps, increasing speed")
                windup *= 1.03
                action[:3] = action[:3] / np.linalg.norm(action[:3]) * windup
            else:
                windup = 0.15
        robot_interface.control(
            controller_type="OSC_POSITION", action=action, controller_cfg=controller_cfg
        )
        eef_pos_history.append(current_pos.flatten().tolist())
        action_history.append(prev_action)
        prev_action = np.array(action)

        # adding check
        _, current_pos = robot_interface.last_eef_rot_and_pos
        diff = (target_pos - current_pos).squeeze(1)
        # print("Before sleep diff is: ", diff)

        if ((abs(diff) < 0.02).all() and current_target_idx < len(target_positions) - 1) or (abs(diff) < allowance).all():
            # print(f"Reached target position {current_target_idx} in {i} steps")
            current_target_idx += 1

    # import time; time.sleep(2)
    # _, current_pos = robot_interface.last_eef_rot_and_pos
    # diff = (target_pos - current_pos).squeeze(1)
    # print("After sleep diff is: ", diff)
    if (abs(diff) > allowance).any():
        print(f'robot might be under-actuating, given error {diff} larger than {allowance}')
        print(f"Robot target position was: {target_pos.squeeze()}")
        print(f"Robot final position was: {current_pos.squeeze()}")

    return eef_pos_history, action_history




def position_only_gripper_move_to_waypoints_keep_yaw(
    robot_interface, target_positions, num_steps=100, controller_cfg: dict = None, grasp=False, max_speed=None, allowance=0.02, min_speed=0.1,
        target_yaw=None,
):
    """_summary_

    Args:
        robot_interface (FrankaInterface): the python interface for robot control
        target_pos (np.array or list): target xyz location
        num_steps (int, optional): number of steps to control. Defaults to 100.
        controller_cfg (dict, optional): controller configurations. Defaults to None.
        grasp (bool, optional): close the gripper if set to True. Defaults to False.
    Return:
        eef_pos_history (list): a list of recorded end effector positions
        action_history (list): a list of recorded action commands
    """
    if max_speed is None:
        max_speed = 0.4
    assert max_speed < 0.6, f'max_speed = {max_speed} is too high, should be less than 0.6'

    if controller_cfg is None:
        controller_cfg = get_default_controller_config(controller_type="OSC_YAW")
    else:
        assert controller_cfg["controller_type"] == "OSC_YAW", (
            "This function is only for OSC_POSITION mode. You specified "
            + controller_cfg["controller_type"]
        )
        controller_cfg = verify_controller_config(controller_cfg)
    eef_pos_history = []
    action_history = []

    current_pos = None
    while current_pos is None:
        _, current_pos = robot_interface.last_eef_rot_and_pos

    prev_action = np.array([0.0] * 6 + [int(grasp) * 2 - 1])
    windup = 0.15
    current_target_idx = 0

    for i in range(num_steps):
        if current_target_idx >= len(target_positions):
            break
        target_pos = target_positions[current_target_idx]
        _, current_pos = robot_interface.last_eef_rot_and_pos
        action = np.array([0.0] * 6 + [int(grasp) * 2 - 1])
        action[:3] = ((target_pos - current_pos).flatten()) * 10
        if target_yaw is not None:
            action[5] = (target_yaw - mat2euler(robot_interface.last_eef_rot_and_pos[0])[2]) * 20
        # bound mean and max value to avoid drastic instant motion or under-actuation
        # if it's not the last target position yet, we want to each action to have absolute value least at min_speed m/s
        if current_target_idx < len(target_positions) - 1:
            # if actions are too small, we want to increase the speed
            if np.linalg.norm(action[:3]) < min_speed:
                action[:3] = action[:3] / np.linalg.norm(action[:3]) * min_speed

        action[:3] = np.clip(action[:3], -max_speed, max_speed)
        # if np.all(np.abs(action[:3]) < 0.01):
        #     action[:3] = 0.0
        # elif np.all(np.abs(action[:3]) < 0.05):
        #     action[:3] = action[:3] / np.linalg.norm(action[:3]) * 0.05

        # if the eef_pos_history shows that the robot hasn't moved much in the last 5 steps, increase the speed
        if len(eef_pos_history) > 5:
            last_5_steps = np.array(eef_pos_history[-5:])
            if np.all(np.abs(last_5_steps - last_5_steps[0]) < 0.005):
                # print("robot hasn't moved much in the last 5 steps, increasing speed")
                windup *= 1.03
                action[:3] = action[:3] / np.linalg.norm(action[:3]) * windup
            else:
                windup = 0.15
        robot_interface.control(
            controller_type="OSC_YAW", action=action, controller_cfg=controller_cfg
        )
        eef_pos_history.append(current_pos.flatten().tolist())
        action_history.append(prev_action)
        prev_action = np.array(action)

        # adding check
        _, current_pos = robot_interface.last_eef_rot_and_pos
        diff = (target_pos - current_pos).squeeze(1)
        # print("Before sleep diff is: ", diff)

        if ((abs(diff) < 0.02).all() and current_target_idx < len(target_positions) - 1) or (abs(diff) < allowance).all():
            # print(f"Reached target position {current_target_idx} in {i} steps")
            current_target_idx += 1

    # import time; time.sleep(2)
    # _, current_pos = robot_interface.last_eef_rot_and_pos
    # diff = (target_pos - current_pos).squeeze(1)
    # print("After sleep diff is: ", diff)
    if (abs(diff) > allowance).any():
        print(f'robot might be under-actuating, given error {diff} larger than {allowance}')
        print(f"Robot target position was: {target_pos.squeeze()}")
        print(f"Robot final position was: {current_pos.squeeze()}")

    return eef_pos_history, action_history


def position_only_gripper_move_by(
    robot_interface, delta_pos, num_steps=100, controller_cfg: dict = None, grasp=True, allowance=0.02
):
    """_summary_

    Args:
        robot_interface (FrankaInterface): the python interface for robot control
        target_pos (np.array or list): target xyz location
        num_steps (int, optional): number of steps to control. Defaults to 100.
        controller_cfg (dict, optional): controller configurations. Defaults to None.
        grasp (bool, optional): close the gripper if set to True. Defaults to False.
    Return:
        eef_pos_history (list): a list of recorded end effector positions
        action_history (list): a list of recorded action commands
    """
    current_pos = None
    while current_pos is None:
        _, current_pos = robot_interface.last_eef_rot_and_pos

    delta_pos = np.array(delta_pos).reshape(3, 1)
    assert delta_pos.shape == current_pos.shape
    target_pos = current_pos + delta_pos
    return position_only_gripper_move_to(
        robot_interface,
        target_pos,
        num_steps=num_steps,
        controller_cfg=controller_cfg,
        grasp=grasp,
        allowance=allowance
    )


def close_gripper(
        robot_interface,
        amount=None
):
    """
    Closes the gripper
    """
    # while robot_interface.last_gripper_q > 0.05:
    while robot_interface.last_gripper_q > 0.04:

        print(f'closing the gripper, curr gap = {robot_interface.last_gripper_q}')
        # robot_interface.gripper_control(-0.1)
        if amount:
            robot_interface.gripper_control(amount)
        else:
            robot_interface.gripper_control(-0.3)

        time.sleep(0.5)


def open_gripper(
        robot_interface
):
    """
    Opens the gripper
    """
    while robot_interface.last_gripper_q < 0.04:
        print(f'opening the gripper, curr gap = {robot_interface.last_gripper_q}')
        robot_interface.gripper_control(-1.5)
        time.sleep(0.1)

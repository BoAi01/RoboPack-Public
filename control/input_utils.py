import logging
import numpy as np


logger = logging.getLogger(__name__)


grasping = False
last_button_state = False
last_dpos = None

def set_initial_grasping(grasp):
    global grasping
    grasping = grasp

def input2action(device, controller_type="OSC_POSE", robot_name="Panda", gripper_dof=1, close_gripper_pos=1):
    state = device.get_controller_state()
    # Note: Devices output rotation with x and z flipped to account for robots starting with gripper facing down
    #       Also note that the outputted rotation is an absolute rotation, while outputted dpos is delta pos
    #       Raw delta rotations from neutral user input is captured in raw_drotation (roll, pitch, yaw)
    dpos, rotation, raw_drotation, grasp_button_state, reset = (
        state["dpos"],
        state["rotation"],
        state["raw_drotation"],
        state["grasp"],
        state["reset"],
    )
    # grasp_button_state is 1 when button pressed down, or 0 otherwise

    drotation = raw_drotation[[1, 0, 2]]

    action = None
    global grasping, last_button_state

    if not reset:
        if controller_type == "OSC_POSE":
            drotation[2] = -drotation[2]
            drotation *= 75
            dpos *= 200
            drotation = drotation

            grasp_button_state = 1 if grasp_button_state else -1
            action = np.concatenate([dpos, drotation, [grasp_button_state] * gripper_dof])

        if controller_type == "OSC_YAW":
            drotation[2] = -drotation[2]
            drotation *= 75
            dpos *= 200
            dpos = np.clip(dpos, -0.3, 0.3)

            # print(f'last_grasp_command = {last_button_state}, grasp = {grasp_button_state}')
            if bool(grasp_button_state) is True and last_button_state is False:         # button pressed down
                grasping = not grasping
            grasp = close_gripper_pos if grasping else -1
            last_button_state = bool(grasp_button_state)
            action = np.concatenate([dpos, drotation, [grasp_button_state] * gripper_dof])

            # drotation = T.quat2axisangle(T.mat2quat(T.euler2mat(drotation)))
        if controller_type == "OSC_POSITION":
            drotation[:] = 0
            dpos *= 200 * 2     # assume the magnitude has been bounded
            dpos = np.clip(dpos, -0.3, 0.3)
            # TODO: smooth out the actions over time to reduce vibration

            # print(f'last_grasp_command = {last_button_state}, grasp = {grasp_button_state}')
            if bool(grasp_button_state) is True and last_button_state is False:         # button pressed down
                grasping = not grasping
            grasp = close_gripper_pos if grasping else -1
            last_button_state = bool(grasp_button_state)

            # last_dpos = dpos
            # if reset:
            #     # if bottom down, fix dpos while setting to zero
            #     dpos = last_dpos
            #     dpos[0] = 0

            action = np.concatenate([dpos, drotation, [grasp] * gripper_dof])
            # drotation = T.quat2axisangle(T.mat2quat(T.euler2mat(drotation)))

        if controller_type == "JOINT_IMPEDANCE":
            grasp_button_state = 1 if grasp_button_state else -1
            action = np.array([0.0] * 7 + [grasp_button_state] * gripper_dof)

    return action, grasp_button_state

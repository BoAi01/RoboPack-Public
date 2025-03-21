import time
from deoxys import config_root
from deoxys.franka_interface import FrankaInterface


def init_franka_interface(interface_cfg="charmander.yml"):
    robot_interface = FrankaInterface(
        config_root + f"/{interface_cfg}", use_visualizer=False
    )
    time.sleep(3)  # robot interface initialization takes time
    assert robot_interface.last_q is not None, \
        f"robot state not available, check: connected to NUC, FCI activated, port available, emergency stop not " \
        f"pressed down"

    return robot_interface


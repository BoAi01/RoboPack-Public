import os

from pathlib import Path

# Directories
### Level 1
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = os.path.join(ROOT_DIR, "data")
RESULT_DIR = os.path.join(ROOT_DIR, "results")
SCRIPT_DIR = os.path.join(ROOT_DIR, "scripts")
SRC_DIR = os.path.join(ROOT_DIR, "src")

### Level 2
CONFIG_DIR = os.path.join(SRC_DIR, "config")
DYNAMICS_DIR = os.path.join(SRC_DIR, "dynamics")
SIM_DIR = os.path.join(SRC_DIR, "sim")
GEOMETRY_DIR = os.path.join(SRC_DIR, "geometry")
PERCEPTION_DIR = os.path.join(SRC_DIR, "perception")
PLANNING_DIR = os.path.join(SRC_DIR, "planning")

# File paths
DEF_PATH = os.path.join(ROOT_DIR, "definitions.py")

# Global parameters
### Perception
N_CAM = 4
DEPTH_OPTICAL_FRAME_POSE = [0, 0, 0, 0.5, -0.5, 0.5, -0.5]
EE_FINGERTIP_T = [
    [0.707, 0.707, 0, 0],
    [-0.707, 0.707, 0, 0],
    [0, 0, 1, 0.1034],
    [0, 0, 0, 1],
]

### Tools
TOOL_NAME_GEOM_DICT = {
    "two_rod_sym_gripper": [
        "finger_l",
        "finger_r",
    ]
}

TOOL_NAME_DIM_DICT = {"two_rod_sym_gripper": [92, 92]}

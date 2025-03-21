import numpy
import open3d
import numpy as np
import open3d as o3d
import os 

from perception.utils_gripper import get_bubble1_to_finger1_transform
from utils_general import read_extrinsics

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

xrange = np.array([0.25, 0.8])
yrange = np.array([-0.4, 0.4])
zrange = np.array([0.01, 0.45])
N_POINTS_PER_OBJ = 20
N_POINTS_BUBBLE = 50
bubble1_to_finger1 = get_bubble1_to_finger1_transform()
path = os.path.join(project_root, 'config/sensor/4cameras_pose_robot_v11.yml')
extrinsics = read_extrinsics(path)
cam_names = ['cam_0', 'cam_1', 'cam_2', 'cam_3']

# intrinsics of the depth stream
intrinsics_depth = {'cam_0': o3d.camera.PinholeCameraIntrinsic(1280, 720, 903.0823364257812, 903.0823364257812,
                                                               648.4481811523438, 365.2071838378906),
                   'cam_1': o3d.camera.PinholeCameraIntrinsic(1280, 720, 908.16015625, 908.16015625,
                                                              650.9434204101562, 351.9290771484375),
                   'cam_2': o3d.camera.PinholeCameraIntrinsic(1280, 720, 894.4971313476562, 894.4971313476562,
                                                              651.0776977539062, 357.4739990234375),
                   'cam_3': o3d.camera.PinholeCameraIntrinsic(1280, 720, 943.1380615234375, 943.1380615234375,
                                                              630.468017578125, 360.4356689453125),
                   'bubble_1': o3d.camera.PinholeCameraIntrinsic(640, 480, 384.2249755859375, 384.2249755859375,
                                                                 307.74658203125, 245.0389404296875),
                   'bubble_2': o3d.camera.PinholeCameraIntrinsic(640, 480, 380.9357604980469, 380.9357604980469,
                                                                 316.4392395019531, 242.6063995361328)
                   }

intrinsics_color = {'cam_0': o3d.camera.PinholeCameraIntrinsic(1280, 720, 915.0818481445312, 914.4964599609375,
                                                                 634.1450805664062, 360.3082580566406),
                    'cam_1': o3d.camera.PinholeCameraIntrinsic(1280, 720, 912.4166259765625, 911.2178344726562,
                                                                 639.5074462890625, 362.4543151855469),
                    'cam_2': o3d.camera.PinholeCameraIntrinsic(1280, 720, 916.1026000976562, 914.8382568359375,
                                                                 641.5242919921875, 361.9787292480469),
                    'cam_3': o3d.camera.PinholeCameraIntrinsic(1280, 720, 925.927734375, 925.7191772460938,
                                                                 627.90283203125, 344.5506591796875),
                    'bubble_1': o3d.camera.PinholeCameraIntrinsic(640, 480, 429.7550964355469, 429.2806396484375,
                                                                  308.99658203125, 248.07232666015625),
                    'bubble_2': o3d.camera.PinholeCameraIntrinsic(640, 480, 431.0281066894531, 430.5440673828125,
                                                                  313.6537170410156, 242.69883728027344)
                    }

vendor_id = 9583
product_id = 50734
controller_type = 'OSC_POSITION'
controller_cfg_file = "osc-position-controller.yml"
intrinsic_matrices = {
    'cam_0': np.array([[903.0823364257812, 0, 648.4481811523438], [0, 903.0823364257812, 365.2071838378906], [0, 0, 1]]),
    'cam_1': np.array([[908.16015625, 0, 650.9434204101562], [0, 908.16015625, 351.9290771484375], [0, 0, 1]]),
    'cam_2': np.array([[894.4971313476562, 0.0, 651.0776977539062], [0.0, 894.4971313476562, 357.4739990234375], [0.0, 0.0, 1.0]]),
    'cam_3': np.array([[943.1380615234375, 0, 630.468017578125], [0, 943.1380615234375, 360.4356689453125], [0, 0, 1]]),
    'bubble_1': np.array([[384.2249755859375, 0, 307.74658203125], [0, 384.2249755859375, 245.0389404296875], [0, 0, 1]]),
    'bubble_2': np.array([[380.9357604980469, 0, 316.4392395019531], [0, 380.9357604980469, 242.6063995361328], [0, 0, 1]]),
}

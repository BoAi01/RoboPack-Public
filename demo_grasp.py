import threading
import time
from utils_deoxys import init_franka_interface

import rclpy
from cv_bridge import CvBridge
from deoxys.experimental.motion_utils import position_only_gripper_move_to, close_gripper, open_gripper

from perception.utils_sam import *
from perception.utils_cv import *
from perception.utils_pc import *
from perception.utils_ros import *
from utils_general import *

robot_interface = init_franka_interface()

# back to default pose
ee_home_pose = np.array([[0.5], [0], [0.4]])
object_home_pose = np.array([[0.5], [0.3], [0]])

position_only_gripper_move_to(
    robot_interface, ee_home_pose
)

# initialize ros
rclpy.init(args=None)
node = RealsenseToOpen3DNode()
ros_thread = threading.Thread(target=lambda: rclpy.spin(node))
ros_thread.start()
time.sleep(3)

pcs_global = node.last_reading
node.destroy_node()
rclpy.shutdown()
ros_thread.join()

# extract data
assert pcs_global is not None

bridge = CvBridge()
cam0_rgb = bridge.imgmsg_to_cv2(pcs_global[2], desired_encoding='passthrough')
cam1_rgb = bridge.imgmsg_to_cv2(pcs_global[3], desired_encoding='passthrough')
cam0_depth = bridge.imgmsg_to_cv2(pcs_global[4], desired_encoding='passthrough')  # mm as unit
cam1_depth = bridge.imgmsg_to_cv2(pcs_global[5], desired_encoding='passthrough')
print('image read.')

# ## preprocess and segment point cloud data
path = '/home/albert/github/robopack/config/sensor/3cameras_pose_robot_v4.yml'
extrinsics = read_extrinsics(path)
intrinsics = {'cam_0': o3d.camera.PinholeCameraIntrinsic(1280, 720, 903.0823364257812, 903.0823364257812,
                                                         648.4481811523438, 365.2071838378906),
              'cam_1': o3d.camera.PinholeCameraIntrinsic(1280, 720, 908.16015625, 908.16015625,
                                                         650.9434204101562, 351.9290771484375)}

# com0
model, predictor = load_models()
cam0_masks, boxes_filt, pred_phrases = predict(model, predictor, cam0_rgb, "one object on table")
mask_image_cam0 = get_mask_image(cam0_masks)
# plt.imshow(mask_image_cam0)

# com1
model, predictor = load_models()
cam1_masks, boxes_filt, pred_phrases = predict(model, predictor, cam1_rgb, "one object on table")
mask_image_cam1 = get_mask_image(cam1_masks)
# plt.imshow(mask_image_cam1)
print('mask obtained')

# cam0_rgb_masked = crop_out_masked_region(cam0_rgb, mask_image_cam0)
# cam1_rgb_masked = crop_out_masked_region(cam1_rgb, mask_image_cam1)

# get the point cloud of every object
cam0_masks = cam0_masks.cpu().numpy()[:, 0]  # remove the second dim
cam0_pcs = []
for mask in cam0_masks:
    # each mask is of shape (1, h, w)
    masked_rgb = crop_out_masked_region(cam0_rgb, mask)
    masked_depth = crop_out_masked_region(cam0_depth, mask)
    constructed_pc = construct_pointcloud_from_rgbd(masked_rgb, masked_depth, intrinsics['cam_0'])
    constructed_pc = remove_points_by_color(constructed_pc)
    constructed_pc = project_pc(convert_pc_color_optical_to_link_frame(constructed_pc), extrinsics['cam_0'])
    constructed_pc = extract_workspace_pc(constructed_pc)
    if constructed_pc.has_points():
        cam0_pcs.append(constructed_pc)

# o3d.visualization.draw_geometries(cam0_pcs)

# get the point cloud of every object
cam1_masks = cam1_masks.cpu().numpy()[:, 0]  # remove the second dim
cam1_pcs = []
for mask in cam1_masks:
    # each mask is of shape (1, h, w)
    masked_rgb = crop_out_masked_region(cam1_rgb, mask)
    masked_depth = crop_out_masked_region(cam1_depth, mask)
    constructed_pc = construct_pointcloud_from_rgbd(masked_rgb, masked_depth, intrinsics['cam_1'])
    constructed_pc = remove_points_by_color(constructed_pc)
    constructed_pc = project_pc(convert_pc_color_optical_to_link_frame(constructed_pc), extrinsics['cam_1'])
    constructed_pc = extract_workspace_pc(constructed_pc)
    if constructed_pc.has_points():
        cam1_pcs.append(constructed_pc)

# o3d.visualization.draw_geometries(cam1_pcs)

# match and merge the point clouds for each object
# need to take note that the num of clouds may differ for the cameras

cam0_centers = [pc.get_center() for pc in cam0_pcs]
cam1_centers = [pc.get_center() for pc in cam1_pcs]

pairs, unmatched_set1, unmatched_set2 = pair_points_minimize_distance(cam0_centers, cam1_centers)

# merge point clouds by object 
object_pcs = []
for i, j in pairs:
    object_pcs.append(cam0_pcs[i] + cam1_pcs[j])

for i in unmatched_set1:
    object_pcs.append(cam0_pcs[i])

for j in unmatched_set2:
    object_pcs.append(cam1_pcs[j])


# o3d.visualization.draw_geometries([*object_pcs])
# o3d.visualization.draw_geometries([object_pcs[0]])


print('point clouds from cameras merged')

min_cloud = object_pcs[1]
for pc in object_pcs:
    if np.asarray(pc.points).shape[0] < np.asarray(min_cloud.points).shape[0]:
        min_cloud = pc

# In[34]:

# denoised_sampled_pcs = [sample_points_from_pc(pc) for pc in object_pcs]
# o3d.visualization.draw_geometries([min_cloud])
min_cloud += fill_point_cloud_by_downward_projection(min_cloud)
min_cloud = denoise_pc_by_stats(min_cloud, denoise_depth=2)
o3d.visualization.draw_geometries([min_cloud])
denoised_sampled_pc = sample_points_from_pc(min_cloud, ratio=3e7, mesh_fix=True)
o3d.visualization.draw_geometries([denoised_sampled_pc])

# In[37]:

center = denoised_sampled_pc.get_center()
print(f'estimated object center is {center}')

# gripper_z_min = 0.17
# if center[-1] < gripper_z_min:
#     print(f'gripper z should be {center[-1]}, but clipped to {gripper_z_min} due to hardware constraint')
#     center[-1] = gripper_z_min

grasp_pose = np.expand_dims(get_grasp_pos(center), 1)
pre_grasp_pose = grasp_pose.copy()
pre_grasp_pose[-1] = 0.4

position_only_gripper_move_to(
    robot_interface, pre_grasp_pose
)

position_only_gripper_move_to(
    robot_interface, grasp_pose
)
# close_gripper(robot_interface)


# In[41]:

time.sleep(3)
while robot_interface.last_gripper_q > 0.05:
    print(f'closing the gripper, curr gap = {robot_interface.last_gripper_q}')
    close_gripper(robot_interface)
    time.sleep(3)

second_pos = grasp_pose.copy()
lift_height = 0.2 - 0.03
second_pos[-1, 0] = second_pos[-1, 0] + lift_height
second_pos[1, 0] += 0.03  # offset under-actuation along y axis
position_only_gripper_move_to(
    robot_interface, second_pos, grasp=True
)

user_input = None
while user_input != 'c':
    user_input = input("Enter c to continue: ")

# In[43]:
third_pos = object_home_pose.copy()
third_pos[2] = second_pos[2]
position_only_gripper_move_to(
    robot_interface, third_pos, grasp=True
)

# In[44]:

fourth_pose = object_home_pose
fourth_pose[-1] = lift_height
position_only_gripper_move_to(
    robot_interface, fourth_pose, grasp=True
)

# In[45]:

time.sleep(3)
while robot_interface.last_gripper_q < 0.05:
    print(f'releasing the gripper, curr gap = {robot_interface.last_gripper_q}')
    open_gripper(robot_interface)
    time.sleep(3)

# In[46]:

position_only_gripper_move_to(
    robot_interface, third_pos, grasp=False
)

position_only_gripper_move_to(
    robot_interface, ee_home_pose, grasp=False
)

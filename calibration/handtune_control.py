"""
Run under native ROS 1 environment.
"""

import copy
import os
import sys

import readchar
import rospy
import tf
import tf2_ros
import yaml
from geometry_msgs.msg import TransformStamped
from transforms3d.quaternions import *

fixed_frame = 'panda_link0'
num_cams = 4
tune_idx = [0]  # set the index of the camera to tune

cd = os.path.dirname(os.path.realpath(sys.argv[0]))
data_dir = '/home/albert/github/robopack/config/sensor'
with open(os.path.join(data_dir, '4cameras_pose_robot_v8.yml'), 'r') as f:
    camera_pose_dict = yaml.load(f, Loader=yaml.FullLoader)
path_to_save = os.path.join(data_dir, '4cameras_pose_robot_v8.yml')

# step size for adjusting
pos_stride = 0.0005
rot_stride = 0.001


def pos_quat_to_matrix(pos, quat):
    assert len(pos) == 3, 'position should be xyz'
    rot = quat2mat(quat)
    pos = np.expand_dims(pos, 1)
    matrix = np.concatenate((np.concatenate((rot, pos), axis=1), [[0, 0, 0, 1]]))
    return matrix


def main():
    rospy.init_node('cam_pose_tuner', anonymous=True)

    static_br = tf2_ros.StaticTransformBroadcaster()
    static_ts_list = []
    for i in list(range(num_cams)):
        static_ts = TransformStamped()
        static_ts.header.stamp = rospy.Time.now()
        static_ts.header.frame_id = fixed_frame
        static_ts.child_frame_id = f"cam_{i}_link"

        static_ts.transform.translation.x = camera_pose_dict[f"cam_{i}"]["position"][0]
        static_ts.transform.translation.y = camera_pose_dict[f"cam_{i}"]["position"][1]
        static_ts.transform.translation.z = camera_pose_dict[f"cam_{i}"]["position"][2]

        static_ts.transform.rotation.x = camera_pose_dict[f"cam_{i}"]["orientation"][1]
        static_ts.transform.rotation.y = camera_pose_dict[f"cam_{i}"]["orientation"][2]
        static_ts.transform.rotation.z = camera_pose_dict[f"cam_{i}"]["orientation"][3]
        static_ts.transform.rotation.w = camera_pose_dict[f"cam_{i}"]["orientation"][0]

        static_ts_list.append(static_ts)

    static_br.sendTransform(static_ts_list)

    pcd_trans_vec = [0.0, 0.0, 0.0]
    pcd_rot_vec = [0.0, 0.0, 0.0]

    camera_pose_dict_new = copy.deepcopy(camera_pose_dict)

    br = tf.TransformBroadcaster()
    rate = rospy.Rate(30)

    # record initial relative transformation
    ref_cam_mat = None
    cam2cam_relative_mats = {}
    for cam_idx in tune_idx:
        ref_cam_init_mat = pos_quat_to_matrix(camera_pose_dict[f"cam_{tune_idx[0]}"]["position"],
                                              camera_pose_dict[f"cam_{tune_idx[0]}"]["orientation"])
        cam_init_mat = pos_quat_to_matrix(camera_pose_dict[f"cam_{cam_idx}"]["position"],
                                          camera_pose_dict[f"cam_{cam_idx}"]["orientation"])
        cam_to_ref = np.linalg.inv(ref_cam_init_mat) @ cam_init_mat
        cam2cam_relative_mats[cam_idx] = cam_to_ref
        # print(ref_cam_init_mat @ cam_to_ref, cam_init_mat)
        #
        # ref_cam_mat = ref_cam_init_mat
        # cam_mat = ref_cam_mat @ cam2cam_relative_mats[cam_idx]
        # cam_ori_cur = mat2quat(cam_mat[:3, :3])
        # cam_pos_cur = cam_mat[:3, -1].tolist()
        #
        # print(cam_ori_cur, camera_pose_dict[f"cam_{tune_idx[cam_idx]}"]["position"])
        # print(cam_ori_cur, camera_pose_dict[f"cam_{tune_idx[cam_idx]}"]["orientation"])
        # print(np.linalg.inv(cam_init_mat) @ ref_cam_init_mat @ cam_to_ref == cam_init_mat)

    #
    # import pdb
    # pdb.set_trace()

    save = False
    while not rospy.is_shutdown():
        key = readchar.readkey()
        if key == 'w':
            pcd_trans_vec[0] += pos_stride
        elif key == 'x':
            pcd_trans_vec[0] -= pos_stride
        elif key == 'a':  # z axis translation
            pcd_trans_vec[1] += pos_stride
        elif key == 'd':
            pcd_trans_vec[1] -= pos_stride
        elif key == 'q':
            pcd_trans_vec[2] += pos_stride
        elif key == 'z':
            pcd_trans_vec[2] -= pos_stride
        elif key == '1':
            pcd_rot_vec[0] += rot_stride
        elif key == '2':
            pcd_rot_vec[0] -= rot_stride
        elif key == '3':
            pcd_rot_vec[1] += rot_stride
        elif key == '4':
            pcd_rot_vec[1] -= rot_stride
        elif key == '5':
            pcd_rot_vec[2] += rot_stride
        elif key == '6':
            pcd_rot_vec[2] -= rot_stride
        elif key == 'm':
            with open(path_to_save, 'w') as f:
                yaml.dump(camera_pose_dict_new, f)
            print(f'saved to path: {path_to_save}')
        elif key == 'b':
            break

        pcd_ori_world = qmult(qmult(qmult(axangle2quat([1, 0, 0], pcd_rot_vec[0]),
                                          axangle2quat([0, 1, 0], pcd_rot_vec[1])),
                                    axangle2quat([0, 0, 1], pcd_rot_vec[2])),
                              [1.0, 0.0, 0.0, 0.0])

        # for each camera, compute the updated pose
        for i, cam_idx in enumerate(tune_idx):
            if i == 0:
                # this is the orientation of reference camera
                cam_pos_init = camera_pose_dict[f"cam_{cam_idx}"]["position"]
                cam_ori_init = camera_pose_dict[f"cam_{cam_idx}"]["orientation"]

                cam_pos_cur = np.array(cam_pos_init) + np.array(pcd_trans_vec)
                cam_pos_cur = [float(x) for x in cam_pos_cur]

                cam_ori_cur = qmult(pcd_ori_world, cam_ori_init)
                cam_ori_cur = [float(x) for x in cam_ori_cur]
                print(f"{cam_idx}: Pos: {cam_pos_cur}\nOri: {cam_ori_cur}")
                ref_cam_mat = pos_quat_to_matrix(cam_pos_cur, cam_ori_cur)
            else:
                # other cameras follow
                cam_mat = ref_cam_mat @ cam2cam_relative_mats[cam_idx]
                cam_ori_cur = mat2quat(cam_mat[:3, :3])
                cam_pos_cur = cam_mat[:3, -1].tolist()

                cam_ori_cur = [float(x) for x in cam_ori_cur]
                cam_pos_cur = [float(x) for x in cam_pos_cur]
                print(f"{cam_idx} follows {tune_idx[0]}: Pos: {cam_pos_cur}\nOri: {cam_ori_cur}")

            # broadcast transformations
            br.sendTransform(tuple(cam_pos_cur),
                             tuple([cam_ori_cur[1], cam_ori_cur[2], cam_ori_cur[3], cam_ori_cur[0]]),
                             rospy.Time.now(), f"cam_{cam_idx}_link", fixed_frame)

            camera_pose_dict_new[f"cam_{cam_idx}"]["position"] = cam_pos_cur
            camera_pose_dict_new[f"cam_{cam_idx}"]["orientation"] = cam_ori_cur
            camera_pose_dict_new[f"cam_{cam_idx}"]['transformation'] = \
                pos_quat_to_matrix(cam_pos_cur, cam_ori_cur).tolist()

        rate.sleep()


if __name__ == '__main__':
    main()

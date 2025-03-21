import collections
import copy
import glob
import inspect
import json
import logging
import os
import pdb
from pathlib import Path

import cv2 as cv
import numpy as np
import open3d as o3d
import rosbag
from pysdf import SDF
from tqdm import tqdm
from transforms3d.quaternions import *

from src.config.macros import *
from src.perception.motion_field import *
from src.perception.utils_pc import *
from src.utils.utils import *
from src.utils.visualizer import *


def read_ros_bag(ros_bag_path):
    try:
        bag = rosbag.Bag(ros_bag_path)
    except rosbag.bag.ROSBagUnindexedException:
        os.system(f"rosbag reindex {ros_bag_path}")
        bag = rosbag.Bag(ros_bag_path)

    pc_msgs = []
    ee_pos = None
    ee_quat = None
    finger_width = None
    for topic, msg, t in bag.read_messages():
        if "/depth/color/points" in topic:
            pc_msgs.append(msg)

        if topic == "/ee_pose":
            ee_pos = np.array([msg.position.x, msg.position.y, msg.position.z])
            ee_quat = np.array(
                [
                    msg.orientation.w,
                    msg.orientation.x,
                    msg.orientation.y,
                    msg.orientation.z,
                ]
            )

        if topic == "/gripper_width":
            finger_width = msg.data

    bag.close()

    return pc_msgs, ee_pos, ee_quat, finger_width


def color_segmentation(pc, hsv_min, hsv_max):
    pc_rgb = np.asarray(pc.colors, dtype=np.float32)
    pc_hsv = cv.cvtColor(pc_rgb[None], cv.COLOR_RGB2HSV)
    mask = cv.inRange(pc_hsv, hsv_min, hsv_max)
    object_label = np.where(mask[0] == 255)

    object_pc = pc.select_by_index(object_label[0])
    rest_pc = pc.select_by_index(object_label[0], invert=True)

    return object_pc, rest_pc


def test_if_close(cube, tool_repr_list):
    cube_hull, _ = cube.compute_convex_hull()
    f = SDF(cube_hull.vertices, cube_hull.triangles)

    for tool_repr in tool_repr_list:
        sdf = f(tool_repr)
        n_points_close = np.sum(sdf > 0)
        if n_points_close > 0:
            return 1

    return 0


class Sampler(object):
    def __init__(self, config, ros_bag_root):
        self.config = config
        self.surface_reconstuction = SurfaceReconstruction()

        self.ros_bag_root = ros_bag_root
        with open(os.path.join(ros_bag_root, "cam_pose.json"), "r") as f:
            self.cam_pose_dict = json.load(f)

        self.tool_geom_list = self.load_tool()

    # @profile
    def sample_video(self, save_dir):
        start_epi_idx, start_seq_idx = [
            int(s) for s in self.config["start_idx"].split("-")
        ]
        subprocess_list = []
        for i in range(self.config["n_videos"]):
            video_idx = i + start_epi_idx * self.config["epi_len"] + start_seq_idx
            video_dir = os.path.join(save_dir, str(video_idx).zfill(3))
            log_dir = os.path.join(video_dir, "log")
            Path(log_dir).mkdir(parents=True, exist_ok=True)

            ros_bag_path = os.path.join(
                self.ros_bag_root,
                f"ep_{str(video_idx // self.config['epi_len']).zfill(3)}",
                f"seq_{str(video_idx % self.config['epi_len']).zfill(3)}",
            )

            ros_bag_list = sorted(
                glob.glob(os.path.join(ros_bag_path, "*.bag")),
                key=lambda x: float(os.path.basename(x)[:-4]),
            )[1:]

            state_seq = []
            state_seq_dense_list = [
                [] for i in range(self.config["motion_field"]["n_samples_vis"])
            ]

            info_dict_prev = {
                "sparse": None,
                "tool": None,
                "train": None,
                "back": 0,
                "ckpt_path": "",
                "dist": float("inf"),
            }

            for j in tqdm(
                # range(29, 32),
                range(len(ros_bag_list)),
                desc=f"Video {video_idx}",
            ):
                logging.getLogger("csv").info(f"{i},{j},")
                info_dict = self.sample_frame(
                    ros_bag_list[j],
                    info_dict_prev,
                    log_dir,
                )
                if len(info_dict["ckpt_path"]) == 0:
                    logging.getLogger("csv").info(f"{0},{0}")

                state = collections.OrderedDict(
                    object=info_dict["sparse"],
                    tool=info_dict["tool"],
                )
                state_seq.append(state)

                if info_dict["train"] is not None:
                    for k in range(self.config["motion_field"]["n_samples_vis"]):
                        state_dense = collections.OrderedDict(
                            object=info_dict["train"][k],
                            tool=info_dict["tool"],
                        )
                        state_seq_dense_list[k].append(state_dense)

                store_h5_data(
                    state,
                    os.path.join(video_dir, str(j).zfill(3) + ".h5"),
                )

                info_dict_prev = info_dict

            if len(state_seq_dense_list) > 0:
                p = render_anim_async(
                    path=os.path.join(video_dir, "repr"),
                    config=self.config["visualizer"],
                    title_list=[
                        "Sandwich",
                        *[
                            f"Supervision {i}"
                            for i in range(self.config["motion_field"]["n_samples_vis"])
                        ],
                    ],
                    state_seq_list=[state_seq, *state_seq_dense_list],
                )
            else:
                p = render_anim_async(
                    path=os.path.join(video_dir, "repr"),
                    config=self.config["visualizer"],
                    title_list=["RoboCraft"],
                    state_seq_list=[state_seq],
                )

            subprocess_list.append(p)

        for p in subprocess_list:
            p.communicate()

    # @profile
    def sample_frame(self, ros_bag_path, info_dict_prev, log_dir):
        info_dict = copy.deepcopy(info_dict_prev)
        pc_msgs, ee_pos, ee_quat, finger_width = read_ros_bag(ros_bag_path)

        tool_geom_list = self.transform_tool(ee_pos, ee_quat, finger_width)
        tool_repr_list = [tool_geom_dict["repr"] for tool_geom_dict in tool_geom_list]
        info_dict["tool"] = np.concatenate(tool_repr_list)

        pc = self.merge(pc_msgs)

        info_dict["dist"] = finger_width / 2
        if (
            not info_dict_prev["back"]
            and info_dict["dist"] > info_dict_prev["dist"] + 0.001
        ):
            info_dict["back"] = 1
            print("Start moving back...")
        if (
            info_dict_prev["back"]
            and info_dict["dist"] < info_dict_prev["dist"] - 0.001
        ):
            info_dict["back"] = 0
            print("End moving back...")

        if info_dict_prev["sparse"] is not None and info_dict_prev["back"]:
            logging.getLogger("stdout").info(
                "reuse the previous frame if not the first frame and moving back"
            )
            return info_dict

        dough = self.color_segmentation(pc)
        dough_denoise = self.denoise(dough)

        is_close = test_if_close(dough_denoise, tool_repr_list)

        if info_dict_prev["sparse"] is not None and not is_close:
            logging.getLogger("stdout").info(
                "reuse the previous frame if not the first frame and not close"
            )
            return info_dict

        sampled_dough = self.sample_in_mesh(dough_denoise)
        dough_inliers = self.filter_by_tool(sampled_dough, tool_geom_list)
        pc_dense = self.denoise(dough_inliers)

        if self.config["use_surface_sampling"]:
            pc_sparse = self.sample_surface(pc_dense, self.config["n_points"])
        else:
            pc_sparse = self.sample_full(pc_dense, self.config["n_points"])

        info_dict["sparse"] = np.asarray(pc_sparse.points, dtype=np.float32)

        if self.config["use_motion_field"]:
            n_points = self.config["motion_field"]["n_points_dense"]
            info_dict["train"] = []
            for i in range(self.config["motion_field"]["n_samples"]):
                if self.config["use_surface_sampling"]:
                    pc_train = self.sample_surface(pc_dense, n_points, poisson=False)
                else:
                    pc_train = self.sample_full(pc_dense, n_points)

                info_dict["train"].append(np.asarray(pc_train.points, dtype=np.float32))

            if info_dict_prev["train"] is not None:
                info_dict["sparse"], info_dict["ckpt_path"] = predict_next(
                    self.config["motion_field"],
                    log_dir,
                    info_dict_prev,
                    info_dict,
                )

        return info_dict

    def load_tool(self):
        tool_name_list = TOOL_NAME_GEOM_DICT[self.config["tool_name"]]
        tool_geom_list = []
        for i in range(len(tool_name_list)):
            tool_mesh = o3d.io.read_triangle_mesh(
                os.path.join(
                    GEOMETRY_DIR,
                    self.config["tool_name"],
                    f"{tool_name_list[i]}_repr.stl",
                )
            )

            tool_repr_path = os.path.join(
                GEOMETRY_DIR,
                self.config["tool_name"],
                f"{tool_name_list[i]}_points.npy",
            )
            if os.path.exists(tool_repr_path):
                tool_repr = np.load(
                    tool_repr_path,
                    allow_pickle=True,
                ).astype(np.float32)
            else:
                tool_pc_dense = o3d.geometry.TriangleMesh.sample_points_uniformly(
                    tool_mesh, 100000
                )

                tool_pc = tool_pc_dense.voxel_down_sample(voxel_size=0.007)
                tool_repr = np.asarray(tool_pc.points)
                with open(tool_repr_path, "wb") as f:
                    np.save(f, tool_repr)

            tool_geom_list.append({"repr": tool_repr, "mesh": tool_mesh})

        return tool_geom_list

    def transform_tool(self, ee_pos, ee_quat, finger_width):
        ee_T = np.concatenate(
            (
                np.concatenate((quat2mat(ee_quat), np.array([ee_pos]).T), axis=1),
                [[0, 0, 0, 1]],
            ),
            axis=0,
        )

        finger_center_T = ee_T @ np.array(EE_FINGERTIP_T)

        tool_geom_list_T = copy.deepcopy(self.tool_geom_list)
        for i in range(len(tool_geom_list_T)):
            finger_T = finger_center_T @ np.concatenate(
                (
                    np.concatenate(
                        (
                            np.eye(3),
                            np.array([[0, (2 * i - 1) * (finger_width / 2), 0]]).T,
                        ),
                        axis=1,
                    ),
                    [[0, 0, 0, 1]],
                ),
                axis=0,
            )

            tool_geom_list_T[i]["mesh"] = tool_geom_list_T[i]["mesh"].transform(
                finger_T
            )
            tool_geom_list_T[i]["repr"] = (
                finger_T
                @ np.concatenate(
                    [
                        tool_geom_list_T[i]["repr"].T,
                        np.ones((1, tool_geom_list_T[i]["repr"].shape[0])),
                    ],
                    axis=0,
                )
            ).T[:, :3]

        return tool_geom_list_T

    def merge(self, pc_msgs):
        pc = o3d.geometry.PointCloud()
        for i in range(len(pc_msgs)):
            cloud_xyz, cloud_rgb = parse_pointcloud2(pc_msgs[i])
            points = (
                quat2mat(DEPTH_OPTICAL_FRAME_POSE[3:]) @ cloud_xyz[:, :3].T
            ).T + DEPTH_OPTICAL_FRAME_POSE[:3]
            cam_pos = self.cam_pose_dict[f"cam_{i+1}"]["position"]
            cam_ori = self.cam_pose_dict[f"cam_{i+1}"]["orientation"]
            points = (quat2mat(cam_ori) @ points.T).T + cam_pos

            points_T = points.T
            x_filter = (
                points_T[0]
                > self.config["crop_center"][0] - self.config["crop_scale"][0]
            ) & (
                points_T[0]
                < self.config["crop_center"][0] + self.config["crop_scale"][0]
            )
            y_filter = (
                points_T[1]
                > self.config["crop_center"][1] - self.config["crop_scale"][1]
            ) & (
                points_T[1]
                < self.config["crop_center"][1] + self.config["crop_scale"][1]
            )
            z_filter = (
                points_T[2]
                > self.config["crop_center"][2] - self.config["crop_scale"][2]
            ) & (
                points_T[2]
                < self.config["crop_center"][2] + self.config["crop_scale"][2]
            )

            points = points[x_filter & y_filter & z_filter]
            cloud_rgb = cloud_rgb[x_filter & y_filter & z_filter]

            pc_crop = o3d.geometry.PointCloud()
            pc_crop.points = o3d.utility.Vector3dVector(points)
            pc_crop.colors = o3d.utility.Vector3dVector(cloud_rgb)

            pc += pc_crop

        if self.config["visualize_flag"][inspect.currentframe().f_code.co_name]:
            visualize_o3d([pc], title="merged_and_cropped_point_cloud")

        return pc

    def color_segmentation(self, pc):
        dough_hsv_min = np.array(self.config["dough_hsv_min"], dtype=np.float32)
        dough_hsv_max = np.array(self.config["dough_hsv_max"], dtype=np.float32)
        dough, rest = color_segmentation(pc, dough_hsv_min, dough_hsv_max)

        if self.config["visualize_flag"][inspect.currentframe().f_code.co_name]:
            dough_copy = copy.deepcopy(dough)
            dough_copy.paint_uniform_color([0, 0, 1])
            visualize_o3d([dough_copy, rest], title="selected_point_cloud_and_the_rest")

        return dough

    def denoise(self, dough, denoise_depth=0):
        dough_denoise = dough.voxel_down_sample(voxel_size=0.001)

        if not denoise_depth:
            denoise_depth = self.config["denoise_depth"]

        iter = 0
        outliers = None
        while iter < denoise_depth:
            cl, inlier_ind = dough_denoise.remove_statistical_outlier(
                nb_neighbors=50, std_ratio=1.5 + 0.25 * iter
            )
            inliers_cur = dough_denoise.select_by_index(inlier_ind)
            outliers_cur = dough_denoise.select_by_index(inlier_ind, invert=True)
            outliers = outliers_cur if outliers is None else outliers + outliers_cur

            if len(outliers_cur.points) == 0:
                break

            dough_denoise = inliers_cur
            iter += 1

        if self.config["visualize_flag"][inspect.currentframe().f_code.co_name]:
            outliers.paint_uniform_color([0.0, 0.8, 0.0])
            visualize_o3d(
                [dough_denoise, outliers], title="denoised_point_cloud_and_outliers"
            )

        return dough_denoise

    # @profile
    def sample_in_mesh(self, dough):
        mesh = self.surface_reconstuction.reconstruct(
            dough, **self.config["dense_recontruct_config"]
        )

        min_bounds = dough.get_min_bound()
        max_bounds = dough.get_max_bound()
        sample_size = self.config["sample_factor"] * self.config["n_points"]
        sampled_points = (
            np.random.rand(sample_size, 3) * (max_bounds - min_bounds) + min_bounds
        )

        f = SDF(mesh.vertices, mesh.triangles)

        sdf = f(sampled_points)
        sampled_points = sampled_points[sdf > 0]

        sampled_dough = o3d.geometry.PointCloud()
        sampled_dough.points = o3d.utility.Vector3dVector(sampled_points)

        if self.config["visualize_flag"][inspect.currentframe().f_code.co_name]:
            visualize_o3d(
                [dough, mesh],
                title=f"{self.config['dense_recontruct_config']['method']}_surface_reconstruction",
            )

        return sampled_dough

    def filter_by_tool(self, dough, tool_geom_list):
        dough_points = np.asarray(dough.points)
        sdf = np.full(dough_points.shape[0], True)

        for tool_geom_dict in tool_geom_list:
            tool_mesh = tool_geom_dict["mesh"]
            f = SDF(tool_mesh.vertices, tool_mesh.triangles)
            sdf_partial = f(dough_points)
            sdf &= sdf_partial < 0.0

        dough_inlier_points = dough_points[sdf, :]

        dough_inliers = o3d.geometry.PointCloud()
        dough_inliers.points = o3d.utility.Vector3dVector(dough_inlier_points)

        if self.config["visualize_flag"][inspect.currentframe().f_code.co_name]:
            dough_inliers.paint_uniform_color([0.5, 0.5, 0.5])

            dough_outlier_points = dough_points[~sdf, :]
            dough_outliers = o3d.geometry.PointCloud()
            dough_outliers.points = o3d.utility.Vector3dVector(dough_outlier_points)
            dough_outliers.paint_uniform_color([0.0, 0.0, 0.0])

            visualize_o3d(
                [
                    *[tool_geom_dict["mesh"] for tool_geom_dict in tool_geom_list],
                    dough_inliers,
                    dough_outliers,
                ],
                title="filtered_point_cloud_with_tools",
            )

        return dough_inliers

    def sample_surface(self, pc, n_points, poisson=True):
        mesh = self.surface_reconstuction.reconstruct(
            pc, **self.config["surface_recontruct_config"]
        )

        if poisson:
            surface_pc = o3d.geometry.TriangleMesh.sample_points_poisson_disk(
                mesh, n_points
            )
        else:
            surface_pc = o3d.geometry.TriangleMesh.sample_points_uniformly(
                mesh, n_points
            )

        if self.config["visualize_flag"][inspect.currentframe().f_code.co_name]:
            visualize_o3d([surface_pc], title="surface_point_cloud")

        return surface_pc

    def sample_full(self, pc, n_points):
        points = np.asarray(pc.points)
        fps_points = FPS(points, n_points)
        fps_pc = o3d.geometry.PointCloud()
        fps_pc.points = o3d.utility.Vector3dVector(fps_points)

        if self.config["visualize_flag"][inspect.currentframe().f_code.co_name]:
            visualize_o3d([fps_pc], title="fps_point_cloud")

        return fps_pc

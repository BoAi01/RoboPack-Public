import sys
import numpy as np
import time
import threading
import open3d as o3d
import rclpy
import torch.cuda
from rclpy.executors import MultiThreadedExecutor
import copy

sys.path.append('/home/albert/github/robopack')

from dino_field_tracking.tracker import Tracker
from dino_field_tracking.dataset.database import RobotDeploymentSoftBubbleDatabase
from config.task_config import intrinsic_matrices, extrinsics

from perception.camera_subscriber import CameraImageSubscriber

class OnRobotTracker:

    def __init__(self, watch_dict, verbose=False, save_gpu_vmem=False, setting="pushing"):
        if setting == "pushing":
            database = RobotDeploymentSoftBubbleDatabase(n_cam=4, intrinsics=intrinsic_matrices, extrinsics=extrinsics,
                                                         image_subscriber=watch_dict)
            self.track_info = {
                'carton package':
                    {'params': {'sam_threshold': 0.3}, },
            }
            self.tracker = Tracker(num_cam=4,
                              boundaries=database.boundaries,
                              poses=np.stack([database.get_pose(i) for i in database.get_img_ids()]),
                              Ks=np.stack([database.get_K(i) for i in database.get_img_ids()]),
                              track_info=self.track_info,
                              is_vis=False,
                              verbose=False,
                              use_scale=False
                              )
        elif setting == "packing":
            database = RobotDeploymentSoftBubbleDatabase(
                cam_ids=[1, 2, 3, 0],
                intrinsics=intrinsic_matrices,
                extrinsics=extrinsics,
                image_subscriber=watch_dict
            )
            sam_threshold = 0.1
            self.prompt_info = {
                'sam_obj':
                    {'params': {'sam_threshold': sam_threshold}, },
            }
            self.tracker = Tracker(num_cam=4,
                                   boundaries=database.boundaries,
                                   poses=np.stack([database.get_pose(i) for i in database.get_img_ids()]),
                                   Ks=np.stack([database.get_K(i) for i in database.get_img_ids()]),
                                   prompt_info=self.prompt_info,
                                   is_vis=False,
                                   verbose=False,
                                   use_scale=True, # deformable tracking enabled
                                   )
        else:
            raise ValueError(f"setting {setting} not recognized")
        self.setting = setting
        self.save_gpu_vmem = save_gpu_vmem

        if self.save_gpu_vmem:
            self.tracker.to(device='cpu')
        self.database = database
        self.all_match_pts = []
        self.labels = None
        self.match_pts_list = None
        self.verbose = verbose

    def perform_tracking_step(self):
        if self.save_gpu_vmem:
            self.tracker.to(device='cuda')
        colors = np.stack([self.database.get_image(i, self.tracker.t) for i in self.database.get_img_ids()])
        depths = np.stack([self.database.get_depth(i, self.tracker.t) for i in self.database.get_img_ids()])
        self.match_pts_list, step_labels = self.tracker.take_obs(colors, depths, self.match_pts_list, verbose=self.verbose)
        if self.labels is None:
            self.labels = step_labels
        self.tracker.visualize_match_pts(self.match_pts_list, None, colors)
        self.all_match_pts.append(self.match_pts_list.copy())

        if self.save_gpu_vmem:
            self.tracker.to(device='cpu')
            torch.cuda.empty_cache()

        # make a dict of {label: [match_pts]}
        if self.setting == "pushing":
            match_pts_dict = {}
            for i, label in enumerate(self.track_info.keys()):
                match_pts_dict[label] = self.match_pts_list[i]
            return match_pts_dict
        else:
            return copy.deepcopy(self.match_pts_list)

    def perform_intermediate_tracking_steps(self, topic_dict):
        if self.save_gpu_vmem:
            self.tracker.to(device='cuda')
        results = []
        for t in range(len(topic_dict['cam_0_rgb'])):
            img_dict = [topic_dict[f'cam_{i}_rgb'][t] for i in self.database.get_img_ids()]
            depth_dict = [topic_dict[f'cam_{i}_depth'][t] for i in self.database.get_img_ids()]
            colors = np.stack([self.database.get_image(i, self.tracker.t, img_dict=img_dict) for i in self.database.get_img_ids()])
            depths = np.stack([self.database.get_depth(i, self.tracker.t, img_dict=depth_dict) for i in self.database.get_img_ids()])
            self.match_pts_list, step_labels = self.tracker.take_obs(colors, depths, self.match_pts_list, verbose=self.verbose)
            if self.labels is None:
                self.labels = step_labels
            self.tracker.visualize_match_pts(self.match_pts_list, None, colors)
            self.all_match_pts.append(self.match_pts_list.copy())
            # make a dict of {label: [match_pts]}
            if self.setting == "pushing":
                match_pts_dict = {}
                for i, label in enumerate(self.track_info.keys()):
                    match_pts_dict[label] = self.match_pts_list[i]
                results.append(match_pts_dict)
            else:
                results.append(self.match_pts_list)
        if self.save_gpu_vmem:
            self.tracker.to(device='cpu')
            torch.cuda.empty_cache()
        return results

    def get_box_points(self):
        assert self.setting == "pushing", "only implemented for pushing setting"
        tracked_pts = self.perform_tracking_step()
        box_pts = tracked_pts['carton package']
        # convert to o3d pointcloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(box_pts)
        return pcd

    def get_all_object_points(self):
        assert self.setting == "packing", "only implemented for packing setting"
        tracked_pts = self.perform_tracking_step()
        # tracked_pts is a list containing 4 items, each of which are (50, 3) in shape
        pcds = []
        for i in range(len(tracked_pts)):
            object_pts = tracked_pts[i]
            # convert to o3d pointcloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(object_pts)
            pcds.append(pcd)
        return pcds

    def close(self):
        self.tracker.release_video()


if __name__ == '__main__':
    rclpy.init(args=None)
    planner_node = rclpy.create_node('mpc_planner')

    camera_node = CameraImageSubscriber()
    executor = MultiThreadedExecutor()
    executor.add_node(camera_node)
    background_thread = threading.Thread(target=lambda: executor.spin())
    background_thread.start()
    tracker = OnRobotTracker(camera_node)
    all_match_pts = []
    labels = None
    match_pts_list = None
    t = 0
    while True:
        pts = tracker.get_box_points()
        time.sleep(1)
        t += 1
        if t == 10:
            tracker.close()
            break
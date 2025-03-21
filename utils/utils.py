import argparse
import copy
import inspect
import json
import logging
import os
import random
import shutil
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import open3d as o3d
import torch
from transforms3d.axangles import axangle2mat
from utils.macros import *
from utils.visualizer import *


class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[94m",
        "INFO": "\033[92m",
        "WARNING": "\033[93m",
        "ERROR": "\033[91m",
        "CRITICAL": "\033[1;91m",
    }

    def format(self, record):
        msg = super().format(record)
        return f"{self.COLORS.get(record.levelname)}{msg}\033[0m"


class CSVFormatter(logging.Formatter):
    def __init__(self, header, hp_prefix):
        super().__init__()
        self.n_cols = len(header.split(","))
        self.hp_prefix = hp_prefix
        self.n_items = 0

    def format(self, record):
        row = []
        items = record.msg.split(",")
        if self.n_items == 0:
            row += self.hp_prefix
            self.n_items += len(self.hp_prefix)

        row += items
        self.n_items += len(items)

        message = ",".join(row)
        if self.n_items >= self.n_cols:
            message += "\n"
            self.n_items = 0

        return message


class CSVFileHandler(logging.FileHandler):
    def __init__(self, filename, header, mode="a", encoding=None, delay=False):
        super().__init__(filename, mode, encoding, delay)
        self.stream.write(f"{header}\n")
        self.terminator = ""


def setup_logging(level, csv_header, hp_prefix, save_dir):
    # Set up the stdout logger with color formatter
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(level)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_formatter = ColorFormatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    stdout_handler.setFormatter(stdout_formatter)
    stdout_logger.addHandler(stdout_handler)

    # Set up the CSV file logger with CSV formatter
    csv_logger = logging.getLogger("csv")
    csv_logger.setLevel(level)
    csv_handler = CSVFileHandler(os.path.join(save_dir, "log.csv"), csv_header)
    csv_formatter = CSVFormatter(csv_header, hp_prefix)
    csv_handler.setFormatter(csv_formatter)
    csv_logger.addHandler(csv_handler)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def prepare_save_dir(module_name, run_id="", suffix=""):
    if len(run_id) == 0:
        run_id = datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")

    if len(suffix) > 0:
        run_id += f"_{suffix}"

    save_dir = os.path.join(RESULT_DIR, module_name, f"{run_id}")

    # shutil.copytree(
    #     os.path.join(SRC_DIR, module_name),
    #     os.path.join(save_dir, "code"),
    #     ignore=shutil.ignore_patterns("__init__.py", "__pycache__"),
    #     dirs_exist_ok=True,
    # )

    source_dir = os.path.join(SRC_DIR, module_name)
    dest_dir = os.path.join(save_dir, "code")

    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    for item in os.listdir(source_dir):
        if not item.startswith("__"):
            s = os.path.join(source_dir, item)
            # Split the filename into name and extension, add suffix to name
            name, ext = os.path.splitext(item)
            name_with_suffix = f"{name}_{run_id}{ext}"
            d = os.path.join(dest_dir, name_with_suffix)
            shutil.copy2(s, d)

    return save_dir


def store_h5_data(data, path):
    hf = h5py.File(path, "w")
    for k, v in data.items():
        hf.create_dataset(k, data=v.astype(np.float32))
    hf.close()


def load_h5_data(path):
    hf = h5py.File(path, "r")
    data = {}
    for key in hf.keys():
        data[key] = hf[key][...].astype(np.float32)
    hf.close()
    return data


def read_video_data(vid_path):
    position_list = []
    frame_list = vid_path.glob("*.h5")
    for h5_path in sorted(frame_list):
        frame_data = load_h5_data(h5_path)
        # take the frame data as input and output a graph
        position_list.append(frame_data)
    return position_list


def create_profile(save_dir, script_name):
    profile_path = os.path.join(ROOT_DIR, f"{script_name}.lprof")
    with open(os.path.join(save_dir, f"{script_name}.txt"), "w") as f:
        subprocess.run(
            [
                "python",
                "-m",
                "line_profiler",
                profile_path,
            ],
            stdout=f,
        )
    subprocess.run(["rm", "-f", profile_path])


def batch_normalize(batch, mean, std, eps=1e-10):
    if len(mean.shape) == 1:
        return (batch - mean) / (std + eps)
    elif len(mean.shape) == 2:
        batch_new = []
        for i in range(batch.shape[0]):
            batch_new.append((batch[i] - mean[i]) / (std[i] + eps))
        return torch.stack(batch_new)
    else:
        raise NotImplementedError


def batch_denormalize(batch, mean, std, eps=1e-10):
    if len(mean.shape) == 1:
        return batch * (std + eps) + mean
    elif len(mean.shape) == 2:
        batch_new = []
        for i in range(batch.shape[0]):
            batch_new.append(batch[i] * (std[i] + eps) + mean[i])
        return torch.stack(batch_new)
    else:
        raise NotImplementedError


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def expand(batch_size, info):
    length = len(info.shape)
    if length == 2:
        info = info.expand([batch_size, -1])
    elif length == 3:
        info = info.expand([batch_size, -1, -1])
    elif length == 4:
        info = info.expand([batch_size, -1, -1, -1])
    return info


def is_rotation_matrix(R):
    """
    Checks if a matrix is a valid rotation matrix.
    """
    Rt = torch.transpose(R, 1, 2)
    shouldBeIdentity = torch.matmul(Rt, R)
    I = torch.eye(3).expand_as(shouldBeIdentity).to(R.device)
    maxError = torch.max(torch.abs(shouldBeIdentity - I))

    return torch.allclose(R.det(), torch.tensor(1.0).to(R.device)) and maxError < 1e-6


def euler_to_rot_matrix(euler_angles):
    B = euler_angles.shape[0]

    # Roll, pitch and yaw
    roll, pitch, yaw = euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]

    cos_roll = torch.cos(roll)
    sin_roll = torch.sin(roll)
    cos_pitch = torch.cos(pitch)
    sin_pitch = torch.sin(pitch)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    # Create the rotation matrix
    R = torch.zeros((B, 3, 3), device=euler_angles.device)

    # Populate rotation matrix
    R[:, 0, 0] = cos_yaw * cos_pitch
    R[:, 0, 1] = cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll
    R[:, 0, 2] = cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll
    R[:, 1, 0] = sin_yaw * cos_pitch
    R[:, 1, 1] = sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll
    R[:, 1, 2] = sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll
    R[:, 2, 0] = -sin_pitch
    R[:, 2, 1] = cos_pitch * sin_roll
    R[:, 2, 2] = cos_pitch * cos_roll

    return R


def get_circle(center, radius, dim, axis, alpha=1.5):
    # sunflower seed arrangement
    # https://stackoverflow.com/a/72226595
    n_exterior = np.round(alpha * np.sqrt(dim)).astype(int)
    n_interior = dim - n_exterior

    k_theta = np.pi * (3 - np.sqrt(5))
    angles = np.linspace(k_theta, k_theta * dim, dim)

    r_interior = np.linspace(0, 1, n_interior)
    r_exterior = np.ones(n_exterior)
    r = radius * np.concatenate((r_interior, r_exterior))

    circle_2d = r * np.stack((np.cos(angles), np.sin(angles)))
    circle = np.concatenate((circle_2d[:axis], np.zeros((1, dim)), circle_2d[axis:]))
    circle = circle.T + center

    return circle


def get_square(center, unit_size, dim, axis):
    state = []
    n_rows = int(np.sqrt(dim))
    for i in range(dim):
        row = i // n_rows - (n_rows - 1) / 2
        col = i % n_rows - (n_rows - 1) / 2
        pos = [unit_size * row, unit_size * col]
        pos.insert(axis, 0)
        state.append(center + np.array(pos))
    return np.array(state, dtype=np.float32)


def get_normals(points, pkg="numpy"):
    B = points.shape[0]
    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=8)
    if pkg == "torch":
        points = points.detach().cpu().numpy().astype(np.float64)

    tool_normals_list = []
    for b in range(B):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[b])
        pcd.estimate_normals(search_param, fast_normal_computation=True)
        pcd.orient_normals_towards_camera_location(pcd.get_center())
        normals = np.negative(np.asarray(pcd.normals))
        # pcd.normals = o3d.utility.Vector3dVector(normals)
        # visualize_o3d([pcd], show_normal=True)
        if pkg == "torch":
            normals = torch.tensor(normals, dtype=torch.float32)

        tool_normals_list.append(normals)

    if pkg == "torch":
        return torch.stack(tool_normals_list)
    else:
        return np.stack(tool_normals_list)


def add_shape_to_seq(args, state_seq, init_pose_seq, act_seq):
    state_seq_new = []
    for i in range(act_seq.shape[0]):
        tool_pos_list = []
        tool_start = 0
        for j in range(len(args.tool_dim[args.env])):
            tool_dim = args.tool_dim[args.env][j]
            tool_pos = init_pose_seq[i, tool_start : tool_start + tool_dim]
            tool_pos_list.append(tool_pos)
            tool_start += tool_dim

        for j in range(act_seq.shape[1]):
            step = i * act_seq.shape[1] + j
            for k in range(len(args.tool_dim[args.env])):
                tool_pos_list[k] += np.tile(
                    act_seq[i, j, 6 * k : 6 * k + 3], (tool_pos_list[k].shape[0], 1)
                )

            state_new = np.concatenate(
                [state_seq[step], args.floor_state, *tool_pos_list]
            )
            state_seq_new.append(state_new)

    return np.array(state_seq_new)


def get_act_seq_from_state_seq(args, state_shape_seq):
    # for rollers
    spread = False
    roller_motion_z_dist_prev = 0

    act_seq = []
    actions = []
    # state_diff_list = []
    for i in range(1, state_shape_seq.shape[0]):
        action = []
        tool_start = args.n_particles + args.floor_dim

        if args.full_repr and "roller" in args.env:
            roller_motion = np.mean(state_shape_seq[i, tool_start:], axis=0) - np.mean(
                state_shape_seq[i - 1, tool_start:], axis=0
            )
            roller_motion_z_dist = abs(roller_motion[2])
            # print(roller_motion_z_dist)
            if (
                not spread
                and roller_motion_z_dist_prev > 0.0001
                and roller_motion_z_dist < 0.0001
            ):
                print("spread!")
                spread = True

            roller_motion_z_dist_prev = roller_motion_z_dist

            roll_angle = 0
            if spread:
                roller_motion_xy_dist = np.linalg.norm(roller_motion[:2])
                if roller_motion_xy_dist > 0:
                    roll_norm = np.cross(
                        roller_motion[:2],
                        (state_shape_seq[i, -1] - state_shape_seq[i, tool_start]),
                    )
                    roll_dir = roll_norm[2] / abs(roll_norm[2])
                    if "large" in args.env:
                        roll_angle = roll_dir * roller_motion_xy_dist / 0.02
                    else:
                        roll_angle = roll_dir * roller_motion_xy_dist / 0.012

            action.extend([roller_motion, [roll_angle, 0, 0]])
        else:
            for j in range(len(args.tool_dim[args.env])):
                tool_dim = args.tool_dim[args.env][j]
                state_diff = (
                    state_shape_seq[i, tool_start] - state_shape_seq[i - 1, tool_start]
                )
                # state_diff_list.append(np.linalg.norm(state_diff))
                action.extend([state_diff, np.zeros(3)])
                tool_start += tool_dim

        actions.append(np.concatenate(action))

    act_seq.append(actions)

    return np.array(act_seq)


def get_normals_from_state(args, state, visualize=False):
    state_normals_list = []

    dough_points = state[: args.n_particles]
    dough_normals = get_normals(dough_points[None])[0]
    state_normals_list.append(dough_normals)

    dough_pcd = o3d.geometry.PointCloud()
    dough_pcd.points = o3d.utility.Vector3dVector(dough_points)
    dough_pcd.normals = o3d.utility.Vector3dVector(dough_normals)

    state_normals_list.append(args.floor_normals)

    floor_pcd = o3d.geometry.PointCloud()
    floor_points = state[args.n_particles : args.n_particles + args.floor_dim]
    floor_pcd.points = o3d.utility.Vector3dVector(floor_points)
    floor_pcd.normals = o3d.utility.Vector3dVector(args.floor_normals)

    tool_start = args.n_particles + args.floor_dim
    tool_pcd_list = []
    for k in range(len(args.tool_dim[args.env])):
        tool_dim = args.tool_dim[args.env][k]
        tool_points = state[tool_start : tool_start + tool_dim]
        tool_normals = get_normals(tool_points[None])[0]
        state_normals_list.append(tool_normals)

        tool_pcd = o3d.geometry.PointCloud()
        tool_pcd.points = o3d.utility.Vector3dVector(tool_points)
        tool_pcd.normals = o3d.utility.Vector3dVector(tool_normals)
        tool_pcd_list.append(tool_pcd)

        tool_start += tool_dim

    # import pdb; pdb.set_trace()
    if visualize:
        o3d.visualization.draw_geometries(
            [dough_pcd, floor_pcd, *tool_pcd_list], point_show_normal=True
        )

    return np.concatenate(state_normals_list, axis=0)


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find("ReLU") != -1:
        m.inplace = True

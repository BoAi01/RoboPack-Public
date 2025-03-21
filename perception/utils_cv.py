import argparse
import json
import logging
import os
import random
import shutil
import sys
import cv2
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy
from PIL import Image

import h5py
import numpy as np
import open3d as o3d
import torch
from scipy.interpolate import RegularGridInterpolator


# from src.config.macros import *
# from src.utils.visualizer import *

def force_arrow_visualize(img, f, centroid, scale=5e2):
    h, w = img.shape[0], img.shape[1]

    if centroid is None:
        center = (int(w / 2.0), int(h / 2.0))
    else:
        center = (centroid[0], centroid[1])

    shear_tip = np.around(np.array([center[0] + f[0] * scale, center[1] + f[1] * scale])).astype(int)

    img = cv2.arrowedLine(
        img,
        pt1=center,
        pt2=tuple(shear_tip),
        color=(250, 250, 250),
        thickness=2,
        tipLength=0.5,
    )
    
    normal_tip = np.around(
        np.array([center[0], center[1] + f[2] * scale / 100])
    ).astype(int)
    
    # print(f'shear tip: {f[0] * scale, f[1] * scale}, normal tip {f[2] * scale}')

    img = cv2.arrowedLine(
        img,
        pt1=center,
        pt2=tuple(normal_tip),
        color=(50, 255, 50),    # green
        thickness=2,
        tipLength=0.5,
    )
    
    return img


def visualize_raw_flow(flow, scale=1, step=20):
    """
    Given a flow, add an overlay of force vectors to the image.

    Modified from visualize_flow_arrows.
    """

    h, w = flow.shape[0], flow.shape[1]
    # flag = False
    # color = (20, 255, 255)  # BGR
    color = (20, 33, 255)  # BGR

    # arrows_img = np.zeros((h, w, 3))
    arrows_img = np.zeros((h, w, 3)) + np.array([245, 245, 237])

    # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # print(flow.min(), flow.max())
    # print(mag)

    # Add the arrows, skipping every *step* pixels
    for i in range(0, h, step):
        for j in range(0, w, step):
            # mags = scale * mag[i, j]
            # if int(mags):
            # ndx = min(i + int(mags * np.sin(ang[i, j])), h)
            # ndy = min(j + int(mags * np.cos(ang[i, j])), w)
            pt1 = (j, i)
            # pt2 = (max(ndy, 0), max(ndx, 0))
            pt2 = (min(int(flow[i, j, 0] * scale + j), w), min(int(flow[i, j, 1] * scale + i), h))
            arrows_img = cv2.arrowedLine(
                arrows_img,
                pt1,
                pt2,
                color,
                thickness=1,
                tipLength=0.5,      # 1, 0.5 makes a clearer figure
            )
            flag = not True

    # if flag:
    #     if len(img.shape) == 3:
    #         # Just want to overlay the arrows
    #         img2gray = cv2.cvtColor(arrows_img, cv2.COLOR_BGR2GRAY)
    #         _, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
    #         mask_inv = cv2.bitwise_not(mask)
    #         masked_source = cv2.bitwise_and(img, img, mask=mask_inv)
    #         img = cv2.addWeighted(arrows_img, 1.0, masked_source, 1.0, 0)
    #     else:
    #         img = cv2.add(img, arrows_img)

    return arrows_img


def visualize_flow_arrows(flow, img, scale=1, step=20):
    """ Given a flow, add an overlay of force vectors to the image. """
    h, w = img.shape[0], img.shape[1]
    flag = False
    color = (20, 255, 255)  # BGR

    arrows_img = np.zeros_like(img)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Add the arrows, skipping every *step* pixels
    for i in range(0, mag.shape[0], step):
        for j in range(0, mag.shape[1], step):
            mags = scale * mag[i, j]
            if int(mags):
                ndx = min(i + int(mags * np.sin(ang[i, j])), h)
                ndy = min(j + int(mags * np.cos(ang[i, j])), w)
                pt1 = (j, i)
                pt2 = (max(ndy, 0), max(ndx, 0))
                arrows_img = cv2.arrowedLine(
                    arrows_img,
                    pt1,
                    pt2,
                    color,
                    2,
                    tipLength=0.25,
                )
                flag = True
    if flag:
        if len(img.shape) == 3:
            # Just want to overlay the arrows
            img2gray = cv2.cvtColor(arrows_img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            masked_source = cv2.bitwise_and(img, img, mask=mask_inv)
            img = cv2.addWeighted(arrows_img, 1.0, masked_source, 1.0, 0)
        else:
            img = cv2.add(img, arrows_img)

    return img


def remove_redundant_images(images, threshold):
    """
    Returns a set of unique images in an image list
    :param images: a list of images in numpy array format
    :param threshold: the number of different pixels below which two images are considered the same
    :return: a list of unique images
    """
    if isinstance(images[0], torch.Tensor):
        images = [image.cpu().numpy() for image in images]

    unique_images = []
    unique_images.append(images[0])  # Add the first image to the unique_images list

    for img in images[1:]:
        # Compare the current image (img) with all unique images
        is_redundant = False
        for unique_img in unique_images:
            pixel_diff = np.sum(np.abs(unique_img.astype(float) - img.astype(float)))
            pixel_same = np.logical_and(unique_img, img).astype(float).sum() / min(unique_img.astype(float).sum(), img.astype(float).sum())
            if pixel_diff <= threshold or pixel_same > 0.9:
                is_redundant = True
                break

        if not is_redundant:
            unique_images.append(img)

    return unique_images



def get_mask_image(masks, random_color=True):
    """
    Returns the mask image corresponding to the masks from SAM.
    :param masks: an np array given by the SAM model
    :param random_color: whether to use different and random colors for different objects
    :return: A np array image
    """
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()

    mask_images = []
    for mask in masks:
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        mask_images.append(mask_image)

    mask_image_rgb = 1 - sum(mask_images)[..., :3]

    return mask_image_rgb


def crop_out_masked_region(image, mask, null_value=0):
    cropped_image = image.copy()
    cropped_image[mask != 1] = null_value  # null value
    return cropped_image


def overlay_mask_to_image(image, mask):
    """
    Assume the image and the mask are of shape (:, :, 3)
    """
    img = image.copy()
    img[mask.mean(2) != 1] = mask[mask.mean(2) != 1]
    return img


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


def prepare_save_dir(module_name, run_id=None):
    if run_id is None:
        run_id = datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")
    save_dir = os.path.join(RESULT_DIR, module_name, f"{run_id}")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        os.path.join(SRC_DIR, module_name),
        os.path.join(save_dir, "code"),
        ignore=shutil.ignore_patterns("__init__.py", "__pycache__"),
        dirs_exist_ok=True,
    )
    shutil.copy(
        os.path.join(CONFIG_DIR, f"{module_name}_config.json"),
        os.path.join(save_dir, "config.json"),
    )

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
            tool_pos = init_pose_seq[i, tool_start: tool_start + tool_dim]
            tool_pos_list.append(tool_pos)
            tool_start += tool_dim

        for j in range(act_seq.shape[1]):
            step = i * act_seq.shape[1] + j
            for k in range(len(args.tool_dim[args.env])):
                tool_pos_list[k] += np.tile(
                    act_seq[i, j, 6 * k: 6 * k + 3], (tool_pos_list[k].shape[0], 1)
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
                    and roller_motion_z_dist_prev > 0.0001 > roller_motion_z_dist
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
    floor_points = state[args.n_particles: args.n_particles + args.floor_dim]
    floor_pcd.points = o3d.utility.Vector3dVector(floor_points)
    floor_pcd.normals = o3d.utility.Vector3dVector(args.floor_normals)

    tool_start = args.n_particles + args.floor_dim
    tool_pcd_list = []
    for k in range(len(args.tool_dim[args.env])):
        tool_dim = args.tool_dim[args.env][k]
        tool_points = state[tool_start: tool_start + tool_dim]
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


# def concatenate_images_side_by_side(*images):
#     """
#     Concatenates PIL images side by side.

#     Args:
#         *images (PIL.Image.Image): Variable number of PIL images.

#     Returns:
#         PIL.Image.Image: Concatenated image.
#     """
#     # Ensure that all images have the same height
#     max_height = max(image.height for image in images)
#     images = [image.resize((int(image.width * max_height / image.height), max_height)) for image in images]

#     # Convert PIL images to NumPy arrays
#     arrays = [np.array(image) for image in images]

#     # Concatenate the arrays horizontally
#     concatenated_array = np.concatenate(arrays, axis=1)

#     # Convert the concatenated array back to PIL image
#     concatenated_image = Image.fromarray(concatenated_array)

#     return concatenated_image


def concatenate_images_side_by_side(*images):
    """
    Concatenates PIL images or NumPy arrays side by side.

    Args:
        *images: Variable number of images. Can be PIL images or NumPy arrays.

    Returns:
        PIL.Image.Image: Concatenated image.
    """
    # Convert NumPy arrays to PIL images if needed
    images = [Image.fromarray(image) if isinstance(image, np.ndarray) else image for image in images]

    # Ensure that all images have the same height
    max_height = max(image.height for image in images)
    images = [image.resize((int(image.width * max_height / image.height), max_height)) for image in images]

    # Convert PIL images to NumPy arrays
    arrays = [np.array(image) for image in images]

    # Concatenate the arrays horizontally
    concatenated_array = np.concatenate(arrays, axis=1)

    # Convert the concatenated array back to PIL image
    concatenated_image = Image.fromarray(concatenated_array)

    return concatenated_image


def resize_image(original_image, H, W):
    h, w, _ = original_image.shape

    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, h - 1, h)
    new_x = np.linspace(0, w - 1, W)
    new_y = np.linspace(0, h - 1, H)

    interpolators = [RegularGridInterpolator((y, x), original_image[..., channel]) for channel in range(2)]

    new_image = np.zeros((H, W, 2), dtype=original_image.dtype)
    new_coords = np.array(np.meshgrid(new_y, new_x)).T.reshape(-1, 2)

    for channel in range(2):
        new_values = interpolators[channel](new_coords)
        new_image[..., channel] = new_values.reshape((H, W))

    return new_image


def find_point_on_line(start_point, end_point, distance):
    """
    Find a point on the line connecting A and B and is at a specific distance
    to one of the points.
    :param start_point: starting point
    :param end_point: end point
    :param distance: distance to the end point along the line AB
    :return: the coordinate for point C
    """
    # Convert the points to NumPy arrays for vector operations
    A = np.array(start_point)
    B = np.array(end_point)

    # Calculate the direction vector from A to B
    direction_vector = B - A

    # Normalize the direction vector to get the unit direction vector
    unit_direction_vector = direction_vector / np.linalg.norm(direction_vector)

    # Calculate the coordinates of the third point C at distance 'd' from point A
    C = B + distance * unit_direction_vector

    return C
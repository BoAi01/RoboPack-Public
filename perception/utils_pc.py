import sys
import warnings

# sys.path.append('/home/albert/github/robopack/ros2_numpy')
import numpy
import numpy as np
import open3d
import open3d as o3d
import cv2
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from scipy.stats import zscore
import copy
from pysdf import SDF
import random

# from cv_bridge import CvBridge
from transforms3d.quaternions import quat2mat, mat2quat

from perception.utils_cv import crop_out_masked_region


def random_select_rows(arr, K):
    np.random.shuffle(arr)
    selected_rows = arr[:K]
    return selected_rows


def extract_workspace_pc(pc, xrange=[0.25, 0.8], yrange=[-0.5, 0.5], zrange=[-0.01, 0.6]):
    """
    Extracts the point cloud of the work space, defined as a 3D cube.
    :param pc: a point cloud in the robot frame
    :param xrange: range for the x-axis coordinates
    :param yrange: range for the y-axis coordinates
    :param zrange: range for the z-axis coordinates
    :return: the point cloud of the cropped region
    """
    new_pc = o3d.geometry.PointCloud()

    points = np.asarray(pc.points)
    mask = (points[:, 0] >= xrange[0]) & (points[:, 0] <= xrange[1]) & \
           (points[:, 1] >= yrange[0]) & (points[:, 1] <= yrange[1]) & \
           (points[:, 2] >= zrange[0]) & (points[:, 2] <= zrange[1])
    filtered_points = points[mask]

    new_pc.points = o3d.utility.Vector3dVector(filtered_points)

    colors = np.asarray(pc.colors)
    if len(colors) > 0:
        filtered_colors = colors[mask]
        new_pc.colors = o3d.utility.Vector3dVector(filtered_colors)

    return new_pc


def apply_transform_matrix(points, transform_matrix):
    assert 3 <= points.shape[1] <= 4, "the xyz should be on the 2nd dim"
    points = np.concatenate([points, np.ones([points.shape[0], 1])], axis=1)
    transformed = (transform_matrix @ points.T).T
    return transformed[:, :3]


def pos_rot_to_matrix(pos, rot):
    """
    Convert position and rotation matrix to transform matrix
    :param pos: x y z
    :param rot: 3x3 rotation matrix
    :return: the corresponding transform matrix
    """
    if list(pos.shape) == [3]:
        pos = np.expand_dims(pos, 1)
    return np.concatenate((np.concatenate((rot, pos), axis=1), [[0, 0, 0, 1]]))


def pos_quat_to_matrix(pos, quat, wxyz=True):
    """
    Converts position and quaternion to transformation matrix
    :param pos: position xyz
    :param quat: quaternion
    :return:
    """
    assert len(pos) == 3, 'position should be xyz'
    if not wxyz:
        assert len(wxyz) == 4, 'the quaternion should be in the format of xyzw'
        quat = xyzw_to_wxyz(quat)
    rot = quat2mat(quat)
    pos = np.expand_dims(pos, 1)
    matrix = pos_rot_to_matrix(pos, rot)
    return matrix


def xyzw_to_wxyz(xyzw):
    """
    Converts quaternion in xyzw to wxyz
    :param xyzw: quant in xyzw
    :return: wxyz
    """
    return xyzw[[3, 0, 1, 2]]


def wxyz_to_xyzw(wxyz):
    """
    Convert quaternion in wxyz to xyzw.
    :param wxyz: quant in wxyz
    :return: xyzw
    """
    return wxyz[[1, 2, 3, 0]]


def apply_quaternion(points, quat):
    rot_mat = quat2mat(quat)
    transformed = apply_transform_matrix(points, rot_mat)
    return transformed


def project_pc(pc, transform_matrix):
    points = np.asarray(pc.points)
    points = apply_transform_matrix(points, transform_matrix)
    new_pc = o3d.geometry.PointCloud()
    new_pc.points = o3d.utility.Vector3dVector(points)
    new_pc.colors = o3d.utility.Vector3dVector(np.asarray(pc.colors).copy())
    return new_pc


def project_points(points, transform_matrix):
    """
    Apply a transformation matrix to a point cloud with features.

    Args:
        points (numpy.ndarray): Point cloud array with shape (N, 3 + feat_dim).
            Each row contains [x, y, z, feat_dim elements].
        transform_matrix (numpy.ndarray): 4x4 transformation matrix.

    Returns:
        numpy.ndarray: Transformed point cloud array with shape (N, 3 + feat_dim).
            Each row contains [x, y, z, feat_dim elements] after the transformation.
    """
    # Pad ones for homogeneous coordinates
    points_homogeneous = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))

    # Apply the transformation matrix
    transformed_homogeneous = np.dot(transform_matrix, points_homogeneous.T).T

    # Extract the transformed xyz coordinates and features
    transformed_points = transformed_homogeneous[:, :3]
    features = points[:, 3:] if points.shape[1] > 3 else None

    # Combine the transformed xyz coordinates and features
    transformed_cloud = np.hstack((transformed_points, features)) if features is not None else transformed_points

    return transformed_cloud


def convert_pc_color_to_link_frame(pc_in_color_opt, cam_idx=None):
    """
    Convert point cloud in color frame to the link frame.
    This function is needed when the depth map is aligned to the color map.
    :param pc_in_color_opt:
    :param cam_idx:
    :return:
    """
    color_opt_to_link = {   # from rviz, position + orientation
        0: [-0.00019038, 0.014772, 5.346e-05, 1, 0.00033169, -0.00072087, 0.00073762],
        1: [0.0001781, 0.014865, 9.6475e-05, 0.99999, 0.00044859, -0.0023165, -0.0033206],
        2: [0.00022101, 0.014945, -3.8073e-05, 1, 0.00057514, 0.00054611, -0.00082246],
        3: [0.00038961, 0.015091, -0.00010821, 0.99999, 0.00019225, -0.0034192, 0.0035338]
    }
    # if cam_idx is None:
    #     warnings.warn('cam_idx is None, which may cause frame conversion to be inaccurate')
    opt_in_link = color_opt_to_link[cam_idx] if cam_idx is not None else color_opt_to_link[0]
    T_opt_to_link = pos_quat_to_matrix(opt_in_link[:3], opt_in_link[-4:], True)

    # perform projection
    if isinstance(pc_in_color_opt, o3d.geometry.PointCloud):
        pc_in_link = project_pc(pc_in_color_opt, T_opt_to_link)
    else:
        pc_in_link = project_points(pc_in_color_opt, T_opt_to_link)

    return pc_in_link


def convert_pc_optical_depth_to_link_frame(pc_in_opt):
    """
    Convert point cloud in optical frame to non-optical frame.
    :param pc_in_opt:
    :return:
    """
    # raise NotImplementedError(f'You should be using convert_pc_color_optical_to_link_frame, '
    #                           f'and make sure you aligned depth to color')
    opt_in_link = [0, 0, 0, 0.5, -0.5, 0.5, -0.5]  # from rviz, position + orientation
    T_opt_to_link = pos_quat_to_matrix(opt_in_link[:3], opt_in_link[-4:], True)
    pc_in_link = project_pc(pc_in_opt, T_opt_to_link)
    return pc_in_link


def convert_pc_optical_color_to_link_frame(pc_in_color_opt, cam_idx=None):
    """
    Converts point cloud in optical color frame directly to the link frame.
    This function might be needed when the depth map is aligned to the color map.
    This is basically a composition of (optical color -> non-optical color -> link)
    :param pc_in_color_opt:
    :param cam_idx:
    :return:
    """
    color_opt_to_link = {   # from rviz, position + orientation
        0: [-0.00019038, 0.014772, 5.346e-05, 0.50089, -0.49984, 0.49944, -0.49983],
        1: [0.0001781, 0.014865, 9.6475e-05, -0.49972, 0.49695, -0.50072, 0.50259],
        2: [0.00022101, 0.014945, -3.8073e-05, 0.4996, -0.49957, 0.50097, -0.49985],
        3: [0.00038961, 0.015091, -0.00010821, 0.50357, -0.49996, 0.49661, -0.49984]
    }

    # if cam_idx is None:
    #     warnings.warn('cam_idx is None, which may cause frame conversion to be inaccurate')

    opt_in_link = color_opt_to_link[cam_idx] if cam_idx is not None else color_opt_to_link[0]
    T_opt_to_link = pos_quat_to_matrix(opt_in_link[:3], opt_in_link[-4:], True)

    if isinstance(pc_in_color_opt, o3d.geometry.PointCloud):
        pc_in_link = project_pc(pc_in_color_opt, T_opt_to_link)
    else:
        pc_in_link = project_points(pc_in_color_opt, T_opt_to_link)

    return pc_in_link


def color_segmentation(pc, hsv_min, hsv_max):
    # (0, 30/255, 30/255), (360, 1, 1) for removing grey and black
    pc_rgb = np.asarray(pc.colors, dtype=np.float32)
    pc_hsv = cv2.cvtColor(pc_rgb[None], cv2.COLOR_RGB2HSV)  # max 360, 1, 1
    mask = cv2.inRange(pc_hsv, hsv_min, hsv_max)
    object_label = np.where(mask[0] == 255)
    object_pc = pc.select_by_index(object_label[0])
    rest_pc = pc.select_by_index(object_label[0], invert=True)

    return object_pc, rest_pc


def remove_distant_points_pc(point_cloud, max_distance):
    # Extract point coordinates and colors as NumPy arrays
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)

    # Compute Euclidean distance from origin for each point
    distances = np.linalg.norm(points, axis=1)

    # Find indices of points that are within the maximum distance
    valid_indices = distances <= max_distance

    # Select valid points and colors
    valid_points = o3d.geometry.PointCloud()
    valid_points.points = o3d.utility.Vector3dVector(points[valid_indices])

    if len(colors) != 0:
        valid_points.colors = o3d.utility.Vector3dVector(colors[valid_indices])

    return valid_points


def remove_distant_points(points, max_distance):
    # Compute Euclidean distance from origin for each point
    distances = np.linalg.norm(points[:, :3], axis=1)

    # Find indices of points that are within the maximum distance
    valid_indices = distances <= max_distance

    return points[valid_indices]


def remove_points_by_color(pc, color=(1, 1, 1)):
    if not pc.has_points():
        return pc
    colors = np.asarray(pc.colors)
    mask = cv2.inRange(colors[None], color, color)
    label = np.where(mask[0] != 255)
    pc = pc.select_by_index(label[0])
    return pc


def construct_pointcloud_from_rgbd(rgb, depth, intrinsics, extrinsics=None):
    #     cam0_depth_clipped = np.clip(cam0_depth, 0, 2000)
    #     camera_intrinsics = intrinsics['cam_0']
    rec_pcd = o3d.geometry.PointCloud()
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb),
        o3d.geometry.Image(depth.astype(np.uint16)),
        convert_rgb_to_intensity=False
    )

    if extrinsics is None:
        extrinsics = pos_quat_to_matrix(np.zeros(3), np.array([1, 0, 0, 0]), True)

    rec = rec_pcd.create_from_rgbd_image(rgbd_image, intrinsics, extrinsics)
    #     o3d.visualization.draw_geometries([cam0_rec])
    return rec


def pair_points_minimize_distance(set1, set2):
    # Compute pairwise distances between points
    # Hungarian algorithm

    distances = np.zeros((len(set1), len(set2)))
    for i, point1 in enumerate(set1):
        for j, point2 in enumerate(set2):
            distances[i, j] = np.linalg.norm(np.array(point1) - np.array(point2))

    # Solve the assignment problem using the Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(distances)

    # Generate the paired points and their indices
    pairs = [(row, col) for row, col in zip(row_indices, col_indices)]

    # Identify unmatched points in set1
    unmatched_set1 = [i for i in range(len(set1)) if i not in row_indices]

    # Identify unmatched points in set2
    unmatched_set2 = [i for i in range(len(set2)) if i not in col_indices]

    return pairs, unmatched_set1, unmatched_set2


def sample_points_from_pc(pc, denoise_strength=3, surface_rec_method='poisson', mesh_fix=1,
                          n_points=None, ratio=3e6):
    assert surface_rec_method in ["alpha_shape", "ball_pivoting", "poisson", "vista"]

    denoised_pc = denoise_pc_by_stats(pc, denoise_strength)
    surface_rec = SurfaceReconstruction()
    mesh = surface_rec.reconstruct(denoised_pc, surface_rec_method, mesh_fix=mesh_fix)
    mesh = mesh.filter_smooth_taubin(denoise_strength)

    if n_points is None:
        n_points = int(ratio * mesh.get_oriented_bounding_box().volume())
    sampled_pc = sample_points_from_mesh(mesh, n_points)
    # denoised_sampled_pc = denoise_pc(sampled_pc)

    return sampled_pc


def fill_point_cloud_by_downward_projection(point_cloud, bottom_z=None):
    """
    Projects points downwards to the bottom z plane.
    :param point_cloud: An Open3D point cloud
    :param bottom_z: The bottom plane for projection
    :return: A point cloud obtained by projecting all points down so that they are uniformly
    distributed between the bottom plane and their original positions.
    """
    # Get the minimum and maximum z-coordinate values
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)
    z_min = np.min(points[:, 2])
    z_max = np.max(points[:, 2])

    # Check if there are points above the plane z = 0
    if z_max > 0:
        # Create a new point cloud with existing points
        sealed_point_cloud = o3d.geometry.PointCloud()

        # Project object points down to a random z = 0 plane
        new_points = points.copy()
        new_colors = colors.copy()
        if bottom_z is None:
            bottom_z = np.quantile(new_points[:, 2], 0.03)  # use this as the z = 0 plane
        new_points[:, 2] = bottom_z + (new_points[:, 2] - bottom_z) * np.random.uniform(0, 1, size=new_points.shape[0])

        # new_points = remove_outliers_pc(new_points)
        # keep the colors of the points
        sealed_point_cloud.points = o3d.utility.Vector3dVector(new_points)
        sealed_point_cloud.colors = o3d.utility.Vector3dVector(new_colors)

        # set the color to be the mean object color
        # object_color = np.mean(point_cloud.colors, axis=0)
        # colors = np.tile(object_color, (len(new_points), 1))
        # sealed_point_cloud.colors = o3d.utility.Vector3dVector(colors)

        return sealed_point_cloud

    return point_cloud


def visualize_point_clouds(point_clouds):
    # Generate a color map for the point clouds
    num_clouds = len(point_clouds)
    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, num_clouds))

    # Create a visualization window
    #     visualizer = o3d.visualization.Visualizer()

    # Iterate over the point clouds and assign colors
    for i, point_cloud in enumerate(point_clouds):
        # Assign a unique color to the current point cloud
        color = colors[i][:3]
        point_cloud.paint_uniform_color(color)

    # Customize visual properties (optional)
    #     visualizer.get_render_option().point_size = 3  # Set point size

    # Run the visualization
    #     visualizer.run()
    o3d.visualization.draw_geometries(point_clouds)


def remove_outliers_pc(points, threshold=2.0):
    n = points.shape[0]

    # Calculate z-scores for x and y axis values
    z_scores = zscore(points[:, :2], axis=0)
    #     print(z_scores.shape)

    # Identify outliers
    outliers = np.logical_or(np.abs(z_scores[:, 0]) > threshold, np.abs(z_scores[:, 1]) > threshold)

    # Remove half of the outliers
    num_outliers = np.sum(outliers)
    #     print(num_outliers)
    num_remove = num_outliers // 3 * 2
    indices = np.random.choice(np.where(outliers)[0], size=num_remove, replace=False)
    points = np.delete(points, indices, 0)

    return points


def parse_pointcloud2(msg):
    """
    Extracts point cloud from ROS1 PointCloud2 type.
    :param msg: ROS1 message
    :return: Open3D point cloud object
    """
    data = np.frombuffer(msg.data, dtype=np.uint8)
    data = data.reshape(-1, msg.point_step)

    cloud_xyz = copy.deepcopy(data[:, :12]).view(dtype=np.float32).reshape(-1, 3)
    cloud_bgr = copy.deepcopy(data[:, 16:20]) / 255
    cloud_rgb = cloud_bgr[:, -2::-1]

    return cloud_xyz, cloud_rgb


def compute_normals(pc):
    """
    Computes the normal surface of a point cloud and stores it as an attribute.
    :param pc: the point cloud
    :return: the updated point cloud
    """
    pc.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    pc.estimate_normals()
    pc.orient_normals_consistent_tangent_plane(100)

    hull, _ = pc.compute_convex_hull()
    center = hull.get_center()

    points = np.asarray(pc.points)
    normals = np.asarray(pc.normals)
    for i, n in enumerate(normals):
        if np.dot(points[i] - center, n) < 0:
            normals[i] = np.negative(normals[i])

    pc.normals = o3d.utility.Vector3dVector(normals)
    return pc


def farthest_point_sampling_dgl_pc(point_cloud, n_points, ratio=None, seed=None):
    import torch
    from dgl.geometry import farthest_point_sampler

    assert n_points is None or ratio is None, f"can only specify a value for n_points OR ratio"

    # Convert the point cloud to a NumPy array
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)

    if n_points is None:
        n_points = int(len(points) * ratio)
    assert n_points > 0, 'number of points to sample should be positive'

    indices = farthest_point_sampler(torch.from_numpy(points).unsqueeze(0), n_points, start_idx=seed).squeeze(0)

    # Create a new point cloud with the selected points
    sampled_point_cloud = o3d.geometry.PointCloud()
    sampled_point_cloud.points = o3d.utility.Vector3dVector(points[indices])
    if len(colors) > 0:
        sampled_point_cloud.colors = o3d.utility.Vector3dVector(colors[indices])

    return sampled_point_cloud


def farthest_point_sampling_dgl(points, n_points):
    import torch
    from dgl.geometry import farthest_point_sampler

    indices = farthest_point_sampler(torch.from_numpy(points[:, :3]).unsqueeze(0), n_points).squeeze(0)

    return points[indices]


def voxel_grid_filter(pc, points, grid_size=0.001, axis=2, visualize=0):
    lower = pc.get_min_bound()
    upper = pc.get_max_bound()
    pc_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pc, 0.9 * grid_size)

    if visualize:
        visualize_o3d([pc, pc_grid], title="voxel_grid_filter")

    ax_grid = np.arange(lower[axis], upper[axis], grid_size)
    tiled_points = np.tile(points[:, None, :], (1, ax_grid.shape[0], 1))
    test_points = copy.deepcopy(tiled_points)
    test_points[:, :, axis] = ax_grid
    exists_mask = np.array(
        pc_grid.check_if_included(
            o3d.utility.Vector3dVector(test_points.reshape((-1, 3)))
        )
    )
    exists_mask = exists_mask.reshape((-1, ax_grid.shape[0]))
    if axis == 2:
        vg_up_mask = (
                np.sum((tiled_points[:, :, axis] < ax_grid) * exists_mask, axis=1) > 0
        )
        vg_down_mask = (
                np.sum((tiled_points[:, :, axis] > ax_grid) * exists_mask, axis=1) > 0
        )
        return vg_up_mask & ~vg_down_mask
    else:
        raise NotImplementedError


class SurfaceReconstruction(object):
    def __init__(self):
        self.methods = ["alpha_shape", "ball_pivoting", "poisson", "vista"]

    def reconstruct(self, pc, method, mesh_fix=0, **kwargs):
        if not method in self.methods:
            raise NotImplementedError(f'method {method} not available')

        else:
            method_func = getattr(self, method)
            mesh = method_func(pc, **kwargs)

        if mesh_fix:
            import pymeshfix
            mf = pymeshfix.MeshFix(
                np.asarray(mesh.vertices), np.asarray(mesh.triangles)
            )
            mf.repair()
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(mf.mesh.points)
            mesh.triangles = o3d.utility.Vector3iVector(
                mf.mesh.faces.reshape(mf.mesh.n_faces, -1)[:, 1:]
            )

        return mesh

    def alpha_shape(self, pc, alpha=0.5):
        tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pc)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pc, alpha, tetra_mesh, pt_map
        )
        return mesh

    def ball_pivoting(self, pc, radii=[0.001, 0.002, 0.004, 0.008]):
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pc, o3d.utility.DoubleVector(radii)
        )
        return mesh

    def vista(self, pc):
        point_cloud = pv.PolyData(np.asarray(pc.points))
        mesh = point_cloud.reconstruct_surface()
        return mesh

    def poisson(self, pc, depth=8):
        pc = compute_normals(pc)
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pc, depth=depth, n_threads=-1
        )
        return mesh


def denoise_pc_by_stats(pc, denoise_depth=1):
    """
    Removes statistical outliers from the point cloud, in multiple iterations
    :param pc: the point cloud to denoise
    :param denoise_depth: the number of iterations to remove outliers
    :return:
    """
    denoised_pc = pc.voxel_down_sample(voxel_size=0.001)

    n_iter = 0
    outliers = None
    while n_iter < denoise_depth:
        cl, inlier_ind = denoised_pc.remove_statistical_outlier(
            nb_neighbors=50, std_ratio=1.5 + 0.25 * n_iter
        )
        inliers_cur = denoised_pc.select_by_index(inlier_ind)
        outliers_cur = denoised_pc.select_by_index(inlier_ind, invert=True)
        outliers = outliers_cur if outliers is None else outliers + outliers_cur

        if len(outliers_cur.points) == 0:
            break

        denoised_pc = inliers_cur
        n_iter += 1

    return denoised_pc


def sample_points_from_mesh(mesh, n_points, seed=None):
    if seed is not None:
        np.random.seed(seed)

    min_bounds = mesh.get_min_bound()
    max_bounds = mesh.get_max_bound()

    valid_sampled_points = []

    while len(valid_sampled_points) < n_points:
        sampled_points = np.random.rand(n_points * 3, 3) * (max_bounds - min_bounds) + min_bounds

        f = SDF(mesh.vertices, mesh.triangles)
        sdf = f(sampled_points)

        valid_sampled_points.extend(sampled_points[sdf > 0])

    # Take the first 'n_points' valid points
    sampled_points = np.array(valid_sampled_points)[:n_points]

    sampled_pc = o3d.geometry.PointCloud()
    sampled_pc.points = o3d.utility.Vector3dVector(sampled_points)

    return sampled_pc


def remove_points_inside_bbox(point_cloud, bbox):
    # Convert the point cloud to a NumPy array
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)  # Get colors from the original point cloud

    # Extract the min and max bounds of the bounding box
    min_bound = bbox.get_min_bound()
    max_bound = bbox.get_max_bound()

    # Create a mask for points outside the bounding box
    mask = np.logical_or(np.any(points < min_bound, axis=1), np.any(points > max_bound, axis=1))

    # Filter out points inside the bounding box
    filtered_points = points[mask]
    filtered_colors = colors[mask]  # Apply the same mask to the colors

    # Create a new Open3D point cloud with the filtered points and colors
    filtered_point_cloud = o3d.geometry.PointCloud()
    filtered_point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)

    return filtered_point_cloud


def generate_point_cloud(num_points, x_range, y_range, zrange=None):
    """
    Generates a point cloud from a cuboid
    :param num_points: number of points to sample
    :param x_range: x-axis range of the cuboid
    :param y_range: y-axis range of the cuboid
    :param zrange: z-axis range of the cuboid
    :return:
    """
    points = []
    for _ in range(num_points):
        x = random.uniform(x_range[0], x_range[1])
        y = random.uniform(y_range[0], y_range[1])
        if zrange is None:
            z = 0  # z-coordinate is fixed at 0
        else:
            z = random.uniform(zrange[0], zrange[1])
        points.append([x, y, z])

    # Convert the list of points to a NumPy array
    points_np = np.array(points)

    # Create an Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_np)

    return point_cloud

import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN


def denoise_by_cluster(point_cloud, epsilon, min_samples, n_largest):
    # CAUTION: when epsilon is too small and min_samples too large, this
    # function may not do anything about the point cloud
    # probably because there will not be any valid clusters

    is_o3d_cloud = isinstance(point_cloud, o3d.cuda.pybind.geometry.PointCloud)

    if is_o3d_cloud:
        # Convert Open3D point cloud to NumPy array
        points = np.asarray(point_cloud.points)
    else:
        points = point_cloud

    # Apply DBSCAN
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    labels = dbscan.fit_predict(points[:, :3])

    # Get the largest clusters
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    largest_cluster_labels = unique_labels[np.argsort(label_counts)][::-1][:n_largest]
    largest_cluster_indices = np.where(np.isin(labels, largest_cluster_labels))[0]

    # Keep only the largest clusters
    if is_o3d_cloud:
        largest_clusters = point_cloud.select_by_index(largest_cluster_indices)
    else:
        largest_clusters = point_cloud[largest_cluster_indices]

    return largest_clusters


def merge_multiview_object_pcs(cam0_pcs, cam1_pcs):
    """
    Merges object point clouds by their geometric distance and color.
    :param cam0_pcs: point clouds from one camera
    :param cam1_pcs: point clouds from another camera
    :return: a list of point clouds for each object
    """
    cam0_pcs = list(filter(lambda x: x.has_points(), cam0_pcs))
    cam1_pcs = list(filter(lambda x: x.has_points(), cam1_pcs))
    cam0_centers = [pc.get_center() for pc in cam0_pcs]
    cam1_centers = [pc.get_center() for pc in cam1_pcs]
    cam0_colors = [np.asarray(pc.colors).mean(0) for pc in cam0_pcs]
    cam1_colors = [np.asarray(pc.colors).mean(0) for pc in cam1_pcs]
    cam0_feats = [np.concatenate([x, y], axis=0) for x, y in zip(cam0_centers, cam0_colors)]
    cam1_feats = [np.concatenate([x, y], axis=0) for x, y in zip(cam1_centers, cam1_colors)]

    pairs, unmatched_set1, unmatched_set2 = pair_points_minimize_distance(cam0_feats, cam1_feats)

    # break false pairs based on color distance
    for i, j in pairs:
        if abs(cam0_colors[i] - cam1_colors[j]).mean() > 0.13:   # 0.18 for brown box and blue ball
            pairs.remove((i, j))
            unmatched_set1.append(i)
            unmatched_set2.append(j)

    # merge point clouds by object
    object_pcs = []
    for i, j in pairs:
        object_pcs.append(cam0_pcs[i] + cam1_pcs[j])

    for i in unmatched_set1:
        object_pcs.append(cam0_pcs[i])

    for j in unmatched_set2:
        object_pcs.append(cam1_pcs[j])

    return object_pcs


xrange = np.array([0.25, 0.8])
yrange = np.array([-0.4, 0.4])
zrange = np.array([0.01, 0.45])
def get_point_cloud_of_every_object(masks, rgb, depth, intrinsics, extrinsics,
                                    xrange=xrange, yrange=yrange, zrange=zrange):
    """
    Constructs the point cloud for every object from mask, rgb, and depth.
    :param masks: Instance segmentation mask obtained from SAM
    :param rgb: RGB image from camera, numpy
    :param depth: depth image from camera, numpy
    :param intrinsics: intrinsics of the camera
    :return: the point cloud of every object in the workspace
    """
    # masks = masks.cpu().numpy()[:, 0]  # remove the second dim
    cam_pcs = []
    for mask in masks:
        # each mask is of shape (1, h, w)
        masked_rgb = crop_out_masked_region(rgb, mask)
        masked_depth = crop_out_masked_region(depth, mask)
        constructed_pc = construct_pointcloud_from_rgbd(masked_rgb, masked_depth, intrinsics)
        if not constructed_pc.has_points():
            import pdb
            pdb.set_trace()
        constructed_pc = remove_points_by_color(constructed_pc)
        constructed_pc = project_pc(convert_pc_optical_depth_to_link_frame(constructed_pc), extrinsics)
        constructed_pc = extract_workspace_pc(constructed_pc, xrange, yrange, zrange)
        if constructed_pc.has_points():
            cam_pcs.append(constructed_pc)
    return cam_pcs


def subtract_point_clouds(point_cloud_1, point_cloud_2, threshold):
    """
    Subtract point_cloud_2 from point_cloud_1 by removing overlapping points based on distance.

    Args:
        point_cloud_1: First point cloud as an Open3D PointCloud object.
        point_cloud_2: Second point cloud as an Open3D PointCloud object.
        threshold: Distance threshold for determining which points to remove.

    Returns:
        A new Open3D PointCloud object representing the subtraction result with preserved colors.
    """
    # Compute distances between the points in point_cloud_1 and point_cloud_2
    distances = point_cloud_1.compute_point_cloud_distance(point_cloud_2)

    # Remove overlapping points based on distances
    indices = np.where(np.asarray(distances) > threshold)[0]
    non_overlapping_points = point_cloud_1.select_by_index(indices)
    non_overlapping_points.colors = o3d.utility.Vector3dVector(np.asarray(point_cloud_1.colors)[indices])

    return non_overlapping_points


def subtract_mesh_from_point_cloud(pc, mesh, threshold):
    threshold = -abs(threshold)
    # mesh_points = mesh.sample_points_uniformly(number_of_points=500)
    # pc_rest = subtract_point_clouds(pc, mesh_points, threshold)
    points = np.asarray(pc.points)
    colors = np.asarray(pc.colors)

    f = SDF(mesh.vertices, mesh.triangles)
    sdf = f(points)

    sampled_pc = o3d.geometry.PointCloud()
    sampled_pc.points = o3d.utility.Vector3dVector(points[sdf < threshold])
    sampled_pc.colors = o3d.utility.Vector3dVector(colors[sdf < threshold])

    return sampled_pc


def point_cloud_to_rgbd(pc, intrinsic_mat, height, width):
    assert height <= width, "height shouldn't be larger than width"

    # Extract XYZ coordinates from the point cloud
    points = np.asarray(pc.points)

    # Extract RGB colors from the point cloud
    colors = np.asarray(pc.colors)

    # Project 3D points to obtain depth values
    depth_values = points[:, 2]

    # Map RGB values using pixel coordinates
    pixel_coordinates = np.dot(intrinsic_mat, points.T).T
    pixel_coordinates[:, :2] /= pixel_coordinates[:, 2:]
    pixel_coordinates = np.round(pixel_coordinates[:, :2]).astype(int)

    # Create depth map
    depth_map = np.zeros((height, width))  # Replace height and width with your image dimensions
    valid_indices = np.logical_and(pixel_coordinates[:, 0] >= 0, pixel_coordinates[:, 0] < width)
    valid_indices = np.logical_and(valid_indices, pixel_coordinates[:, 1] >= 0)
    valid_indices = np.logical_and(valid_indices, pixel_coordinates[:, 1] < height)
    depth_map[pixel_coordinates[valid_indices, 1], pixel_coordinates[valid_indices, 0]] = depth_values[valid_indices]

    # Create RGB image
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)  # Replace height and width with your image dimensions
    rgb_image[pixel_coordinates[valid_indices, 1], pixel_coordinates[valid_indices, 0]] = (
                colors[valid_indices] * 255).astype(np.uint8)

    return rgb_image, depth_map * 1000      # m to mm


def is_pc_center_in_mesh(pc, mesh, threshold=0):
    """
    Checks if the center of a point cloud lies inside a mesh.
    :param pc: the point cloud
    :param mesh: the mesh
    :param threshold: the threshold SDF value above which the center will be considered as in mesh
    :return: boolean
    """
    f = SDF(mesh.vertices, mesh.triangles)
    sdf = f(pc.get_center())
    return sdf > threshold


def get_point_closest_dist_to_pc(point, pcd):
    points = np.asarray(pcd.points)
    distances = np.linalg.norm(points - point, axis=1)
    min_distance = np.min(distances)
    return min_distance


def remove_sensor_pcs(pcs, b1_mesh, b2_mesh, b1_pc, b2_pc):
    pcs_filtered = []
    for pc in pcs:
        if is_pc_center_in_mesh(pc, b1_mesh, -0.02) or is_pc_center_in_mesh(pc, b2_mesh, -0.02):
            continue
        if get_point_closest_dist_to_pc(pc.get_center(), b1_pc) < 0.02 \
                or get_point_closest_dist_to_pc(pc.get_center(), b2_pc) < 0.02:
            continue
        pcs_filtered.append(pc)
    return pcs_filtered


def get_pc_xyz_color_array(pc):
    assert pc.has_colors() and pc.has_points()
    return np.concatenate([
        np.asarray(pc.points),
        np.asarray(pc.colors)
    ], axis=1)


def is_inhand_object(object_pc, b1_pc, b2_pc):
    """
    Check if a point cloud in an in-hand object.
    To do this, we check if the object point cloud center is within the bounding
    box of the two soft bubble grippers.
    Here we further assume the two bubble point clouds, as well as the
    in hand object are aligned with the world axis.
    :param object_pc: object point cloud
    :return: a boolean indicating an in-hand object or not
    """
    if not object_pc.has_points():
        return False
    box = (b1_pc + b2_pc).get_axis_aligned_bounding_box()
    max_p = box.get_max_bound()
    min_p = box.get_min_bound()
    return (object_pc.get_center() >= min_p).all() \
        and (object_pc.get_center() <= max_p).all()


def xyz_to_pc(xyz):
    if len(xyz.shape) != 2:
        import pdb; pdb.set_trace()
    assert len(xyz.shape) == 2, f"xyz should be a 2D array but got {xyz.shape}"
    xyz = xyz[:, :3]
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    return pc


def xyz_colors_to_pc(xyz_colors):
    assert len(xyz_colors.shape) == 2 and xyz_colors.shape[1] == 6
    xyz, colors = xyz_colors[:, :3], xyz_colors[:, 3:]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    pc.colors = o3d.utility.Vector3dVector(colors)

    return pc


def remove_small_pieces(point_cloud, eps, min_samples, density_check=False):
    """
    Removes small scattered pieces from an Open3D point cloud using DBSCAN clustering.

    Args:
        point_cloud (open3d.geometry.PointCloud): Open3D point cloud object.
        eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
        open3d.geometry.PointCloud: Filtered point cloud with small pieces removed.
    """
    if density_check:
        if get_bbox_points_density(point_cloud) > 3e8:
            return point_cloud

    labels = point_cloud.cluster_dbscan(eps=eps, min_points=min_samples)

    core_mask = np.asarray(labels) != -1  # Identify core points (not noise)
    filtered_cloud = point_cloud.select_by_index(np.where(core_mask)[0])

    return filtered_cloud


def get_bbox_points_density(pc):
    max_bound = pc.get_axis_aligned_bounding_box().get_max_bound()
    min_bound = pc.get_axis_aligned_bounding_box().get_min_bound()
    n_points = np.asarray(pc.points).shape[0]

    return n_points / np.prod(max_bound - min_bound)


def rgbd_feat_to_point_cloud(rgb_features, depth_map, intrinsic_mat):
    """
    Convert RGB features and depth map to a 3D point cloud.

    Args:
        rgb_features (numpy.ndarray): RGB features array with shape (height, width, feat_dim).
            feat_dim is the dimension of the feature vector.
        depth_map (numpy.ndarray): Depth map with shape (height, width).
        intrinsic_mat (numpy.ndarray): Camera intrinsic matrix (3x3).

    Returns:
        numpy.ndarray: 3D point cloud array with shape (height * width, 3 + feat_dim).
            Each row contains [x, y, z, feat_dim elements].
    """
    # Get height and width from the RGB features array
    height, width, feat_dim = rgb_features.shape

    # Get depth values from the depth map (convert back to meters from millimeters)
    depth_values = depth_map / 1000.0

    # Create a pixel grid for the image
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    pixel_grid = np.stack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))

    # Inverse projection: map pixels back to 3D coordinates
    inverse_projection = np.linalg.inv(intrinsic_mat)
    points = depth_values.flatten() * np.dot(inverse_projection, pixel_grid)

    # Reshape RGB features to match the number of pixels
    rgb_features = rgb_features.reshape(-1, feat_dim)

    # Stack the 3D coordinates with RGB features
    point_cloud = np.hstack((points.T, rgb_features))

    return point_cloud


def find_vector_in_array(array_Nk, vector_k):
    """
    Find the index of a vector within a 2D array.

    Args:
    array_Nk (numpy.ndarray): The 2D array of shape (N, k).
    vector_k (numpy.ndarray): The vector of shape (k) to search for.

    Returns:
    int: The index of the vector in the array, or -1 if not found.
    """
    array_Nk = np.array(array_Nk)
    vector_k = np.array(vector_k)

    # Check if vector_k is present in any row of array_Nk
    mask = np.all(array_Nk == vector_k, axis=1)

    if np.any(mask):
        return np.where(mask)[0][0]  # Return the index of the first match
    else:
        return -1  # Vector not found
    
    
def find_indices(arr_NK, arr_nK):
    indices = []
    for array in arr_nK:
        indices.append(find_vector_in_array(arr_NK, array))
    return indices


def is_coordinate_inside_point_cloud(point_cloud, coordinate, threshold=1e-6):
    # Convert the 3D coordinate to an Open3D point cloud with a single point
    single_point_cloud = o3d.geometry.PointCloud()
    single_point_cloud.points = o3d.utility.Vector3dVector([coordinate])

    # Compute the distance to the closest point in the point cloud
    distances = point_cloud.compute_point_cloud_distance(single_point_cloud)

    # Get the minimum distance
    min_distance = np.min(distances)

    # o3d.visualization.draw_geometries([point_cloud, single_point_cloud])

    # Check if the minimum distance is below the threshold
    return min_distance < threshold


def are_coordinates_within_point_cloud_np(point_cloud, coordinates, threshold=1e-6, return_distance=False):
    if isinstance(coordinates, list):
        coordinates = np.array(coordinates)

    if len(coordinates.shape) == 1:
        coordinates = coordinates[np.newaxis, :]

    # Extract the points from the Open3D point cloud as a NumPy array
    points = np.asarray(point_cloud.points)

    # Compute the Euclidean distances between 'coordinates' and all points
    distances = np.linalg.norm(points[:, np.newaxis, :] - coordinates, axis=2)

    # Get the minimum distances for each coordinate
    min_distances = np.min(distances, axis=0)

    # Check if the minimum distances are below the threshold for each coordinate
    results = min_distances < threshold

    if return_distance:
        return results, min_distances

    return results


def sample_points_on_line(init_pos, target_pos, N):
    # Create an array of evenly spaced values from 0 to 1
    t_values = np.linspace(0, 1, N)

    # Use linear interpolation to calculate the intermediate points
    sampled_points = (1 - t_values[:, np.newaxis]) * init_pos + t_values[:, np.newaxis] * target_pos

    return sampled_points


def center_and_rotate_to(pc, new_center, euler_to_rotate):
    """
    Translates an Open3D point cloud to a new center and rotates it by specified Euler angles.

    Parameters:
        pc (o3d.geometry.PointCloud): The input point cloud.
        new_center (list or numpy.ndarray): The new center coordinates as [x, y, z].
        euler_to_rotate (list or numpy.ndarray): Euler angles for rotation in degrees as [roll, pitch, yaw].

    Returns:
        o3d.geometry.PointCloud: A new point cloud with the transformation applied.
    """

    work_with_numpy = isinstance(pc, np.ndarray)
    if work_with_numpy:
        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(pc.copy())
        o3d_pc.colors = o3d.utility.Vector3dVector()
        pc = o3d_pc

    # Create a copy of the input point cloud
    points = np.asarray(pc.points)
    colors = np.asarray(pc.colors)

    transformed_pc = o3d.geometry.PointCloud()
    transformed_pc.points = o3d.utility.Vector3dVector(points.copy())
    transformed_pc.colors = o3d.utility.Vector3dVector(colors.copy())

    # Translate the copied point cloud to the new center
    translation_vector = np.array(new_center) - np.mean(np.asarray(transformed_pc.points), axis=0)
    transformed_pc.translate(translation_vector)

    # Define the rotation matrix based on Euler angles
    euler_angles = np.deg2rad(euler_to_rotate)
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(euler_angles)

    # Apply rotation to the copied point cloud
    transformed_pc.rotate(rotation_matrix, center=new_center)
    
    if work_with_numpy:
        return np.asarray(transformed_pc.points)
    else:
        return transformed_pc

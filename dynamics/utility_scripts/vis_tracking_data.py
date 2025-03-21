import os.path
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from utils.visualizer import play_and_save_video, from_ax_to_pil_img

import argparse
import tqdm
import matplotlib.pyplot as plt

# from perception.utils_ros import *
from perception.utils_gripper import *
from utils_general import *
from perception.utils_cv import concatenate_images_side_by_side, resize_image
from perception.utils_pc import xyz_to_pc

from PIL import Image


# def create_frames_for_pcs(pcs, multiview, view_size=5):
#     fig = plt.figure(figsize=(view_size, view_size))

#     viewing_directions = [(-90, -90), (0, -90), (None, None)]
#     if not multiview:
#         viewing_directions = viewing_directions[-1:]

#     def get_image_from_direction(points, elev, azim):
#         # Create a scatter plot
#         ax = fig.add_subplot(111, projection='3d')
#         ax.set_xlim(0.25, 0.7)
#         ax.set_ylim(-0.35, 0.35)
#         ax.set_zlim(0, 0.25)

#         n_points = 500
#         ax.view_init(elev=elev, azim=azim)

#         ax.scatter(points[:n_points * 2, 0], points[:n_points * 2, 1], points[:n_points * 2, 2],
#                    c='#d9138a', marker="o", s=10, alpha=0.4)  # c = c/r/b
#         ax.scatter(points[n_points * 2:n_points * 3, 0], points[n_points * 2:n_points * 3, 1],
#                    points[n_points * 2:n_points * 3, 2],
#                    c='#f3ca20', marker="o", s=10, alpha=0.4)  # c = c/r/b
#         ax.scatter(points[n_points * 3:, 0], points[n_points * 3:, 1], points[n_points * 3:, 2],
#                    c='#12a4d9', marker="o", s=10, alpha=0.4)  # c = c/r/b

#         # Set axis labels
#         ax.set_xlabel('x')
#         ax.set_ylabel('y')
#         ax.set_zlabel('z')

#         image = from_ax_to_pil_img(fig)
#         fig.clear()     # important; otherwise the plot will accumulate, and it becomes super slow over time

#         return image

#     frames = []
#     for pc in tqdm.tqdm(pcs):
#         points = np.asarray(pc.points)
#         images = [get_image_from_direction(points, elev, azim) for elev, azim in viewing_directions]
#         image = concatenate_images_side_by_side(*images)
#         frames.append(image)

#     return frames


def interpolate_points(points, k):
    """
    Interpolate between the given points and return the interpolated points.

    Parameters:
    - points: A numpy array of shape (n, m) where n is the number of points, and m is the dimension.
    - k: The number of points to interpolate between each pair of points.

    Returns:
    - A numpy array of shape (n * (k + 1), m) containing the interpolated points.
    """
    return points

    # Initialize an array to store the resampled points
    resampled_points = []

    for i in range(len(points) - 1):
        for j in range(k):  # Interpolate k points between each pair
            t = (j + 1) / (k + 1)  # Interpolation parameter between 0 and 1
            interpolated_point = (1 - t) * points[i] + t * points[i + 1]
            resampled_points.append(interpolated_point)

    # Add the last point
    resampled_points.append(points[-1])

    # Convert the result to a NumPy array
    resampled_points = np.array(resampled_points)

    return resampled_points

def create_structured_grid_mesh(points, n):
    """
    Create a structured grid mesh from an array of points.

    Parameters:
    - points: A numpy array of shape (n, m) where n is the number of points, and m is the dimension.
    - n: The number of points along each axis for the regular grid.

    Returns:
    - A Poly3DCollection object representing the structured grid mesh.
    """

    # Define the connectivity of the structured grid to create a mesh
    n_cells = (n - 1) ** 3
    connectivity = []
    for i in range(n_cells):
        z_idx = i // ((n - 1) ** 2)
        y_idx = (i // (n - 1)) % (n - 1)
        x_idx = i % (n - 1)

        cell = [
            x_idx + y_idx * n + z_idx * n * n,
            x_idx + 1 + y_idx * n + z_idx * n * n,
            x_idx + (y_idx + 1) * n + z_idx * n * n,
            x_idx + 1 + (y_idx + 1) * n + z_idx * n * n,
            x_idx + y_idx * n + (z_idx + 1) * n * n,
            x_idx + 1 + y_idx * n + (z_idx + 1) * n * n,
            x_idx + (y_idx + 1) * n + (z_idx + 1) * n * n,
            x_idx + 1 + (y_idx + 1) * n + (z_idx + 1) * n * n
        ]
        connectivity.append(cell)

    # Create the structured grid mesh
    mesh = Poly3DCollection(points[connectivity], alpha=0.5)

    return mesh



from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def create_frames_for_pcs(pcs, multiview, view_size=5):
    fig = plt.figure(figsize=(view_size, view_size))

    viewing_directions = [(-90, -90), (0, -90), (None, None)]
    if not multiview:
        viewing_directions = viewing_directions[-1:]

    # Choose a different colormap for more diverse colors
    colormap = plt.get_cmap('tab20')

    def get_image_from_direction(points, elev, azim):
        # Create a scatter plot
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(0.25, 0.7)
        ax.set_ylim(-0.35, 0.35)
        ax.set_zlim(0, 0.25)

        n_points = len(points)  # Use all points
        ax.view_init(elev=elev, azim=azim)

        # Assign colors from the colormap based on the point's index
        color_index = np.arange(n_points) % colormap.N
        # import pdb; pdb.set_trace()
        n = 50
        
        object_pts = interpolate_points(points[: n * 2], 3)
        
        # # print(object_pts.shape)
        ax.scatter(object_pts[:, 0], object_pts[:, 1], object_pts[:, 2],
                   c='red', marker="o", s=20, alpha=0.8)  # Use colors from the colormap
        
        # object_pts = interpolate_points(points[n * 2 : n * 3], 3)
        # # # print(object_pts.shape)
        # # triangles = Delaunay(object_pts)
        # # ax.add_collection(Poly3DCollection(object_pts[triangles.simplices], facecolors='green', alpha=0.5))
        
        # ax.scatter(object_pts[:, 0], object_pts[:, 1], object_pts[:, 2],
        #            c='green', marker="o", s=20, alpha=1)  # Use colors from the colormap
        # # mesh = create_structured_grid_mesh(object_pts, 2)
        # # ax.add_collection3d(mesh)
        
        # object_pts = interpolate_points(points[n * 3:], 3)
        # # print(object_pts.shape)
        # # triangles = Delaunay(object_pts)
        # # ax.add_collection(Poly3DCollection(object_pts[triangles.simplices], facecolors='blue', alpha=0.5))
        # # mesh = create_structured_grid_mesh(object_pts, 2)
        # # ax.add_collection3d(mesh)
        # ax.scatter(object_pts[:, 0], object_pts[:, 1], object_pts[:, 2],
        #            c='blue', marker="o", s=20, alpha=1)  # Use colors from the colormap
        
        # ax.scatter(points[:n * 2, 0], points[:n * 2, 1], points[:n * 2, 2],
        #            c=colormap(color_index[:n * 2]), marker="o", s=20, alpha=1)  # Use colors from the colormap
        # ax.scatter(points[n * 2:n * 3, 0], points[n * 2:n * 3, 1], points[n * 2:n * 3, 2],
        #            c=colormap(color_index[n * 2:n * 3]), marker="o", s=20, alpha=1)  # Use colors from the colormap
        # ax.scatter(points[n * 3:, 0], points[n * 3:, 1], points[n * 3:, 2],
        #            c=colormap(color_index[n * 3:]), marker="o", s=20, alpha=1)  # Use colors from the colormap
        
        # Split interpolate_points into chunks of size n
        
        colors = plt.get_cmap('viridis')
        
        point_chunks = [points[i:i+n] for i in range(2 * n, len(points), n)]

        # # Create a figure and axis
        # fig, ax = plt.subplots()

        # Loop through the point chunks and assign colors
        for i, chunk in enumerate(point_chunks):
            color = colors(i / len(point_chunks))
            
            ax.scatter(chunk[:, 0], chunk[:, 1], chunk[:, 2],
                       c=color, marker="o", s=20, alpha=0.6)

        # Set axis labels
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        image = from_ax_to_pil_img(fig)
        fig.clear()  # Important; otherwise, the plot will accumulate, and it becomes super slow over time

        return image

    frames = []
    for pc in tqdm.tqdm(pcs):
        points = np.asarray(pc.points)
        images = [get_image_from_direction(points, elev, azim) for elev, azim in viewing_directions]
        image = concatenate_images_side_by_side(*images)
        frames.append(image)

    return frames


def read_pc_t_from_dic(dic, t):
    pcs = o3d.geometry.PointCloud()

    # vs = [
    #     dic['bubble_pcs'][t][0],
    #     dic['bubble_pcs'][t][1],
    #     dic['inhand_object_pcs'][t]
    # ]
    # vs += [x for x in dic['object_pcs'][t]]
    vs = [v[t] for v in dic.values()]

    vs = [xyz_to_pc(x[:, :6]) for x in vs]

    for v in vs:
        pcs += v

    return pcs


import numpy as np


def main(args):
    directory_path = args.data_path  # Replace with the desired directory path
    directory_contents = get_directory_contents(directory_path)

    for path in tqdm.tqdm(directory_contents):
        if path[-3:] != ".h5":
            continue

        if 'stats.h5' in path:
            continue

        print(f'path = {path}')

        dst_file_path = path.replace(os.path.basename(path), f'{os.path.basename(path)[:-3]}_vid.mp4')
        if os.path.exists(dst_file_path):
            print(f'path {dst_file_path} exists, skipping this one')
            continue

        dic = load_dictionary_from_hdf5(path)
        # # dic = np.load(path, allow_pickle=True).item()
        # # dic = {k: np.array(v) for k, v in dic.items()}

        # # set unrelated things to zeros
        # dic['inhand_object_pcs'] = np.zeros_like(dic['inhand_object_pcs'])
        # dic['object_pcs'] = np.zeros_like(dic['object_pcs'])

        pcs = []
        for t in range(list(dic.values())[0].shape[0]):
             pcs.append(read_pc_t_from_dic(dic, t))

        # pcs = [read_pc_t_from_dic(dic, t) for t in range(list(dic.values())[0].shape[0])]
        frames_particles = create_frames_for_pcs(pcs[:args.max_nframes], multiview=True, view_size=5)
        # play_and_save_video(frames_particles, dst_file_path, args.fps)
        print(f'saved to {dst_file_path}')
        
        # pcs = [read_pc_t_from_dic(dic, t) for t in range(dic['bubble_pcs'].shape[0])]
        # frames = create_frames_for_pcs(pcs[:50], multiview=False, view_size=5)
        # combined_frames = [concatenate_images_side_by_side(x, y) for x, y in zip(frames_particles, frames_tac)]
        combined_frames = frames_particles

        # if os.path.exists(dst_file_path):
        #     print(f'skipping {dst_file_path}')
        #     continue

        play_and_save_video(combined_frames, dst_file_path, args.fps)
        print(f'saved to {dst_file_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str,
                        help="Path to directory containing the seq_x folders")
    # parser.add_argument("--every_n_frames", type=int, default=5, help="Sample a frame every x frames")
    parser.add_argument("--debug", type=int, default=0, help="Debug mode or not")
    parser.add_argument("--fps", type=int, default=5, help="number of frames per second")
    parser.add_argument("--max-nframes", type=int, default=10000, help="number of frames to visualize")
    # parser.add_argument("--target_path", type=str, default='/media/albert/ExternalHDD/bubble_data/v4_0711_parsed',
    #                     help="Directory to store parsed data")
    args = parser.parse_args()

    main(args)

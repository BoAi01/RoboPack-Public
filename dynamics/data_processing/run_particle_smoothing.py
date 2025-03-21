import os
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import h5py
from PIL import Image
import argparse
from moviepy.editor import ImageSequenceClip
from scipy.interpolate import splev, splrep
from scipy.signal import savgol_filter

def save_moviepy_gif(obs_list, name, fps=5):

    clip = ImageSequenceClip(obs_list, fps=fps)
    if name[-4:] != ".gif":
        clip.write_gif(f"{name}.gif", fps=fps)
    else:
        clip.write_gif(name, fps=fps)

def concatenate_images_side_by_side(*images):
    """
    Concatenates PIL images side by side.

    Args:
        *images (PIL.Image.Image): Variable number of PIL images.

    Returns:
        PIL.Image.Image: Concatenated image.
    """
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
def from_ax_to_pil_img(fig):
    # Draw the figure canvas
    fig.canvas.draw()

    # Convert the rendered figure to a string of RGB values
    image_data = fig.canvas.tostring_rgb()

    # Get the width and height of the figure
    width, height = fig.canvas.get_width_height()

    # Create a PIL Image object from the string of RGB values
    image = Image.frombytes('RGB', (width, height), image_data)

    return image

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
        # every 20 points should have the same color
        n = 50
        color_index = [0] * n + [1] * n + [2] * n + [3] * n

        # import pdb; pdb.set_trace()
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=colormap(color_index)[:n_points], marker="o", s=20, alpha=1)  # Use colors from the colormap

        # Set axis labels
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        image = from_ax_to_pil_img(fig)
        fig.clear()  # Important; otherwise, the plot will accumulate, and it becomes super slow over time

        return image

    frames = []
    for pc in tqdm.tqdm(pcs):
        if not isinstance(pc, np.ndarray):
            points = np.asarray(pc.points)
        else:
            points = pc
        images = [get_image_from_direction(points, elev, azim) for elev, azim in viewing_directions]
        image = concatenate_images_side_by_side(*images)
        # convert PIL image to numpy array
        image = np.asarray(image)
        frames.append(image)

    return frames


def smooth(points, s):
    # points: [time]

    # spl = splrep(np.arange(len(points)), points, s=s)
    # y2 = splev(np.arange(len(points)), spl)
    y2 = savgol_filter(points, 15, 3)
    return y2


def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


def smooth_points(all_points, s):
    # all_points: [time, points, xyz]
    # first let's try just smoothing each point individually, and each dimension individually
    print("Applying smoothing to all points...")
    all_points_smoothed = all_points.copy()
    for i in range(all_points.shape[1]):
        for j in range(all_points.shape[2]):
            all_points_smoothed[:, i, j] = smooth(all_points[:, i, j], s)
    # make the transformations rigid again
    # first, find the rigid transformations that align the points
    # then apply that transformation to the smoothed points
    # estimate the rigid transformation
    print("Estimating rigid transformation...")
    smoothed_rigid_all_points = all_points_smoothed.copy()
    for obj_idx in range(all_points_smoothed.shape[1], 50):
        for i in range(all_points_smoothed.shape[0]-1):
            R, t = rigid_transform_3D(smoothed_rigid_all_points[i, obj_idx*50:(obj_idx+1)*50, :].T, all_points_smoothed[i+1, obj_idx*50:(obj_idx+1)*50, :].T)
            smoothed_rigid_all_points[i+1, obj_idx*50:(obj_idx+1)*50, :] = (R @ smoothed_rigid_all_points[i, obj_idx*50:(obj_idx+1)*50, :].T).T + t.T
    return smoothed_rigid_all_points


def smooth_tracked_points_from_h5(h5_file, output_folder):
    # get the filename from the h5 file
    filename = h5_file.split('/')[-1]
    print("Smoothing tracked points from file", h5_file)
    with h5py.File(h5_file, 'r') as f:
        with h5py.File(os.path.join(output_folder, filename), 'w') as f_out:
            keys = list(f.keys())
            for key in keys:
                if np.any(f[key][()][:, :, 2] > 0.6):
                    print(f"Key {key} in file {h5_file} has z values greater than 0.6")
                    # update time to take to be the first time that z is greater than 0.6 minus 50
                    # time_to_take = min(time_to_take, np.where(f[key][()][:, :, 2] > 0.6)[0][0] - 50)
                    return
                if np.any(f[key][()][:, :, 0] > 0.8):
                    print(f"Key {key} in file {h5_file} has x values greater than 0.8")
                    # time_to_take = min(time_to_take, np.where(f[key][()][:, :, 0] > 0.8)[0][0] - 50)
                    return
            for key in keys:
                f_out.create_dataset(key, data=smooth_points(f[key][()], s=0))
    print("Done!")


def main():
    args = argparse.ArgumentParser("Smooth tracked points from h5 files")
    args.add_argument("--input_folder", type=str)
    args.add_argument("--output_folder", type=str)
    args = args.parse_args()
    # if output folder doesn't exist, create it
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    for root, dirs, files in os.walk(args.input_folder):
        for file in files:
            if file.endswith(".h5"):
                smooth_tracked_points_from_h5(os.path.join(root, file), args.output_folder)


def test_smoothing():
    time_length = 500

    with h5py.File("/svl/u/stian/softbubble_data/outputs/09_22_box3/seq_24.h5", 'r') as f:
        pole_points = f['blue pole'][()]
        print(pole_points.shape)
        blue_pole = f['blue pole'][()]
        print(blue_pole.shape)
        box_points = f['carton package'][()]
        # combine the two point clouds
        all_points = np.concatenate((box_points, blue_pole), axis=1)
    all_points = all_points[:time_length]
    s = time_length - np.sqrt(2*time_length)
    s = 0.0005
    smoothed_points = smooth_points(all_points, s=s)
    vis_pole = create_frames_for_pcs(all_points, multiview=True)
    vis_smoothed = create_frames_for_pcs(smoothed_points, multiview=True)

    # combine the two visualizations stacked vertically
    vis = []
    for i in range(len(vis_pole)):
        vis.append(np.concatenate((vis_pole[i], vis_smoothed[i]), axis=0))

    save_moviepy_gif(vis, "pole.gif", fps=20)


def visualize_point_distribution():
    all_points = []
    for i in range(1, 25):
        try:
            with h5py.File(f"/svl/u/stian/softbubble_data/outputs/09_22_box3/seq_{i}.h5", 'r') as f:
                keys = list(f.keys())
                for key in keys:
                    points = f[key][()]
                    if np.any(points[:, :, 2] > 0.6):
                        print(f"Key {key} in file {i} has z values greater than 0.6")
                        continue
                    if np.any(points[:, :, 0] > 0.9):
                        print(f"Key {key} in file {i} has x values greater than 0.6")
                        continue
                    all_points.append(points)
                    # if any points have z values greater than 0.4, print the file

        except:
            print(f"Failed to load file {i}")
    all_points = np.concatenate(all_points, axis=0)
    # flatten the points
    all_points = all_points.reshape((-1, 3))
    # scatter plot the points in 3d
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(all_points[:, 0].min(), all_points[:, 0].max())
    ax.set_ylim(all_points[:, 1].min(), all_points[:, 1].max())
    ax.set_zlim(all_points[:, 2].min(), all_points[:, 2].max())
    # print all limits
    print(f"X limits: {ax.get_xlim()}")
    print(f"Y limits: {ax.get_ylim()}")
    print(f"Z limits: {ax.get_zlim()}")
    ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2], marker="o", s=20, alpha=1)  # Use colors from the colormap

    # write the histograms to a file
    plt.savefig("point_distribution.png")


if __name__ == "__main__":
    # visualize_point_distribution()
    # test_smoothing()
    main()
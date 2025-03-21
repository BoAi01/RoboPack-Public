import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tqdm
import logging
logging.getLogger("moviepy").setLevel(logging.ERROR)  # Set to ERROR to suppress most output


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


def save_moviepy(obs_list, name, fps=5):
    from moviepy.editor import ImageSequenceClip
    clip = ImageSequenceClip(obs_list, fps=fps)
    if name[:-4] != ".gif":
        clip.write_videofile(name, fps=fps, verbose=False)
    else:
        clip.write_gif(name, fps=fps, verbose=False)


def create_frames_for_pcs(pcs, goal, multiview, view_size=5, title=None, extra_title_func=None, hide_tqdm=False, observed_his_len=20,
                          action_history=None, action=None):
    fig = plt.figure(figsize=(view_size, view_size))
    # print(action)
    # print(action_history)

    viewing_directions = [(90, -90), (0, 0), (None, None)]
    if not multiview:
        viewing_directions = [viewing_directions[0]]

    # # Choose a different colormap for more diverse colors
    # colormap = plt.get_cmap('tab20')

    # def status_str_lambda(x):
    #     if x == 0:
    #         return "History.."  # no action to show
    #     elif x < observed_his_len:
    #         try:
    #             return f"History.. Action={[f'{num:.3f}' for num in action_history[0, -1, x - 1]]}"  # show the action in history
    #         except:
    #             breakpoint()
    #     else:
    #         return f"Planning... Action = {[f'{num:.3f}' for num in action[x - observed_his_len]]}"  # show the planned action

    def get_image_from_direction(points, elev, azim, frame_idx):
        # import time
        # start_time = time.time()
        
        # Create a scatter plot
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(0.25, 0.7)
        ax.set_ylim(-0.35, 0.35)
        ax.set_zlim(0, 0.25)

        n_points = len(points)  # Use all points
        ax.view_init(elev=elev, azim=azim)

        # # Assign colors from the colormap based on the point's index
        # color_index = np.arange(20) % colormap.N
        # n = 20
        # ax.scatter(points[n * 0:n * 1, 0], points[n * 0:n * 1, 1], points[n * 0:n * 1, 2],
        #            c="g", marker="o", s=60, alpha=1)  # Use colors from the colormap
        # ax.scatter(points[n * 1:n * 2, 0], points[n * 1:n * 2, 1], points[n * 1:n * 2, 2],
        #            c="b", marker="o", s=60, alpha=1)  # Use colors from the colormap
        # ax.scatter(points[n * 2:n * 3, 0], points[n * 2:n * 3, 1], points[n * 2:n * 3, 2],
        #            c="r", marker="o", s=60, alpha=0.33)  # Use colors from the colormap
        # ax.scatter(points[n * 3:, 0], points[n * 3:, 1], points[n * 3:, 2],
        #            c="r", marker="o", s=60, alpha=0.33)  # Use colors from the colormap

        n = 20
        point_size = 60
        
        bubble_points = points[-n * 2:]
        ax.scatter(bubble_points[:, 0], bubble_points[:, 1], bubble_points[:, 2],
                   c='red', marker="o", s=point_size, alpha=0.9)  # Use colors from the colormap
        
        colors = plt.get_cmap('viridis')
        
        point_chunks = [points[i:i+n] for i in range(0, len(points) - 2 * n, n)]

        # Loop through the point chunks and assign colors
        for i, chunk in enumerate(point_chunks):
            color = colors(i / len(point_chunks))
            
            ax.scatter(chunk[:, 0], chunk[:, 1], chunk[:, 2],
                       c=color, marker="o", s=point_size, alpha=0.9)
        
        # plot the goal as points
        ax.scatter(goal[:, 0], goal[:, 1], goal[:, 2], c='black', marker="o", s=point_size, alpha=0.05)
        
        # Set axis labels
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        # Set title
        if title is not None:
            if extra_title_func is not None:
                full_title = title + f"\n{extra_title_func(frame_idx)}"
                ax.set_title(full_title)
            else:
                ax.set_title(title)
        # print("Time to plot: ", time.time() - start_time)
        # start_time = time.time()
        image = from_ax_to_pil_img(fig)
        # print(f'time to convert to PIL: {time.time() - start_time}')
        
        fig.clear()  # Important; otherwise, the plot will accumulate, and it becomes super slow over time
        
        return image

    frames = []
    if hide_tqdm:
        for pc in pcs:
            points = np.asarray(pc)
            images = [get_image_from_direction(points, elev, azim, len(frames)) for elev, azim in viewing_directions]
            image = concatenate_images_side_by_side(*images)
            frames.append(image)
    else:
        for pc in tqdm.tqdm(pcs):
            points = np.asarray(pc)
            images = [get_image_from_direction(points, elev, azim, len(frames)) for elev, azim in viewing_directions]
            image = concatenate_images_side_by_side(*images)
            frames.append(image)
    return frames

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


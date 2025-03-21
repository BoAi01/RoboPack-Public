import os
import pdb
import pickle
import subprocess
import sys
import imageio
from datetime import datetime

import cv2
import h5py
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy
import numpy as np
import open3d as o3d
import pandas as pd
import seaborn as sns
from PIL import Image

from sklearn import metrics

from utils.macros import *


matplotlib.rcParams["legend.loc"] = "upper left"


def plot_pereception_results(df, fig_res=8, path=""):
    sns.set_theme(style="darkgrid")
    sns.set(font_scale=2, rc={"lines.linewidth": 3})

    y_list = ["pred_loss", "emd"]
    hue_list = [
        "motion_field.n_samples",
        "motion_field.data_loader.max_freq",
        "motion_field.MLP.D",
        "motion_field.MLP.W",
    ]

    for y in y_list:
        df.loc[df[y] == 0, y] = np.nan

    n_rows = len(y_list)
    n_cols = len(hue_list)
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(fig_res * n_cols, fig_res * n_rows)
    )

    for i in range(n_rows):
        for j in range(n_cols):
            # Plot 1: Pred_loss vs Frame_idx
            sns.lineplot(
                x="frame_idx",
                y=y_list[i],
                hue=hue_list[j],
                data=df,
                ax=axs[i, j],
            )
            axs[i, j].set_title(f"{y_list[i]} vs frame_idx")
            axs[i, j].legend(title=hue_list[j].split(".")[-1])

    plt.tight_layout()

    if len(path) > 0:
        plt.savefig(path)
    else:
        plt.show()

    plt.close()


def visualize_points(config, ax, state, target, render, obj_color=None):
    """
    Plot the points in state and target using ax with the specified config parameters.
    :param config: configuration parameters
    :param ax: ax object from fig
    :param state: a dictionary containing the point of objects
    :param target: ground-truth state
    :return: a dictionary of ax figures
    """
    
    # use different color for each object
    colors = plt.get_cmap('viridis')

    ax.computed_zorder = config["pc_zorder"] is None
    output = {}
    for idx, (key, value) in enumerate(state.items()):
        if not render:
            continue
        if key in config["pc_visible"]:
            if obj_color is not None and key == "object_obs":
                points_color = obj_color
            else:
                points_color = config["pc_color"][key][0] if key in config["pc_color"] else "b"
            
            pc = None
            if key != "object_obs":
                pc = ax.scatter(
                    value[:, 0],
                    value[:, 1],
                    value[:, 2],
                    c=points_color,
                    alpha=config["pc_color"][key][1],
                    s=config["point_size"],
                    zorder=None if ax.computed_zorder else config["z_order"][key],
                )
            else:
                colors = plt.get_cmap('viridis')
                n = 20      # num of points per object
                point_chunks = [value[i:i+n] for i in range(0, len(value), n)]
                # Loop through the point chunks and assign colors
                for i, chunk in enumerate(point_chunks):
                    color = colors(i / len(point_chunks))
                    pc = ax.scatter(chunk[:, 0], chunk[:, 1], chunk[:, 2],
                                c=color, marker="o", s=config["point_size"], 
                                alpha=config["pc_color"][key][1])

            # use random color for each point 
            # colormap = plt.get_cmap('tab20')
            # color_index = np.arange(len(value)) % colormap.N

            # pc = ax.scatter(
            #     value[:, 0],
            #     value[:, 1],
            #     value[:, 2],
            #     c=colormap(color_index),
            #     s=config["point_size"],
            #     zorder=None if ax.computed_zorder else config["z_order"][key],
            # )

            output[key] = pc

    if target is not None:
        ax.scatter(
            target[:, 0],
            target[:, 1],
            target[:, 2],
            c=config["pc_color"]["target"][0],
            alpha=config["pc_color"]["target"][1],
            s=config["point_size"],
        )

    if not config["axis_on"]:
        ax.axis("off")

    if not config["axis_ticklabel_on"]:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    if config["object_centric"]:
        center = np.mean(state["object"], axis=0)
        scale = (
            np.max(np.max(state["object"], axis=0) - np.min(state["object"], axis=0))
            / 2
        )
    else:
        center = config["vis_center"]
        scale = config["vis_scale"]

    for ctr, dim in zip(center, "xyz"):
        getattr(ax, "set_{}lim".format(dim))(ctr - scale, ctr + scale)

    return output


def render_anim_async(path, **kwargs):
    pkl_path = f"{path}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(kwargs, f)

    p = subprocess.Popen(
        ["python", os.path.join(SCRIPT_DIR, "utils", "run_visualizer.py"), pkl_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    return p


def visualize_object_class_pred(config, batch_pred, path=None):
    # print(f'Plotting... skip the first {num_skip_frames} frames of predictions')
    states, predictions = batch_pred

    from utils_general import break_trajectory_dic_into_sequence
    # states = break_trajectory_dic_into_sequence(batch_data)
    predictions = [x.argmax(dim=1).cpu().item() for x in predictions]
    seq_len = len(states) - len(predictions)

    H = len(states)
    R = 1
    views = config["views"][-1:]
    C = len(views)

    images = []

    def plot_one_frame(step):
        fig, big_axes = plt.subplots(
            R, 1, figsize=(config["subfigsize"] * C, config["subfigsize"] * R)
        )

        plot_info_dict = {}
        for i in range(R):
            # target_cur = target[i] if isinstance(target, list) else target

            if R == 1:
                ax_cur = big_axes
            else:
                ax_cur = big_axes[i]

            # ax_cur.set_title(
            #     title_list[i], fontweight="semibold", fontsize=config["title_fontsize"]
            # )
            ax_cur.axis("off")

            plot_info = []
            for j in range(C):
                ax = fig.add_subplot(R, C, i * C + j + 1, projection="3d")
                ax.view_init(*views[j])

                # ax.set_title(
                #     views[j],
                #     fontweight="semibold",
                #     fontsize=config["subtitle_fontsize"],
                #     loc="left",
                #     y=0.0,
                # )

                object_color_dict = {
                    0: 'gold',
                    1: 'turquoise',
                    2: 'slateblue',
                    3: 'grey'
                }
                legend_labels = ['Solid box', '500g @ corner', '200g, 50g @ 2 corners', 'Unlabeled']

                ax.set_title("GT: " + legend_labels[int(states[step]['object_cls'])],
                             fontweight="semibold", fontsize=config["title_fontsize"] // 2,
                             loc="center")

                obj_color = object_color_dict[predictions[step - seq_len]] if step - seq_len > 0 else 'grey'
                visualize_points(config, ax, states[step], None, True, obj_color)

                legend_handles = [
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                               markersize=8, label=label) for
                    color, label in zip(object_color_dict.values(), legend_labels)]
                legend = ax.legend(handles=legend_handles, loc='upper left', fontsize='small')

        images.append(from_ax_to_pil_img(fig))
        fig.clear()

    for i in range(H):
        plot_one_frame(i)

    if path:
        fps = 1
        play_and_save_video(images, output_file=str(path) + '.mp4', fps=fps, pause_time_last_frame=0)
        images[0].save(str(path) + '.gif', save_all=True, append_images=images[1:], loop=0, duration=int(1000 / fps))

    plt.close()


def visualize_object_class_pred(config, batch_pred, path=None):
    # print(f'Plotting... skip the first {num_skip_frames} frames of predictions')
    states, predictions = batch_pred
    from utils_general import break_trajectory_dic_into_sequence
    import pdb
    pdb.set_trace()

    # states = break_trajectory_dic_into_sequence(batch_data)
    predictions = predictions.argmax(dim=1).cpu().numpy()
    seq_len = len(states) - len(predictions)

    H = len(states)
    R = 1
    views = config["views"][-1:]
    C = len(views)

    images = []

    def plot_one_frame(step):
        fig, big_axes = plt.subplots(
            R, 1, figsize=(config["subfigsize"] * C, config["subfigsize"] * R)
        )

        plot_info_dict = {}
        for i in range(R):
            # target_cur = target[i] if isinstance(target, list) else target

            if R == 1:
                ax_cur = big_axes
            else:
                ax_cur = big_axes[i]

            # ax_cur.set_title(
            #     title_list[i], fontweight="semibold", fontsize=config["title_fontsize"]
            # )
            ax_cur.axis("off")

            plot_info = []
            for j in range(C):
                ax = fig.add_subplot(R, C, i * C + j + 1, projection="3d")
                ax.view_init(*views[j])
                ax.set_title(
                    views[j],
                    fontweight="semibold",
                    fontsize=config["subtitle_fontsize"],
                    loc="left",
                    y=0.0,
                )

                object_color_dict = {
                    0: 'gold',
                    1: 'turquoise',
                    2: 'slateblue'
                }

                obj_color = object_color_dict[predictions[step - seq_len]] if step - seq_len > 0 else 'grey'
                visualize_points(config, ax, states[step], None, True, obj_color)

        images.append(from_ax_to_pil_img(fig))
        fig.clear()

    for i in range(H):
        plot_one_frame(i)

    if path:
        fps = 2
        play_and_save_video(images, output_file=str(path) + '.mp4', fps=fps)
        images[0].save(str(path) + '.gif', save_all=True, append_images=images[1:], loop=0, duration=int(1000 / fps))

    plt.close()


def visualize_pred_gt_pos(config, title_list, pred_gt_pos_seqs, target=None, path=None, num_skip_frames=0):
    print(f'Plotting... skip the first {num_skip_frames} frames of predictions') 

    H = max([len(x) for x in pred_gt_pos_seqs])
    R = len(title_list)
    C = len(config["views"])

    images = []
    def plot_one_frame(step):
        fig, big_axes = plt.subplots(
            R, 1, figsize=(config["subfigsize"] * C, config["subfigsize"] * R)
        )

        plot_info_dict = {}
        for i in range(R):
            target_cur = target[i] if isinstance(target, list) else target

            if R == 1:
                ax_cur = big_axes
            else:
                ax_cur = big_axes[i]

            ax_cur.set_title(
                title_list[i], fontweight="semibold", fontsize=config["title_fontsize"]
            )
            ax_cur.axis("off")

            plot_info = []
            for j in range(C):
                ax = fig.add_subplot(R, C, i * C + j + 1, projection="3d")
                ax.view_init(*config["views"][j])
                ax.set_title(
                    config["view_names"][j],
                    fontweight="semibold",
                    fontsize=config["subtitle_fontsize"],
                    loc="left",
                    y=0.0,
                )
                
                if i == 0 and step < num_skip_frames:  # skip render prediction 
                    continue
                else:
                    output = visualize_points(config, ax, pred_gt_pos_seqs[i][step], target_cur, True)

        images.append(from_ax_to_pil_img(fig))
        fig.clear()

    for i in range(H):
        plot_one_frame(i)

    if path:
        fps = 2
        play_and_save_video(images, output_file=str(path)+'.mp4', fps=fps)
        images[0].save(str(path)+'.gif', save_all=True, append_images=images[1:], loop=0, duration=int(1000 / fps))

    plt.close()


def visualize_pred_gt_pos_simple(config, title_list, pred_gt_pos_seqs, target=None, path=None, num_skip_frames=0, multiview=False):
    print(f'Plotting... skip the first {num_skip_frames} frames of predictions')

    H = max([len(x) for x in pred_gt_pos_seqs])
    R = len(title_list)
    views = config["views"][:1] if not multiview else [config["views"][0], config["views"][2]]  # only use the first and the third view for multivew vis
    C = len(views)

    images = []

    def plot_one_frame(step):
        fig, big_axes = plt.subplots(
            R, 1, figsize=(config["subfigsize"] * C, config["subfigsize"] * R)
        )

        plot_info_dict = {}
        for i in range(R):
            target_cur = target[i] if isinstance(target, list) else target

            if R == 1:
                ax_cur = big_axes
            else:
                ax_cur = big_axes[i]

            ax_cur.set_title(
                title_list[i], fontweight="semibold", fontsize=config["title_fontsize"] // 2
            )
            ax_cur.axis("off")

            plot_info = []
            for j in range(C):
                ax = fig.add_subplot(R, C, i * C + j + 1, projection="3d")
                ax.view_init(*views[j])
                # ax.set_title(
                #     views[j],
                #     fontweight="semibold",
                #     fontsize=config["subtitle_fontsize"] // 2,
                #     loc="left",
                #     y=0.0,
                # )
                if i == 0:  
                    if step < num_skip_frames:  # skip render prediction
                        visualize_points(config, ax, pred_gt_pos_seqs[i][step], target_cur, False)
                    else:
                        visualize_points(config, ax, pred_gt_pos_seqs[i][step], target_cur, True)
                else:
                    visualize_points(config, ax, pred_gt_pos_seqs[i][step], target_cur, True)

        images.append(from_ax_to_pil_img(fig))
        fig.clear()

    for i in range(H):
        plot_one_frame(i)

    if path:
        fps = 2
        play_and_save_video(images, output_file=str(path) + '.mp4', fps=fps, pause_time_last_frame=1)
        # images[0].save(str(path) + '.gif', save_all=True, append_images=images[1:], loop=0, duration=int(1000 / fps))

    plt.close()
    return images


def visualize_o3d(
    geometry_list,
    title="O3D",
    view_point=None,
    point_size=5,
    pcd_color=[0, 0, 0],
    mesh_color=[0.5, 0.5, 0.5],
    show_normal=False,
    show_frame=True,
    path="",
):
    vis = o3d.visualization.Visualizer()
    vis.create_window(title)
    types = []

    for geometry in geometry_list:
        type = geometry.get_geometry_type()
        # Point Cloud
        # if type == o3d.geometry.Geometry.Type.PointCloud:
        #     geometry.paint_uniform_color(pcd_color)
        # Triangle Mesh
        if type == o3d.geometry.Geometry.Type.TriangleMesh:
            geometry.paint_uniform_color(mesh_color)
        types.append(type)

        vis.add_geometry(geometry)
        vis.update_geometry(geometry)

    vis.get_render_option().background_color = np.array([0.0, 0.0, 0.0])
    if show_frame:
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
        vis.add_geometry(mesh)
        vis.update_geometry(mesh)

    if o3d.geometry.Geometry.Type.PointCloud in types:
        vis.get_render_option().point_size = point_size
        vis.get_render_option().point_show_normal = show_normal
    if o3d.geometry.Geometry.Type.TriangleMesh in types:
        vis.get_render_option().mesh_show_back_face = True
        vis.get_render_option().mesh_show_wireframe = True

    vis.poll_events()
    vis.update_renderer()

    if view_point is None:
        vis.get_view_control().set_front(np.array([0.305, -0.463, 0.832]))
        vis.get_view_control().set_lookat(np.array([0.4, -0.1, 0.0]))
        vis.get_view_control().set_up(np.array([-0.560, 0.620, 0.550]))
        vis.get_view_control().set_zoom(0.2)
    else:
        vis.get_view_control().set_front(view_point["front"])
        vis.get_view_control().set_lookat(view_point["lookat"])
        vis.get_view_control().set_up(view_point["up"])
        vis.get_view_control().set_zoom(view_point["zoom"])

    if len(path) > 0:
        vis.capture_screen_image(path, True)
        vis.destroy_window()
    else:
        vis.run()


def visualize_networkx(config, nx_graph, point_cloud, tool_offset, path=""):
    fig = plt.figure(figsize=(config["subfigsize"] * 2, config["subfigsize"] * 2))

    # specify the number of rows, columns, and index of the subplot to create
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    # set unit distance to be the same for all axis
    ax.axis('equal')

    # Set labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # set axis range
    ax.set_xlim(0.3, 0.7)
    ax.set_ylim(-0.3, +0.3)
    ax.set_zlim(0, 0.2)

    # Draw nodes
    for key in nx_graph.nodes:
        type = "object_obs" if key < tool_offset else "tool"
        if type == "tool":
            # print('tool points are skipped')
            continue    # skip tool points
        ax.scatter(
            point_cloud[key, 0],
            point_cloud[key, 1],
            point_cloud[key, 2],
            c=config["pc_color"][type][0],
            alpha=config["pc_color"][type][1],
            s=config["point_size"],
        )

    # Draw edges
    for edge in nx_graph.edges:
        # if edges are within an object, do not visualize them 
        if (edge[0] < 20 and edge[1] < 20) or (20 < edge[0] < 40 and 20 < edge[1] < 40) or (edge[0] > 40 and edge[1] > 40):
            continue 
        # if edge[0] < tool_offset and edge[1] < tool_offset:
        #     # c = config["pc_color"]["object"][0]
        #     continue 
        # elif edge[0] >= tool_offset and edge[1] >= tool_offset:
        #     c = config["pc_color"]["tool"][0]
        c = "b"
        p1, p2 = point_cloud[edge[0]], point_cloud[edge[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c=c, linewidth=config["point_size"]/20, alpha=0.5)

    if len(path) > 0:
        plt.savefig(path)
    else:
        plt.show()
        pdb.set_trace()
        plt.close()


def plot_train_loss(csv_file, path=""):
    sns.set_theme(style="darkgrid")
    sns.set(font_scale=2, rc={"lines.linewidth": 3})

    df = pd.read_csv(csv_file)

    fig, axs = plt.subplots(2, 2, figsize=(20, 10))

    # Training loss per step
    sns.lineplot(data=df, x="step", y="train_loss_step", ax=axs[0, 0])
    axs[0, 0].set_title("Training Loss vs Steps")

    # Training loss per epoch
    sns.lineplot(data=df, x="epoch", y="train_loss_epoch", ax=axs[0, 1])
    axs[0, 1].set_title("Training Loss vs Epochs")

    # Validation loss per step
    sns.lineplot(data=df, x="step", y="val_loss_step", ax=axs[1, 0])
    axs[1, 0].set_title("Validation Loss vs Steps")

    # Validation loss per epoch
    sns.lineplot(data=df, x="epoch", y="val_loss_epoch", ax=axs[1, 1])
    axs[1, 1].set_title("Validation Loss vs Epochs")

    plt.tight_layout()

    if len(path) > 0:
        plt.savefig(path)
    else:
        plt.show()

    plt.close()


def plot_eval_loss(values, metrics, path=""):
    fig, axs = plt.subplots(len(metrics), 1, figsize=(10, 15))

    # Visualization
    for i in range(len(metrics)):
        sns.lineplot(
            ax=axs[i],
            x=range(len(values[metrics[i]]["mean"])),
            y=values[metrics[i]]["mean"],
            label=metrics[i],
        )
        axs[i].errorbar(
            range(len(values[metrics[i]]["mean"])),
            values[metrics[i]]["mean"],
            yerr=values[metrics[i]]["std"],
            fmt="-o",
        )
        axs[i].set_xlabel("Frames")
        axs[i].set_ylabel(metrics[i])
        axs[i].set_title(f"{metrics[i]} vs Frames")

    if len(path) > 0:
        plt.savefig(path)
    else:
        plt.show()

    plt.close()


def render_frames(
    args,
    row_titles,
    state_seq,
    frame_list=[],
    axis_off=True,
    focus=True,
    draw_set=["object_obs", "inhand", "bubble"],
    target=None,
    views=[(90, -90), (0, -90), (45, -45)],
    res="high",
    path="",
    name="",
):
    n_frames = state_seq[0].shape[0]
    n_rows = len(row_titles)
    n_cols = len(views)

    figres = 12 if res == "high" else 3
    title_fontsize = 60 if res == "high" else 10
    fig, big_axes = plt.subplots(n_rows, 1, figsize=(figres * n_cols, figres * n_rows))

    if len(frame_list) == 0:
        frame_list = range(n_frames)

    for frame in frame_list:
        for i in range(n_rows):
            state = state_seq[i]
            target_cur = target[i] if isinstance(target, list) else target
            focus_cur = focus[i] if isinstance(focus, list) else focus
            if n_rows == 1:
                big_axes.set_title(
                    row_titles[i], fontweight="semibold", fontsize=title_fontsize
                )
                big_axes.axis("off")
            else:
                big_axes[i].set_title(
                    row_titles[i], fontweight="semibold", fontsize=title_fontsize
                )
                big_axes[i].axis("off")

            for j in range(n_cols):
                ax = fig.add_subplot(
                    n_rows, n_cols, i * n_cols + j + 1, projection="3d"
                )
                ax.view_init(*views[j])
                visualize_points(
                    ax,
                    args,
                    state[frame],
                    draw_set,
                    target_cur,
                    axis_off=axis_off,
                    focus=focus_cur,
                    res=res,
                )

        # plt.tight_layout()

        if len(path) > 0:
            if len(name) == 0:
                plt.savefig(os.path.join(path, f"{str(frame).zfill(3)}.pdf"))
            else:
                plt.savefig(os.path.join(path, name))
        else:
            plt.show()

    plt.close()


def render_o3d(
    geometry_list,
    axis_off=False,
    focus=True,
    views=[(90, -90), (0, -90), (45, -45)],
    label_list=[],
    point_size_list=[],
    path="",
):
    n_rows = 2
    n_cols = 3

    fig, big_axes = plt.subplots(n_rows, 1, figsize=(12 * n_cols, 12 * n_rows))

    for i in range(n_rows):
        ax_cur = big_axes[i]

        title_fontsize = 60
        ax_cur.set_title("Test", fontweight="semibold", fontsize=title_fontsize)
        ax_cur.axis("off")

        for j in range(n_cols):
            ax = fig.add_subplot(n_rows, n_cols, i * n_cols + j + 1, projection="3d")
            ax.computed_zorder = False
            ax.view_init(*views[j])

            for k in range(len(geometry_list)):
                type = geometry_list[k].get_geometry_type()
                # Point Cloud
                # if type == o3d.geometry.Geometry.Type.PointCloud:
                #     geometry.paint_uniform_color(pcd_color)
                # Triangle Mesh
                if type == o3d.geometry.Geometry.Type.TriangleMesh:
                    import pymeshfix
                    mf = pymeshfix.MeshFix(
                        np.asarray(geometry_list[k].vertices),
                        np.asarray(geometry_list[k].triangles),
                    )
                    mf.repair()
                    mesh = mf.mesh
                    vertices = np.asarray(mesh.points)
                    triangles = np.asarray(mesh.faces).reshape(mesh.n_faces, -1)[:, 1:]
                    ax.plot_trisurf(
                        vertices[:, 0],
                        vertices[:, 1],
                        triangles=triangles,
                        Z=vertices[:, 2],
                    )
                    # ax.set_aspect('equal')
                elif type == o3d.geometry.Geometry.Type.PointCloud:
                    particles = np.asarray(geometry_list[k].points)
                    colors = np.asarray(geometry_list[k].colors)
                    if len(point_size_list) > 0:
                        point_size = point_size_list[k]
                    else:
                        point_size = 160
                    if len(label_list) > 0:
                        label = label_list[k]
                        if "object_obs" in label:
                            ax.scatter(
                                particles[:, 0],
                                particles[:, 1],
                                particles[:, 2],
                                c="b",
                                s=point_size,
                                label=label,
                            )
                        elif "bubble" in label:
                            ax.scatter(
                                particles[:, 0],
                                particles[:, 1],
                                particles[:, 2],
                                c="r",
                                alpha=0.2,
                                zorder=4.2,
                                s=point_size,
                                label=label,
                            )
                        else:
                            ax.scatter(
                                particles[:, 0],
                                particles[:, 1],
                                particles[:, 2],
                                c="yellowgreen",
                                zorder=4.1,
                                s=point_size,
                                label=label,
                            )
                    else:
                        label = None
                        ax.scatter(
                            particles[:, 0],
                            particles[:, 1],
                            particles[:, 2],
                            c=colors,
                            s=point_size,
                            label=label,
                        )
                else:
                    raise NotImplementedError

            if axis_off:
                ax.axis("off")

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

            if len(label_list) > 0:
                ax.legend(fontsize=30, loc="upper right", bbox_to_anchor=(0.0, 0.0))

            # extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
            # size = extents[:, 1] - extents[:, 0]
            centers = geometry_list[0].get_center()
            if focus:
                r = 0.05
                for ctr, dim in zip(centers, "xyz"):
                    getattr(ax, "set_{}lim".format(dim))(ctr - r, ctr + r)

    # plt.tight_layout()

    if len(path) > 0:
        plt.savefig(f'{path}_{datetime.now().strftime("%b-%d-%H:%M:%S")}.png')
    else:
        plt.show()

    plt.close()


def visualize_target(args, target_shape_name):
    target_frame_path = os.path.join(
        os.getcwd(),
        "target_shapes",
        target_shape_name,
        f'{target_shape_name.split("/")[-1]}.h5',
    )
    visualize_h5(args, target_frame_path)


def visualize_h5(args, file_path):
    hf = h5py.File(file_path, "r")
    data = []
    for i in range(len(args.data_names)):
        d = np.array(hf.get(args.data_names[i]))
        data.append(d)
    hf.close()
    target_shape = data[0][: args.n_particles, :]
    render_frames(args, ["H5"], [np.array([target_shape])], draw_set=["object_obs"])


def visualize_neighbors(args, particles, target, neighbors, path=""):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # red is the target and blue are the neighbors
    ax.scatter(
        particles[: args.n_particles, args.axes[0]],
        particles[: args.n_particles, args.axes[1]],
        particles[: args.n_particles, args.axes[2]],
        c="c",
        alpha=0.2,
        s=30,
    )
    ax.scatter(
        particles[args.n_particles :, args.axes[0]],
        particles[args.n_particles :, args.axes[1]],
        particles[args.n_particles :, args.axes[2]],
        c="r",
        alpha=0.2,
        s=30,
    )

    ax.scatter(
        particles[neighbors, args.axes[0]],
        particles[neighbors, args.axes[1]],
        particles[neighbors, args.axes[2]],
        c="b",
        s=60,
    )
    ax.scatter(
        particles[target, args.axes[0]],
        particles[target, args.axes[1]],
        particles[target, args.axes[2]],
        c="r",
        s=60,
    )

    extents = np.array([getattr(ax, "get_{}lim".format(dim))() for dim in "xyz"])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, "xyz"):
        getattr(ax, "set_{}lim".format(dim))(ctr - r, ctr + r)

    if len(path) > 0:
        plt.savefig(path)
    else:
        plt.show()

    plt.close()


def plot_cm(test_set, y_true, y_pred, path=""):
    confusion_matrix = metrics.confusion_matrix(
        [test_set.classes[x] for x in y_true],
        [test_set.classes[x] for x in y_pred],
        labels=test_set.classes,
    )
    cm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, display_labels=test_set.classes
    )
    cm_display.plot(xticks_rotation="vertical")
    plt.gcf().set_size_inches(12, 12)
    plt.tight_layout()
    if len(path) > 0:
        plt.savefig(path)
    else:
        plt.show()

    plt.close()


def concat_images(imga, imgb, direction="h"):
    # combines two color image ndarrays side-by-side.
    ha, wa = imga.shape[:2]
    hb, wb = imgb.shape[:2]

    if direction == "h":
        max_height = np.max([ha, hb])
        total_width = wa + wb
        new_img = np.zeros(shape=(max_height, total_width, 3), dtype=imga.dtype)
        new_img[:ha, :wa] = imga
        new_img[:hb, wa : wa + wb] = imgb
    else:
        max_width = np.max([wa, wb])
        total_height = ha + hb
        new_img = np.zeros(shape=(total_height, max_width, 3), dtype=imga.dtype)
        new_img[:ha, :wa] = imga
        new_img[ha : ha + hb, :wb] = imgb

    return new_img


def concat_n_images(image_path_list, n_rows, n_cols):
    # combines N color images from a list of image paths
    row_images = []
    for i in range(n_rows):
        output = None
        for j in range(n_cols):
            idx = i * n_cols + j
            img_path = image_path_list[idx]
            img = plt.imread(img_path)[:, :, :3]
            if j == 0:
                output = img
            else:
                output = concat_images(output, img)
        row_images.append(output)

    output = row_images[0]
    # row_images.append(abs(row_images[1] - row_images[0]))
    for img in row_images[1:]:
        output = concat_images(output, img, direction="v")

    return output


def visualize_image_pred(img_paths, target, output, classes, path=""):
    concat_imgs = concat_n_images(img_paths, n_rows=2, n_cols=4)
    plt.imshow(concat_imgs)

    pred_str = ", ".join([classes[x] for x in output])
    plt.text(10, -30, f"prediction: {pred_str}", c="black")
    if target is not None:
        plt.text(10, -60, f"label: {classes[target]}", c="black")
    if len(path) > 0:
        plt.savefig(path)
    else:
        plt.show()

    plt.close()


def visualize_pcd_pred(
    row_titles,
    state_list,
    views=[(90, -90), (0, -90), (45, -45)],
    axis_off=False,
    res="low",
    path="",
):
    n_rows = len(row_titles)
    n_cols = len(views)

    fig_size = 12 if res == "high" else 3
    title_fontsize = 60 if res == "high" else 10
    point_size = 160 if res == "high" else 10
    fig, big_axes = plt.subplots(
        n_rows, 1, figsize=(fig_size * n_cols, fig_size * n_rows)
    )

    for i in range(n_rows):
        state = state_list[i]
        if n_rows == 1:
            big_axes.set_title(
                row_titles[i], fontweight="semibold", fontsize=title_fontsize
            )
            big_axes.axis("off")
        else:
            big_axes[i].set_title(
                row_titles[i], fontweight="semibold", fontsize=title_fontsize
            )
            big_axes[i].axis("off")

        for j in range(n_cols):
            ax = fig.add_subplot(n_rows, n_cols, i * n_cols + j + 1, projection="3d")
            ax.view_init(*views[j])
            state_colors = state[:, 3:6] if state.shape[1] > 3 else "b"
            ax.scatter(
                state[:, 0], state[:, 1], state[:, 2], c=state_colors, s=point_size
            )
            # ax.set_zlim(-0.075, 0.075)

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

            if axis_off:
                ax.axis("off")

    if len(path) > 0:
        plt.savefig(path)
    else:
        plt.show()

    plt.close()


# def play_and_save_video(images, output_file, fps, pause_time_last_frame=3):
#     """
#     Plays the video in a pop-up window and saves it to the specified file.
#
#     Args:
#         images (list): List of PIL Images or NumPy arrays.
#         output_file (str): Output video file path.
#         fps (int): Frames per second for the video.
#         pause_time_last_frame (int): Time to pause on the last frame (milliseconds).
#     """
#     if isinstance(images[0], Image.Image):
#         frame_height, frame_width = images[0].height, images[0].width
#     else:
#         frame_height, frame_width, _ = images[0].shape
#
#     import pdb
#     pdb.set_trace()
#
#     writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"avc1"), fps, (frame_width, frame_height))
#
#
#     for image in images:
#         if isinstance(image, Image.Image):
#             frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#         else:
#             frame = image
#         writer.write(frame)
#
#     # Pause longer on the last frame in the saved video
#     for _ in range(int(pause_time_last_frame * fps)):
#         writer.write(images[-1])  # Write the last frame multiple times for the desired pause
#
#     writer.release()


def play_and_save_video(images, output_file, fps, pause_time_last_frame=3):
    """
    Plays the video in a pop-up window and saves it to the specified file using H.264 codec.

    Args:
        images (list): List of PIL Images or NumPy arrays.
        output_file (str): Output video file path.
        fps (int): Frames per second for the video.
        pause_time_last_frame (int): Time to pause on the last frame (milliseconds).
    """
    if isinstance(images[0], Image.Image):
        frame_height, frame_width = images[0].height, images[0].width
        images = [np.array(image) for image in images]
    else:
        frame_height, frame_width, _ = images[0].shape
        images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]

    # Create an imageio VideoWriter using H.264 codec
    writer = imageio.get_writer(output_file, fps=fps, codec='libx264')

    for image in images:
        writer.append_data(image)

    # Pause longer on the last frame in the saved video
    last_frame = images[-1]
    for _ in range(int(pause_time_last_frame * fps)):
        writer.append_data(last_frame)  # Write the last frame multiple times for the desired pause

    writer.close()


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

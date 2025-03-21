import os
import sys
# add project directory to PATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import numpy as np
import scipy

from tqdm import tqdm
from pathlib import Path
from utils.macros import *
from utils.utils import *
from utils_general import get_directory_contents

import os
import random
import shutil


def random_sample_and_move(source_folder, split_ratio=0.8):
    """
    Randomly samples files from subfolders under source_folder and moves them to train and validation subfolders
    based on the split_ratio.

    Parameters:
        source_folder (str): Path to the source folder containing subfolders (seq_x).
        split_ratio (float, optional): The ratio of files to be moved to the train subfolder.
                                      The rest will be moved to the validation subfolder. Default is 0.8 (80%).

    Returns:
        None
    """

    # Get a list of all subfolders (seq_x) in the source folder
    subfolders = [f for f in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, f))]

    for subfolder in subfolders:
        # Create the train and validation subfolders under the current subfolder (seq_x)
        train_folder = os.path.join(source_folder, "train", subfolder)
        validation_folder = os.path.join(source_folder, "validation", subfolder)
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(validation_folder, exist_ok=True)

        # Get a list of all files in the current subfolder
        files = os.listdir(os.path.join(source_folder, subfolder))

        # Shuffle the list of files randomly
        random.shuffle(files)

        # Calculate the number of files to move to train_folder based on the split_ratio
        num_files_train = int(len(files) * split_ratio)

        # Split the files into train and validation sets
        train_files = files[:num_files_train]
        validation_files = files[num_files_train:]

        # Move the train files to the train subfolder
        for file_name in train_files:
            source_file_path = os.path.join(source_folder, subfolder, file_name)
            dest_file_path = os.path.join(train_folder, file_name)
            shutil.move(source_file_path, dest_file_path)
            print(f'moved {source_file_path} to {dest_file_path}')

        # Move the validation files to the validation subfolder
        for file_name in validation_files:
            source_file_path = os.path.join(source_folder, subfolder, file_name)
            dest_file_path = os.path.join(validation_folder, file_name)
            shutil.move(source_file_path, dest_file_path)
            print(f'moved {source_file_path} to {dest_file_path}')

        # delete the folder it self
        subfolder_path = os.path.join(source_folder, subfolder)
        assert not os.listdir(subfolder_path)
        os.rmdir(subfolder_path)
        print(f'deleted {subfolder_path}')


def compute_stats_folder(folder, given_stats=None):
    pos_seq_batch = []
    force_batch, flow_batch, pressure_batch, action_batch = [], [], [], []
    for vid_path in tqdm(sorted(Path(folder).glob("*"))):
        # vid_path = vid_path_list[idx]
        if "processed" in os.path.basename(vid_path):
            continue

        video = read_video_data(vid_path)
        if len(video) == 0:
            print(f'video path contains no frames: {vid_path}. This might be due to wrong directory structure, e.g., missing one folder level')
            continue

        from utils_general import break_trajectory_dic_into_sequence
        seqs = []
        for trajectory in video:
            seq = break_trajectory_dic_into_sequence(trajectory)
            # print(seq[0]['inhand_object_pcs'].shape, seq[0]['object_pcs'].shape)
            # breakpoint()
            
            seqs += seq
            # tool_actions = np.stack([d['bubble_pcs'][:, :, :3].reshape(-1, 3).mean(0) for d in video])
            # tool_actions = tool_actions[1:, :3] - tool_actions[:-1, :3]
            tool_actions = np.stack([d['bubble_pcs'] for d in seq])
            tool_actions = tool_actions[1:, :, :, :3] - tool_actions[:-1, :, :, :3]
            # tool_actions = tool_actions[1:, :, :3] - tool_actions[:-1, :, :3]
            # tool_actions = np.clip(tool_actions, -0.05, 0.05)
            tool_actions = tool_actions.mean(1).mean(1)
            action_batch.append(tool_actions.reshape(-1, 3))
        
        video = seqs

        # from utils_general import replace_consecutive_failing_elements
        # video = replace_consecutive_failing_elements(video, lambda x: x['bubble_pcs'][..., 2].mean() < 0.3,
        #                                              num_consecutive=3)
        # video = list(filter(lambda x: x is not None, video))
        
        object_pcs = np.stack([d["object_pcs"] for d in video])[..., :3].reshape(-1, 3)
        inhand_pcs = np.stack([np.expand_dims(d["inhand_object_pcs"], axis=1) for d in video])[..., :3].reshape(-1, 3)
        bubble_pcs = np.stack([d['bubble_pcs'] for d in video])[..., :3].reshape(-1, 3)
        # pdb.set_trace()
        
        pos_seq = np.concatenate((object_pcs, inhand_pcs, bubble_pcs), axis=0)
        pos_seq_batch.append(pos_seq.reshape(-1, 3))

        forces = np.stack([d["forces"] for d in video])
        forces = np.concatenate([forces[..., :2], forces[..., -1:]], axis=-1).reshape(-1, 3)

        flows = np.stack([d['bubble_pcs'] for d in video])[..., -2:].reshape(-1, 2)
        # pressure = np.stack([d["pressure"] for d in video]).reshape(-1, 1)

        force_batch.append(forces)
        flow_batch.append(flows)
        # pressure_batch.append(pressure)
    
    pos_seq_batch = np.concatenate(pos_seq_batch, axis=0)
    position_mean = np.mean(pos_seq_batch, axis=0)
    position_scale = np.mean(np.linalg.norm(pos_seq_batch - position_mean, axis=1))
    
    # compute action statistics
    # not needed by dataset, but needed for planning
    action_batch = np.concatenate(action_batch, axis=0)
    # action_batch = action_batch.clip(-0.1, +0.1)
    action_mean = np.mean(action_batch, axis=0)
    action_scale = np.std(action_batch, axis=0)
    # import pdb; pdb.set_trace()
    # np.save('bubble_action.npy', action_batch)
    
    print(f'action mean = {action_mean}, std = {action_scale}')

    force_batch = np.concatenate(force_batch, axis=0)
    force_mean = np.mean(force_batch, axis=0)
    # force_scale = np.mean(np.linalg.norm(force_batch - force_mean, axis=1))
    force_scale = np.std(force_batch, axis=0)

    flow_batch = np.concatenate(flow_batch, axis=0)
    flow_mean = np.mean(flow_batch, axis=0)
    # flow_scale = np.mean(np.linalg.norm(flow_batch - flow_mean, axis=1))
    flow_scale = np.std(flow_batch, axis=0)

    # pressure_batch = np.concatenate(pressure_batch, axis=0)
    # pressure_mean = np.mean(pressure_batch, axis=0)
    # # pressure_scale = np.mean(np.linalg.norm(pressure_batch - pressure_mean, axis=1))
    # pressure_scale = np.std(pressure_batch, axis=0)

    stats = {
        "position_mean": position_mean,
        "position_scale": position_scale,
        "force_mean": force_mean,
        "force_scale": force_scale,
        "flow_mean": flow_mean,
        "flow_scale": flow_scale,
        # "pressure_mean": pressure_mean,
        # "pressure_scale": pressure_scale
    }
    print(stats)

    if given_stats is not None:
        print(f'stats overridden to {given_stats}')
        store_h5_data(
            given_stats,
            os.path.join(folder, "stats.h5")
        )
    else:
        store_h5_data(
            stats,
            os.path.join(folder, "stats.h5")
        )

    return stats


def main(args):
    root = os.path.join(DATA_DIR, args.data_path)
    assert os.path.exists(root), f'{root} does not exist'

    # random_sample_and_move(root)
    train_stats = compute_stats_folder(os.path.join(root, 'train'))
    compute_stats_folder(os.path.join(root, 'validation'), train_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data")
    args = parser.parse_args()
    main(args)

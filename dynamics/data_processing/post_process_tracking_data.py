#!/usr/bin/env python3
import os.path
import pdb
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import numpy as np
from tqdm import tqdm
import time
import multiprocessing as mp

from config.task_config import N_POINTS_BUBBLE, extrinsics, intrinsics_depth, intrinsic_matrices
from perception.utils_gripper import *
from perception.utils_cv import *
from perception.utils_pc import *
from utils_general import *
from utils.autosave_list import AutosaveList


# hard-code indices to downsample the softbubble points (640*480=307200) to 20 points
# derived with fps on a specific point cloud sample
# see find_indices_softbubble.ipynb under notebooks/ for details  
# softbubble_mask = [233380, 49416, 52968, 302255, 130858, 168922, 141873, 171692, 32966, 
#                    240928, 251440, 87419, 100404, 178918, 289672, 100695, 191847, 28609, 183301, 236996]


def generate_random_bool_mask(N, K, seed=None):
    """
    Generate a random boolean mask of length N with K True values.

    Args:
    - N (int): Length of the mask.
    - K (int): Number of True values.
    - seed (int, optional): Seed for NumPy's random number generator for reproducibility.

    Returns:
    - numpy.ndarray: Random boolean mask.
    """
    if K > N:
        raise ValueError("K must be less than or equal to N")

    # Set the random seed if provided
    if seed is not None:
        np.random.seed(seed)

    mask = np.zeros(N, dtype=bool)
    mask[:K] = True
    np.random.shuffle(mask)

    return mask


# We evenlly ample points instead
softbubble_mask = generate_random_bool_mask(640 * 480, 20, seed=0)
bubble1_mask, bubble2_mask = None, None 


def process_seq_folder(seq_folder, args):
    save_path = os.path.join(args.target_path, f'{seq_folder}.h5')

    if os.path.isfile(save_path):
        print(f'skipping {seq_folder} because it already exists')
        return

    print(f'start processing {seq_folder}')
    os.makedirs(args.target_path, exist_ok=True)

    seq_folder_path = os.path.join(args.raw_data_path, seq_folder)

    if not os.path.isdir(seq_folder_path) or 'seq_' not in seq_folder:
        return
    
    tracking_file_path = os.path.join(args.tracking_data_path, f'{seq_folder}.h5')
    
    if not os.path.exists(tracking_file_path): 
        print(f'Tracking file path not exist, skipped: {tracking_file_path}')
        return
    
    particle_arrays = load_dictionary_from_hdf5(tracking_file_path)
    if len(particle_arrays) == 0:
        print(f'No data found in {tracking_file_path}, skipped')
        return
    particle_arrays = np.stack(particle_arrays.values(), axis=1)  # assuming the order is 'blue pole', 'red carton box'
    
    ee_states = AutosaveList.read_all_data(os.path.join(seq_folder_path, 'ee_states'), 'ee_states')
    joint_states = AutosaveList.read_all_data(os.path.join(seq_folder_path, 'joint_states'), 'joint_states')
    b1_raw_flows = AutosaveList.read_all_data(os.path.join(seq_folder_path, 'bubble_1'), 'raw_flow')
    b2_raw_flows = AutosaveList.read_all_data(os.path.join(seq_folder_path, 'bubble_2'), 'raw_flow')
    b1_depths = AutosaveList.read_all_data(os.path.join(seq_folder_path, 'bubble_1'), 'depth')
    b2_depths = AutosaveList.read_all_data(os.path.join(seq_folder_path, 'bubble_2'), 'depth')
    b1_forces = AutosaveList.read_all_data(os.path.join(seq_folder_path, 'bubble_1'), 'force')
    b2_forces = AutosaveList.read_all_data(os.path.join(seq_folder_path, 'bubble_2'), 'force')
    topics_dict = {
        '/ee_states': ee_states,
        '/joint_states': joint_states, 
        '/bubble_1/raw_flow': b1_raw_flows,
        '/bubble_2/raw_flow': b2_raw_flows,
        '/bubble_1/force': b1_forces,
        '/bubble_2/force': b2_forces
    }
    
    # topics_dict = np.load(os.path.join(seq_folder_path, 'other_topics.npy'), allow_pickle=True).item()
    # topics_dict = {k: np.array(v) for k, v in topics_dict.items()}
    boolean_mask = generate_random_bool_mask(50, 50, seed=0)       # all True 

    # # get soft bubble points
    # b1_raw_flows = topics_dict['/bubble_1/raw_flow']
    # b2_raw_flows = topics_dict['/bubble_2/raw_flow']

    # check the length of the data
    data_len = min(particle_arrays.shape[0], len(b1_raw_flows), len(b2_raw_flows)) if args.max_nframe is None else args.max_nframe
    if particle_arrays.shape[0] != len(b1_raw_flows):
        warnings.warn(f'length of particle arrays ({particle_arrays.shape[0]}) '
                      f'does not match the length of raw flows ({len(b1_raw_flows)})')

    save_data_dict = {
        'object_pcs': [],
        'bubble_pcs': [],
        'inhand_object_pcs': [],
        'forces': [],
        'ee_pos': []
    }

    prev_rod_points = None
    for t in tqdm(range(len(b1_raw_flows))[:data_len]):
        # robot-related info
        ee_pose = topics_dict['/ee_states'][t]
        ee_pos, ee_orient = ee_pose[:3], ee_pose[3:]
        gripper_dist = topics_dict['/joint_states'][t][-1]
        t_bubbles_to_robot = get_bubble_cameras_to_robot(ee_pos, ee_orient, gripper_dist)

        # process soft bubble points
        b1_flow = b1_raw_flows[t].reshape(240, 320, 2)
        b2_flow = b2_raw_flows[t].reshape(240, 320, 2)

        b1_flow = cv2.resize(b1_flow, (640, 480))
        b2_flow = cv2.resize(b2_flow, (640, 480))

        b1_depth, b2_depth = b1_depths[t], b2_depths[t]
        # b1_depth = cv2.imread(os.path.join(seq_folder_path, 'bubble_1', f'depth_{t}.png'), cv2.IMREAD_ANYDEPTH)
        # b2_depth = cv2.imread(os.path.join(seq_folder_path, 'bubble_2', f'depth_{t}.png'), cv2.IMREAD_ANYDEPTH)

        b1_feat_points = rgbd_feat_to_point_cloud(b1_flow, b1_depth, intrinsic_matrices['bubble_1'])
        b2_feat_points = rgbd_feat_to_point_cloud(b2_flow, b2_depth, intrinsic_matrices['bubble_2'])

        global bubble1_mask, bubble2_mask
        if bubble1_mask is None and bubble2_mask is None:
            assert t == 0, f'bubble1_mask and bubble2_mask should be None if t is not zero (t = {t})'
            bubble_pcs = [b1_feat_points, b2_feat_points]
            bubble_pcs = [remove_distant_points(pc, 0.15) for pc in bubble_pcs]
            bubble_pcs = [random_select_rows(pc, 10000) for pc in bubble_pcs]
            bubble_pcs = [denoise_by_cluster(pc, 0.01, 10, 1) for pc in bubble_pcs]
            bubble_sampled_pcs = [farthest_point_sampling_dgl(pc, n_points=N_POINTS_BUBBLE)
                                for pc in bubble_pcs]
            bubble1_mask = find_indices(b1_feat_points, bubble_sampled_pcs[0])
            bubble2_mask = find_indices(b2_feat_points, bubble_sampled_pcs[1])
        else:
            bubble_sampled_pcs = [b1_feat_points[bubble1_mask], b2_feat_points[bubble1_mask]]
        
        bubble_sampled_pcs = [project_points(convert_pc_optical_color_to_link_frame(pc), t_bubble_to_robot) 
                              for pc, t_bubble_to_robot in zip(bubble_sampled_pcs, t_bubbles_to_robot)]
        
        # # bubble_sampled_pcs = [b1_feat_points, b2_feat_points]
        # np.save(f'b1_feat_points_sampled{t}.npy', bubble_sampled_pcs[0])
        # import pdb; pdb.set_trace()

        # obtain object particles
        obj_particle_array = particle_arrays[t][:, boolean_mask]
        
        # read rod points from 3d model instead
        if args.fix_rod:
            obj_particle_array[0] = get_rod_points_from_model('../../asset/rod_3x3x25_fixed_Aug1.ply', ee_pos, obj_particle_array[0].shape[-2], 0)

        save_data_dict['object_pcs'].append(obj_particle_array[1:])
        save_data_dict['bubble_pcs'].append(bubble_sampled_pcs)
        save_data_dict['inhand_object_pcs'].append(obj_particle_array[0])
        save_data_dict['forces'].append(
            np.stack([topics_dict['/bubble_1/force'][t], topics_dict['/bubble_2/force'][t]], axis=0)
        )
        save_data_dict['ee_pos'].append(ee_pos)

        if args.debug:
            bubble_sampled_pcs_o3d = [xyz_to_pc(rec_pc) for rec_pc in bubble_sampled_pcs]
            obj_pc_o3d = [xyz_to_pc(x) for x in obj_particle_array]
            o3d.visualization.draw_geometries(bubble_sampled_pcs_o3d + obj_pc_o3d)

    save_dictionary_to_hdf5(save_data_dict, save_path)
    print(f'Data saved successfully at {save_path}')
    del save_data_dict


def main(args):
    """
    For each folder named seq_x in the source path, read the rosbag files,
    reconstruct the scene, sample the point clouds, and save the training data
    into a h5 file.
    :param args: user-input args
    :return: None
    """

    # Get a list of seq_folders from os.listdir(args.raw_data_path)
    seq_folders = os.listdir(args.raw_data_path)
    for i, folder in enumerate(seq_folders):
        process_seq_folder(folder, args)
        print(f'Finished processing {i + 1} / {len(seq_folders)} folders')

    # # Create a multiprocessing pool with the desired number of worker processes
    # num_processes = 4  # Adjust this to the number of desired parallel processes
    #
    # # Create a partial function with fixed 'args'
    # from functools import partial
    # partial_process_seq_folder = partial(process_seq_folder, args=args)
    # # partial_process_seq_folder(seq_folders[0])
    #
    # with mp.Pool(processes=num_processes) as pool:
    #     # Use the map method to apply the partial_process_seq_folder function in parallel
    #     results = pool.map(partial_process_seq_folder, seq_folders)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-path", type=str,
                        help="Path to directory containing the seq_x folders")
    parser.add_argument('--tracking-data-path', type=str,
                        help='Path to a directory containing tracking data')
    parser.add_argument("--debug", type=int, default=0, help="Debug mode or not")
    parser.add_argument("--target-path", type=str,
                        help="Directory to store parsed data")
    # parser.add_argument("--num_objects", type=int, default=1, help="number of objects on the table")
    parser.add_argument("--max-nframe", type=int, default=None,
                        help="max number of frames in one h5 file")
    parser.add_argument("--fix-rod", type=int, default=1, 
                        help="read rod points from 3D model or not")
    args = parser.parse_args()

    main(args)

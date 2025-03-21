import itertools
import pdb
import random

import pytorch_lightning as pl
import torch_geometric as pyg

from torch.utils.data import ConcatDataset

from utils.utils import *
from utils.visualizer import *
from utils_general import AverageMeter, replace_consecutive_failing_elements


def get_edge_attr(pos, pos_p, edge_index, radius):
    # edge-level features: displacement, distance
    dim = pos.shape[-1]
    edge_displacement = torch.gather(
        pos, dim=0, index=edge_index[1].unsqueeze(-1).expand(-1, dim)
    ) - torch.gather(pos_p, dim=0, index=edge_index[0].unsqueeze(-1).expand(-1, dim))
    edge_displacement /= radius
    edge_distance = torch.norm(edge_displacement, dim=-1, keepdim=True)

    edge_attr = torch.cat((edge_displacement, edge_distance), dim=-1)

    return edge_attr


def compute_radius_unsorted_batch(x, y, r, batch_x, batch_y, max_num_neighbors, batch_x_presorted=False, batch_y_presorted=False):
    # compute pyg.nn.radius for a batch of graphs
    # pyg.nn.radius usually assumes that the batch_x and batch_y are sorted
    # in this function, we assume that the batch_x and batch_y are not sorted by default
    # if either are sorted, set the corresponding flag to True to save a minro amount of computation time

    if batch_x_presorted:
        batch_x_sorted = batch_x
        batch_x_sorted_idx = torch.arange(batch_x.shape[0], device=batch_x.device)
    else:
        batch_x_sorted, batch_x_sorted_idx = torch.sort(batch_x)
        x = x[batch_x_sorted_idx]

    if batch_y_presorted:
        batch_y_sorted = batch_y
        batch_y_sorted_idx = torch.arange(batch_y.shape[0], device=batch_y.device)
    else:
        batch_y_sorted, batch_y_sorted_idx = torch.sort(batch_y)
        y = y[batch_y_sorted_idx]

    edge_pairs = pyg.nn.radius(
        x=x,
        y=y,
        r=r,
        batch_x=batch_x_sorted,
        batch_y=batch_y_sorted,
        max_num_neighbors=max_num_neighbors,
    )

    # sort the edge pairs back to the original order
    edge_pairs_y, edge_pairs_x = edge_pairs[0], edge_pairs[1]

    if batch_x_presorted:
        edge_pairs_x_unsorted = edge_pairs_x
    else:
        edge_pairs_x_unsorted = torch.arange(batch_x.shape[0], device=batch_x.device)[batch_x_sorted_idx][edge_pairs_x]

    if batch_y_presorted:
        edge_pairs_y_unsorted = edge_pairs_y
    else:
        edge_pairs_y_unsorted = torch.arange(batch_y.shape[0], device=batch_y.device)[batch_y_sorted_idx][edge_pairs_y]

    edge_pairs = torch.stack((edge_pairs_y_unsorted, edge_pairs_x_unsorted), dim=0)
    return edge_pairs


def get_batch_vector(num_nodes, batch_size, node_index_cumsum, shift_obj_idx=0, device='cuda'):
    """
    :param num_nodes: total number of nodes in each graph e.g. 80
    :param batch_size: number of graphs in a batch e.g. 8
    :param node_index_cumsum: length num(obj) array of the starting index of each object. e.g. (0, 20, 40)
    :param shift_obj_idx: shift the indices of the objects by this amount, then mod by num(obj)
    :param device: device
    :return: a tensor that assigns each object from each batch into a different batch index. guaranteed to be monotonic if shift_obj_idx is 0, but not otherwise
    """
    num_obj = len(node_index_cumsum)
    node_index_cumsum = np.concatenate([node_index_cumsum, [num_nodes]])
    tensor = torch.zeros(batch_size, num_nodes, device=device)
    obj_idxs = torch.remainder(torch.arange(num_obj) + shift_obj_idx, num_obj)
    for i in range(num_obj):
        tensor[:, node_index_cumsum[i]: node_index_cumsum[i + 1]] = obj_idxs[i]
    # now, add 3 times the column index to each row
    tensor += torch.arange(batch_size, device=device).view(-1, 1) * (len(node_index_cumsum)-1)
    # now reshape it to (batch_size * num_nodes)
    tensor = tensor.view(-1)
    return tensor


def connect_edges_batched(config, pos_obj, pos_tool, total_num_points, batch_size, cumul_points):
    N = total_num_points
    num_tool_points = N - sum(config["n_points"])
    max_object_pts = max(max(config["n_points"]), num_tool_points)
    all_pts = torch.cat((pos_obj, pos_tool), dim=1)
    all_edges = list()
    all_edge_attr = list()
    if config['use_knn']:
        raise NotImplementedError("Batched KNN not yet implemented, but very doable, see radius graph for example")
    object_obs_edges = pyg.nn.radius_graph(all_pts.reshape(N * batch_size, 3), r=config["connectivity_radius_inner"],
                                           batch=get_batch_vector(N, batch_size, cumul_points, device=all_pts.device),
                                           max_num_neighbors=max_object_pts)
    all_edge_attr.append(get_edge_attr(all_pts.reshape(N * batch_size, 3), all_pts.reshape(N * batch_size, 3),
                              object_obs_edges, config["connectivity_radius_inner"]))
    all_edges.append(object_obs_edges)
    num_obj = len(cumul_points)
    for obj_comparison_idx in range(num_obj - 1):
        edge_pairs = compute_radius_unsorted_batch(
            x=all_pts.reshape(N * batch_size, 3),
            y=all_pts.reshape(N * batch_size, 3),
            r=config["connectivity_radius_outer"],
            batch_x=get_batch_vector(N, batch_size, cumul_points,
                                     shift_obj_idx=obj_comparison_idx + 1, device=all_pts.device).long(),
            batch_y=get_batch_vector(N, batch_size, cumul_points, device=all_pts.device).long(),
            max_num_neighbors=max_object_pts,
            batch_y_presorted=True,
        )
        all_edges.append(edge_pairs)
        all_edge_attr.append(get_edge_attr(all_pts.reshape(N * batch_size, 3), all_pts.reshape(N * batch_size, 3),
                                           edge_pairs, config["connectivity_radius_outer"]))

    all_edges = torch.cat(all_edges, dim=1)
    all_edge_attr = torch.cat(all_edge_attr, dim=0)
    return all_edges, all_edge_attr


def compute_slice_indices(tensor, points_per_graph, batch_size):
    # Calculate the threshold values for each location in the result tensor
    thresholds = torch.arange(0, batch_size).to(tensor.device) * points_per_graph
    # Create a mask indicating where each element in the index tensor is greater than its threshold
    mask = tensor.unsqueeze(0) >= thresholds.unsqueeze(1)
    # Find the first occurrence of True along the columns (axis=1) in the mask
    first_occurrence = mask.long().argmax(dim=1)
    # also add the total number of points to the end of the list
    first_occurrence = torch.cat((first_occurrence, torch.tensor([tensor.shape[0]]).to(tensor.device)))
    return first_occurrence


def connect_edges(config, pos_dict):
    edge_pairs_list = []
    edge_attr_list = []

    # connect edges inside each object
    for key, (idx_offset, pos) in pos_dict.items():
        # if "action" in key:
        #     continue

        # radius graph is bidirectional, but knn graph is not+
        if config["use_knn"]:
            edge_pairs = pyg.nn.knn_graph(pos, k=config["k_neighbors"], loop=False)
        else:
            edge_pairs = pyg.nn.radius_graph(
                pos, config["connectivity_radius_inner"], max_num_neighbors=pos.shape[0]
            )
        edge_attr = get_edge_attr(
            pos, pos, edge_pairs, config["connectivity_radius_inner"]
        )
        edge_attr_list.append(edge_attr)

        edge_pairs_abs = edge_pairs + idx_offset
        edge_pairs_list.append(edge_pairs_abs)

    # connect edges between objects
    pairs = list(itertools.permutations(pos_dict.items(), 2))
    for pair in pairs:
        key, (idx_offset, pos) = pair[0]
        key_p, (idx_offset_p, pos_p) = pair[1]
        # assert idx_offset_p > idx_offset
        edge_pairs = pyg.nn.radius(
            pos,
            pos_p,
            config["connectivity_radius_outer"],
            max_num_neighbors=pos.shape[0],
        )

        edge_attr = get_edge_attr(
            pos, pos_p, edge_pairs, config["connectivity_radius_outer"]
        )
        edge_attr_list.append(edge_attr)

        edge_pairs_abs = torch.cat(
            (edge_pairs[:1] + idx_offset_p, edge_pairs[1:] + idx_offset), dim=0
        )
        edge_pairs_list.append(edge_pairs_abs)

        if config['debug']:
            print(f'inter-object ({idx_offset}, {idx_offset_p}) connections count: {edge_pairs.shape[-1]}')

    # merge all edges
    edge_pairs = torch.cat(edge_pairs_list, dim=-1)
    edge_attr = torch.cat(edge_attr_list, dim=0)

    # check if the edge is bidirectional
    # if config['debug']:
    #     edge_pairs_list = edge_pairs.permute(1, 0).tolist()
    #     for x, y in edge_pairs_list:
    #         assert [y, x] in edge_pairs_list

    # import pdb
    # pdb.set_trace()

    return edge_pairs, edge_attr


# def random_sample_from_first_dim(arr, n):
#     """
#     Randomly sample 'n' elements along the first dimension of the input array.

#     Parameters:
#         arr (numpy.ndarray): Input array of shape (N, .., K).
#         n (int): Number of elements to sample.

#     Returns:
#         numpy.ndarray: New array of shape (n, K) containing the randomly sampled elements.
#     """
#     # Get the size of the first dimension (N)
#     N = arr.shape[0]

#     # Randomly choose 'n' indices from the first dimension without replacement
#     sampled_indices = np.random.choice(N, n, replace=False)

#     # Use array indexing to select the sampled elements
#     sampled_data = arr[sampled_indices]

#     return sampled_data


from dgl.geometry import farthest_point_sampler
def farthest_point_sampling_numpy(points, n_points):
    # raise NotImplementedError
    """
    Sample specified number of farthest points from a NumPy array.

    Parameters:
        points (numpy.ndarray): Input array of shape (N, k) representing N points in k-dimensional space.
        n_points (int): The number of points to sample.

    Returns:
        numpy.ndarray: An array of shape (n_points, k) containing the sampled points.
    """
    assert n_points > 0, 'Number of points to sample should be positive'
    N, k = points.shape
    assert n_points <= N, 'Number of points to sample cannot exceed the total number of points'
    
    # shortcut solution for trivial case
    if points.shape[0] == n_points:
        return points.copy(), np.arange(n_points)

    # Convert the NumPy array to a PyTorch tensor
    points_tensor = torch.from_numpy(points)

    # Call the farthest_point_sampler to get the indices of the sampled points3
    indices = farthest_point_sampler(points_tensor.unsqueeze(0), n_points, start_idx=0).squeeze(0).numpy()   # , start_idx=0 for deterministic fps

    # Get the sampled points from the original array
    sampled_points = points[indices]

    return sampled_points, indices


def preprocess_points_to_particles(points, n):
    assert len(points.shape) == 2, \
        f"shape of points should be (N, k) but got {points.shape}"
        
    if points.shape[0] % 20 == 0 and n == 20:
        return points
    elif points.shape[0] % 50 == 0 and n == 50:
        return points 
    else:
        raise NotImplementedError()
    
    # return farthest_point_sampling_numpy(points, n_particles)[0]


tick = 0


def get_object_pcs_from_state_dict(state_dict, config):
    """
    A sub-function of construct_graph_from_video, which extracts the object point clouds from the state_dict
    """
    objects = []
    for i in range(len(state_dict['object_pcs'])):
        objects.append(torch.from_numpy(preprocess_points_to_particles(state_dict["object_pcs"][i], 
                                                                       config["particles_per_obj"])))
    return torch.concat(objects, dim=0)


def construct_graph_from_video(config, state_list, target=False, trace=False):
    # object_pcs: (1, 500, 6)
    # inhand_object_pcs: (500, 6)
    # bubble_pcs: (2, 500, 6)

    # objects' point clouds
    object_obs = []
    for d in state_list:
        object_obs.append(get_object_pcs_from_state_dict(d, config))
    object_obs = torch.stack(object_obs, dim=0)
    
    # object_obs = torch.stack([torch.from_numpy(preprocess_points_to_particles(d["object_pcs"][0],
    #                                                                           config["particles_per_obj"]))
    #                           for d in state_list])  # assume one object

    # in-hand object point cloud, pose obtained by estimation
    in_hand_object_obs = torch.stack([torch.from_numpy(preprocess_points_to_particles(d["inhand_object_pcs"],
                                                                                      config["particles_per_obj"]))
                                      for d in state_list])

    # bubble point cloud, ground truth (from tactile sensor)
    bubble_gt = torch.stack([torch.from_numpy(preprocess_points_to_particles(d["bubble_pcs"].reshape(-1, d["bubble_pcs"].shape[-1]),
                                                                             config["particles_per_obj"]))
                            for d in state_list])  # view two bubbles as one
    # object class
    object_cls = torch.tensor([d["object_cls"] for d in state_list])
    # object_cls = torch.tensor([-1 for d in state_list])
    
    # # end effector position
    # ee_positions = torch.stack([torch.from_numpy(d['ee_pos']) for d in state_list], dim=0)

    # extract the particles and flows from bubble_gt
    flows = bubble_gt[..., -2:]
    bubble_gt = bubble_gt[..., :-2]

    # tactile reading
    forces = torch.stack([torch.from_numpy(d["forces"]) for d in state_list])       # (T, 2, 7)
    # flows = torch.stack([torch.from_numpy(d["flows"]) for d in state_list])        # (T, 2, 240, 320, 3)
    # pressure = torch.stack([torch.from_numpy(1010) for d in state_list])       # (T, 2)

    his_len = config["history_length"]
    T = object_obs.shape[0]
    n_points_object = object_obs.shape[1]
    n_points_inhand = in_hand_object_obs.shape[1]
    n_points_bubble = bubble_gt.shape[1]
    n_points = n_points_object + n_points_inhand + n_points_bubble

    # Generate a integer mask of shape (N, 1) indicating
    # the identity of each particle
    particle_type = torch.cat(
        [torch.full((n, 1), i, dtype=torch.int)
         for i, n in enumerate([n_points_object, n_points_inhand, n_points_bubble])],
        dim=0,
    )

    # get the RGB values associated to each pixel
    # rgb_features = torch.cat(
    #     (
    #         object_obs[his_len - 1, :, 3:6],
    #         in_hand_object_obs[his_len - 1, :, 3:6],
    #         bubble_gt[his_len - 1, :, 3:6]
    #     ),
    #     dim=0,
    # )
    rgb_features = torch.cat(
        (
            torch.zeros_like(object_obs)[his_len - 1, :, :3],
            torch.zeros_like(in_hand_object_obs)[his_len - 1, :, :3],
            torch.zeros_like(bubble_gt)[his_len - 1, :, :3]
        ),
        dim=0,
    )

    # obtain the action applied on the tool
    # action is the particle-wise displacement of the tool from
    # the last to this time step
    # the action of the other objects is zero
    # When the tool is a softbubble, the action should be computed from the mean movment of all particles
    # even if point to point correspondence is available 
    # otherwise sampling action sequences at planning time will not be feasible 
    # for box pushing: 
    tool_actions = in_hand_object_obs[1:, :, :3] - in_hand_object_obs[:-1, :, :3]
    tool_actions = np.clip(tool_actions, -0.10, 0.10)
    tool_actions = tool_actions.mean(1).unsqueeze(1).repeat(1, n_points_bubble, 1)
    
    # for packing: let's use the ee displacement as the action, which is more accurate than estimating it from bubble points, 
    # since bubble points are noisy due to depth sensing
    # ee_displacement = ee_positions[1:] - ee_positions[:-1]
    # tool_actions = ee_displacement.unsqueeze(1).repeat(1, n_points_bubble, 1)
    
    actions = torch.cat(
        (
            torch.zeros((T - 1, n_points_object + n_points_inhand, 3), dtype=torch.float32),
            tool_actions,
        ),
        dim=1,
    )
    actions = actions.transpose(0, 1).reshape(n_points, -1)

    # node features is the concatenation of particle type, rgb feature, and actions
    # 1 + 3 + 3 * (H + W - 1)
    node_features = torch.cat((particle_type, rgb_features, actions), dim=-1)   # first dim N_o + n_points_inhand + N_t

    # construct the graph based on the distances between particles
    pos_dict = OrderedDict(
        object_obs=(0, object_obs[his_len - 1, :, :3]),        # format: (start_index, position)
        inhand=(n_points_object, in_hand_object_obs[his_len - 1, :, :3]),
        bubble=(n_points_object + n_points_inhand, bubble_gt[his_len - 1, :, :3]),
    )

    edge_index, edge_attr = connect_edges(
        config,
        pos_dict,
    )

    # ground truth for training
    # if target:
    #     target_state = torch.cat((object_obs[-1], in_hand_object_obs[-1]), dim=0)
    # else:
    #     target_state = None

    pos_seq = (
        torch.cat((object_obs[..., :3], in_hand_object_obs[..., :3], bubble_gt[..., :3]), dim=1)
        .transpose(0, 1)
        .reshape(n_points, -1)
    )

    # return the graph with features
    # x: (N, 1 + (F - 1) * 3)
    # y: (N_p, 3)
    # pos: (N, (F - 1) * D)
    data = pyg.data.Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        # y=target_state,
        pos=pos_seq,        # ground truth for loss computation
        forces=forces,
        flows=flows,
        # pressure=pressure,
        object_cls=object_cls,
        # rand_index = torch.cat([torch.from_numpy(d['rand_index']) for d in state_list], dim=0)
    )

    if trace:
        import pdb
        pdb.set_trace()

    # for debugging
    n = config["particles_per_obj"]
    if config['debug']:
        global tick
        if tick > 15 and tick % 2 == 0:
            # import pdb
            # pdb.set_trace()
            print(f'num of nodes {data.num_nodes}, num of edges {data.num_edges}')
            nx_graph = pyg.utils.to_networkx(data, to_undirected=False)
            print(f'tick is {tick} and n is {n}. make sure n is correct')
            visualize_networkx(config["visualizer"], nx_graph, pos_seq,
                               config['particles_per_obj'] * 2, f"networkx/tick_{tick}.jpg")
        tick += 1

    return data


counter = 0 
def downsample_points_state_dict_seq(state_dict_seq, config, return_mask=False, object_masks=None, inhand_object_mask=None):
    # return state_dict_seq
    # assert that n_points have the same value as particles_per_obj
    assert (np.array(config["n_points"]) == config["particles_per_obj"]).all(), \
        f"n_points should be the same as particles_per_obj, but got {config['n_points']} and {config['particles_per_obj']}"
    
    n_points = config["particles_per_obj"]
    
    softbubble_masks = []
    if object_masks is None:
        object_masks = []
        preloaded_object_masks = False
    else:
        preloaded_object_masks = True

    if inhand_object_mask is None:
        inhand_object_mask = []
        preloaded_inhand_object_mask = False
    else:
        preloaded_inhand_object_mask = True

    new_state_dict_seq = []

    for i, state_dict in enumerate(state_dict_seq):
        # dict_keys(['bubble_pcs', 'forces', 'inhand_object_pcs', 'object_pcs'])
        # assert state_dict['bubble_pcs'].shape[1] == 50, f'soft bubble should have 20 particles each, but got {state_dict["bubble_pcs"].shape[1]}'
        state_dict = {k: v for k, v in state_dict.items()}      # critical, we don't want inplace operations
        
        # process inhand object points
        if "inhand_object_pcs" in state_dict:
            if i == 0 and not preloaded_inhand_object_mask:
                state_dict['inhand_object_pcs'], inhand_object_mask = farthest_point_sampling_numpy(state_dict['inhand_object_pcs'], n_points)
            else:
                state_dict['inhand_object_pcs'] = state_dict['inhand_object_pcs'][inhand_object_mask]

        # process softbubble points
        new_bubble_pcs = [[], []]
        for j, bubble_pc in enumerate(state_dict['bubble_pcs']):        # for each bubble out of the two 
            if i == 0:
                new_bubble_pc, softbubble_mask = farthest_point_sampling_numpy(bubble_pc, n_points)
                softbubble_masks.append(softbubble_mask)
                new_bubble_pcs[j].append(new_bubble_pc)
            else:
                new_bubble_pc = bubble_pc[softbubble_masks[j]]
                new_bubble_pcs[j].append(new_bubble_pc)
        state_dict['bubble_pcs'] = np.array(new_bubble_pcs)
        
        # perform this for every object
        new_object_pcs = []
        for j, object_pc in enumerate(state_dict['object_pcs']):
            if i == 0 and not preloaded_object_masks:
                new_object_pc, object_mask = farthest_point_sampling_numpy(object_pc, n_points)
                object_masks.append(object_mask)
                new_object_pcs.append(new_object_pc)
            else:
                new_object_pc = object_pc[object_masks[j]]
                new_object_pcs.append(new_object_pc)
        state_dict['object_pcs'] = np.stack(new_object_pcs, axis=0)

        global counter
        state_dict['rand_index'] = np.zeros(len(state_dict['object_pcs']), dtype=np.int32) + counter

        new_state_dict_seq.append(state_dict)
        
    counter += 1
    # print(f'counter  = {counter}')

    if return_mask:
        return new_state_dict_seq, softbubble_masks, object_masks, inhand_object_mask
        
    return new_state_dict_seq


def replicate_first_bubble_pcs_for_all(state_dict_seq):
    new_state_dict_seq = []
    bubble_points = None
    for i, state_dict in enumerate(state_dict_seq):
        state_dict = {k: v.copy() for k, v in state_dict.items()}      # a shallow copy
        if bubble_points is None : 
            bubble_points = state_dict['bubble_pcs'].copy()
        else:
            # shift the bubble points to the current bubble center
            delta = state_dict['bubble_pcs'][..., :3].reshape(-1, 3).mean(0) - bubble_points[..., :3].reshape(-1, 3).mean(0)
            delta = np.clip(delta, -0.05, 0.05)     # clip outliers
            bubble_points = bubble_points.copy()
            bubble_points[..., :3] += delta      
            bubble_points[..., 3:] = state_dict['bubble_pcs'][..., 3:]
            state_dict['bubble_pcs'] = bubble_points
        new_state_dict_seq.append(state_dict)
    return new_state_dict_seq


def shift_state_dict_seq(state_list):
    # manually adjust delay between observation and ee
    offset_t = int(np.round(0.2 * 8))  # number of frames for latency in observations: 6 frames / sec * 0.2 secs
    late_keys = ['object_pcs', 'inhand_object_pcs']
    for state in state_list:
        assert set(late_keys).issubset(set(state.keys()))
        for key, array in state.items():
            # if key == 'object_cls':
            #     continue
            if key in late_keys:
                array = array[offset_t:]
            else:
                array = array[:-offset_t]
            state[key] = array
    print(f'applying offset_t {offset_t} while reading data')


def filter_points_by_distance_to_center(point_cloud, distance_limit=1.0, limit_percentile=50):
    """
    Filter points based on distance from the center of the point cloud.

    Parameters:
    - point_cloud: numpy array of shape (N, 3) representing the point cloud.
    - limit_percentile: The percentile range for determining the center (default is 50).
    - distance_limit: The distance limit from the center for retaining points (default is 1.0).

    Returns:
    - Filtered point cloud.
    """

    # Calculate the center based on the mean of the middle percentiles
    lower_limit = np.percentile(point_cloud, (100 - limit_percentile) / 2, axis=0)
    upper_limit = np.percentile(point_cloud, 100 - (100 - limit_percentile) / 2, axis=0)
    center = np.mean([lower_limit, upper_limit], axis=0)

    # Calculate the distance of each point from the center
    distances = np.linalg.norm(point_cloud - center, axis=1)

    # Retain points whose distance to the center is smaller than the limit
    valid_points_mask = distances <= distance_limit
    filtered_point_cloud = point_cloud[valid_points_mask]

    return filtered_point_cloud


def denoise_softbubble_points(state_list):
    new_state_list = [] 
    for state_dict in state_list:
        state_dict = {k: v.copy() for k, v in state_dict.items()}
        denoised_bubble_points = filter_points_by_distance_to_center(state_dict['bubble_pcs'], distance_limit=0.5)
        state_dict['bubble_pcs'] = denoised_bubble_points
    return new_state_list


class OneStepDataset(pyg.data.InMemoryDataset):
    def __init__(
        self,
        config,
        split="train",
        transform=None,
    ):
        self.config = config
        self.root = os.path.join(DATA_DIR, config["data_dir_prefix"], split)
        assert os.path.exists(self.root), f"path {self.root} does not exist"

        super().__init__(self.root, transform=transform)

        if config["rebuild_dataset"]:
            print(f"OneStepDataset: Rebuild dataset. Root directory: {self.root}")
            self.process()

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["one_step_train"]

    def process(self):
        # If self.processed_paths are present, this function will not be invoked
        H = self.config["history_length"]
        W = self.config["sequence_length"]
        data_list = []
        filter_logger = AverageMeter() 
        for vid_path in sorted(Path(self.root).glob("*")):
            if "processed" in os.path.basename(vid_path):
                continue

            if str(vid_path).split("/")[-1] in self.config["skip_data_folder"]:
                print(f"skipping folder {str(vid_path)}")
                continue
            print(f'processing {vid_path}')

            state_list = read_video_data(vid_path)
            
            # shift_state_dict_seq(state_list)
            # # only shift validation set 
            # if 'validation' in self.root:
            #     shift_state_dict_seq(state_list)

            # merge all dictionaries, each of which is a dict of a trajectory
            from utils_general import break_trajectory_dic_into_sequence
            seqs = []
            for trajectory in state_list:
                seq = break_trajectory_dic_into_sequence(trajectory)
                seqs += seq[: len(seq) - len(seq) % (H + W)]
                seqs += [None]
            state_list = seqs

            state_list = replace_consecutive_failing_elements(state_list,
                                                              lambda x: x['bubble_pcs'][..., 2].mean() < 0.27,
                                                              num_consecutive=3)
            # state_list = list(filter(lambda x: x is not None, state_list))
            # print(f'\t {self.root}: {len(state_list)}')
            
            for i in range(
                random.randint(0, self.config["every_n_frame"] - 1), len(state_list) - (H + W) + 1, self.config["every_n_frame"]  
            ):
                # history frames
                state_window = state_list[i : i + H + W]
                if None in state_window:
                    filter_logger.update(0)
                    continue
                filter_logger.update(1)

                state_window = downsample_points_state_dict_seq(state_window, self.config)
                if self.config["fix_bubble"]:
                    state_window = replicate_first_bubble_pcs_for_all(state_window)

                data = construct_graph_from_video(
                    self.config, state_window, target=True
                )
                assert (data.object_cls == data.object_cls[0]).all(), 'object_cls should be the same for all steps'

                data_list.append(data)

        print(f'{filter_logger.avg} of the trajectories is kept')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class RolloutDataset(pyg.data.InMemoryDataset):
    def __init__(
        self,
        config,
        split="validation",
        transform=None,
    ):
        self.config = config
        self.root = os.path.join(DATA_DIR, config["data_dir_prefix"], split)
        self.test_seq_len = 32

        self.gt_state_seqs = []

        all_state_list = []
        for idx, vid_path in enumerate(sorted(Path(self.root).glob("*"))):
            if "processed" in os.path.basename(vid_path):
                continue
            if str(vid_path).split("/")[-1] in self.config["skip_data_folder"]:
                print(f"skipping folder {str(vid_path)}")
                continue
            print(f'processing {vid_path}')

            state_list = read_video_data(vid_path)
            # shift_state_dict_seq(state_list)

            # merge all dictionaries, each of which is a dict of a trajectory
            from utils_general import break_trajectory_dic_into_sequence
            seqs = []
            for trajectory in state_list:
                seq = break_trajectory_dic_into_sequence(trajectory)
                seqs += seq[: len(seq) - len(seq) % self.test_seq_len]
                seqs += [None]
            state_list = seqs

            all_state_list += state_list

            # pos_dict = OrderedDict(
            #     object_obs=(0, object_obs[his_len - 1, :, :3]),  # format: (start_index, position)
            #     inhand=(n_points_object, in_hand_object_obs[his_len - 1, :, :3]),
            #     bubble=(n_points_object + n_points_inhand, bubble_gt[his_len - 1, :, :3]),
            # )
            #
            # # object point cloud
            # object_obs = torch.stack([torch.from_numpy(random_sample_from_first_dim(d["object_pcs"][0], n))
            #                           for d in state_list])  # assume one object
            #
            # # in-hand object point cloud, pose obtained by estimation
            # in_hand_object_obs = torch.stack([torch.from_numpy(random_sample_from_first_dim(d["inhand_object_pcs"], n))
            #                                   for d in state_list])
            #
            # # bubble point cloud, ground truth (from tactile sensor)
            # bubble_gt = torch.stack(
            #     [torch.from_numpy(random_sample_from_first_dim(d["bubble_pcs"].reshape(-1, 6), 2 * n))
            #      for d in state_list])  # view two bubbles as one

        all_state_list = replace_consecutive_failing_elements(all_state_list,
                                                              lambda x: x['bubble_pcs'][..., 2].mean() < 0.27,
                                                              num_consecutive=3)
        # all_state_list = list(filter(lambda x: x is not None, all_state_list))

        # data_list = []
        # construct graph, using 16 bubble_pcsframes as a trajectory
        for idx in range(
            0, len(all_state_list), self.config["every_n_frame"]
        ):    
            if idx + self.test_seq_len > len(all_state_list):
                continue
            
            trajectory = all_state_list[idx:idx + self.test_seq_len]
            if None in trajectory:
                continue
            trajectory = downsample_points_state_dict_seq(trajectory, self.config)
            if self.config["fix_bubble"]:
                    trajectory = replicate_first_bubble_pcs_for_all(trajectory)

            gt_state_seq = []
            for d in trajectory:
                bubble_gt = preprocess_points_to_particles(d["bubble_pcs"].reshape(-1, d["bubble_pcs"].shape[-1]),
                                                           config["particles_per_obj"])
                flows = bubble_gt[:, -2:]
                bubble_gt = bubble_gt[:, :-2]

                object_obs = get_object_pcs_from_state_dict(d, config)
                inhand = preprocess_points_to_particles(d["inhand_object_pcs"],
                                                          config["particles_per_obj"])
                bubble = np.concatenate(
                        (
                            bubble_gt,
                            np.tile([1.0, 0.0, 0.0],
                                    (bubble_gt.shape[0], 1)),
                        ),
                        axis=-1,
                    )

                data_dic = dict(
                    object_obs=object_obs,  # squeeze the second dim
                    inhand=inhand,
                    bubble=bubble,
                    forces=d["forces"],
                    flows=flows,
                    # pressure=d["pressure"],
                    object_cls=d["object_cls"], 
                    # rand_index =d['rand_index']
                )

                gt_state_seq.append(data_dic)

            self.gt_state_seqs.append(gt_state_seq)

        super().__init__(self.root, transform=transform)

        if config["rebuild_dataset"]:
            print(f"RolloutDataset: Rebuild dataset. Root directory: {self.root}")
            self.process()

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f"rollout_test"]

    def process(self):
        all_state_list = []
        filter_logger = AverageMeter()
        for idx, vid_path in enumerate(sorted(Path(self.root).glob("*"))):
            if "processed" in os.path.basename(vid_path):
                continue
            if str(vid_path).split("/")[-1] in self.config["skip_data_folder"]:
                print(f"skipping folder {str(vid_path)}")
                continue
            print(f'processing {vid_path}')

            state_list = read_video_data(vid_path)
            # shift_state_dict_seq(state_list)

            # merge all dictionaries, each of which is a dict of a trajectory
            from utils_general import break_trajectory_dic_into_sequence
            seqs = []
            for trajectory in state_list:
                seq = break_trajectory_dic_into_sequence(trajectory)
                seqs += seq[: len(seq) - len(seq) % self.test_seq_len]
                seqs += [None]
            state_list = seqs

            # if the path is wrong (no readable file), skip
            if len(state_list) == 0:
                continue

            all_state_list += state_list

        all_state_list = replace_consecutive_failing_elements(all_state_list,
                                                              lambda x: x['bubble_pcs'][..., 2].mean() < 0.27,
                                                              num_consecutive=3)
        # all_state_list = list(filter(lambda x: x is not None, all_state_list))
        
        data_list = []
        # construct graph, using 16 frames as a trajectory
        for idx in range(
            0, len(all_state_list), self.config["every_n_frame"]
        ):
            if idx + self.test_seq_len > len(all_state_list):
                continue

            trajectory = all_state_list[idx:idx + self.test_seq_len]
            if None in trajectory:
                filter_logger.update(0)
                continue
            filter_logger.update(1)

            trajectory = downsample_points_state_dict_seq(trajectory, self.config)
            if self.config["fix_bubble"]:
                    trajectory = replicate_first_bubble_pcs_for_all(trajectory)

            data = construct_graph_from_video(self.config, trajectory, target=True, trace=False)
            data_list.append(data)

        print(f'{filter_logger.avg} of the trajectories is kept')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class DynamicsDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.config = config

    def setup(self, stage):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            # self.train = OneStepDataset(self.config, split="train")
            # self.val = OneStepDataset(self.config, split="valid")

            # dataset = OneStepDataset(self.config, split="train")

            # # Define the sizes of training and validation sets
            # train_size = int(0.8 * len(dataset))  # 80% for training
            # val_size = len(dataset) - train_size  # Remaining 20% for validation

            # # Split the dataset into training and validation sets
            # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            train_set = OneStepDataset(self.config, split="train")
            val_set = OneStepDataset(self.config, split="validation")

            self.train, self.val = train_set, val_set

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test = ConcatDataset(
                (
                    RolloutDataset(self.config, split="validation"),
                    # RolloutDataset(self.config, split="train"),
                    # RolloutDataset(self.config, split="valid"),
                )
            )

        if stage == "predict":
            raise NotImplementedError
            # self.predict = RolloutDataset(self.config)

    def train_dataloader(self):
        return pyg.loader.DataLoader(
            self.train,
            batch_size=self.config["train_batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=True,
        )

    def val_dataloader(self):
        return pyg.loader.DataLoader(
            self.val,
            batch_size=self.config["train_batch_size"],
            num_workers=self.config["num_workers"],
        )

    def test_dataloader(self):
        return pyg.loader.DataLoader(
            self.test,
            batch_size=self.config["test_batch_size"],
            num_workers=self.config["num_workers"],
        )

    def predict_dataloader(self):
        return pyg.loader.DataLoader(
            self.predict,
            batch_size=self.config["test_batch_size"],
            num_workers=self.config["num_workers"],
        )

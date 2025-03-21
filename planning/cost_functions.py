import sys
# sys.path.append('/svl/u/boai/robopack')     # change path accordingly
sys.path.append('~/github/robopack')     # change path accordingly
from dynamics.loss import Chamfer
import scipy
import torch
import numpy as np

from perception.utils_cv import load_h5_data


def pointcloud_cost_function(predictions, goal_points, last_state_only=False):
    # return the reward for each prediction
    # prediction is a dict containing the following keys:
    # - object_obs: [N_samples, T, N_points, 6]
    # - inhand: [N_samples, T, N_points, 6]
    # - bubble: [N_samples, T, N_points, 6]
    # first 3 channels are position, last 3 channels are visual (RGB)

    if last_state_only:
        states = predictions['object_obs'][:, -1:, :, :3]
    else:
        states = predictions['object_obs'][..., :3]

    n_samples, T, n_points, vec_dim = states.shape
    targets = np.tile(goal_points[np.newaxis, np.newaxis], (n_samples, T, 1, 1))
    states, targets = states.reshape(n_samples * T, n_points, vec_dim), \
                      targets.reshape(n_samples * T, n_points, vec_dim)

    # import pdb; pdb.set_trace()
    # box center MSE loss
    # costs = np.linalg.norm(states.mean(-2)[:, :2] - targets.mean(-2)[:, :2], axis=-1)
    # MSE loss (when corresponding points are available)
    costs = np.linalg.norm(states - targets, axis=-1).mean(-1)
    # EMD loss (when corresponding points are not available)
    # costs = compute_emd(states, targets)
    costs = -costs.reshape(n_samples, T)

    total_cost = apply_reversed_exponential_decay(costs, 0.8).sum(-1)

    return total_cost


def compute_emd(samples, targets):
    B = samples.shape[0]

    y_ind_list = []
    for i in range(B):
        cost_matrix = scipy.spatial.distance.cdist(samples[i], targets[i])
        try:
            ind1, ind2 = scipy.optimize.linear_sum_assignment(
                cost_matrix, maximize=False
            )
        except:
            print("Error in linear sum assignment!")

        y_ind_list.append(ind2)

    y_ind = np.stack(y_ind_list)
    batch_ind = torch.arange(B)[:, None]

    emd_pos = np.mean(np.linalg.norm(samples - targets[batch_ind, y_ind], axis=-1), axis=-1)

    return emd_pos


def position_cost_function(predictions, goal, last_state_only=False):
    # return the reward for each prediction
    # prediction is a dict containing the following keys:
    # - object_obs: [N_samples, T, N_points, 6]
    # - inhand: [N_samples, T, N_points, 6]
    # - bubble: [N_samples, T, N_points, 6]
    # first 3 channels are position, last 3 channels are visual (RGB)

    if last_state_only:
        states = predictions['object_obs'][:, -1:, :, :2].mean(2)
    else:
        states = predictions['object_obs'][..., :2].mean(2)
    goal = goal[:2]

    per_step_costs = -np.linalg.norm(states - goal, axis=-1)[..., :]
    total_cost = apply_reversed_exponential_decay(per_step_costs, 0.2).sum(-1)

    return total_cost


def apply_reversed_exponential_decay(costs, decay_factor):
    """
    Apply exponential decay to a 2D array of costs with inverse proportionality to time step.

    Args:
    - costs (numpy.ndarray): 2D array of costs with shape (N, T).
    - decay_factor (float): The decay factor (c) for the exponential decay.

    Returns:
    - decayed_costs (numpy.ndarray): 2D array of decayed costs with shape (N, T).
    """
    # Ensure that the decay factor is a float in the range (0, 1].
    if not (0 < decay_factor <= 1):
        raise ValueError("Decay factor must be in the range (0, 1].")

    # Get the number of batches (N) and time steps (T).
    N, T = costs.shape

    # Create an array to represent the time step indices.
    time_indices = np.arange(T)

    # Calculate the exponential decay factors for each time step.
    decay_factors = decay_factor ** (T - time_indices - 1)

    # Apply the decay factors to the costs.
    decayed_costs = costs * decay_factors

    return decayed_costs


def get_goal_box_point_cloud(box_target_position, box_target_orientation, box_points=None, object_masks=None):
    # Get the "canonical" point cloud of a box, picked from a random trajectory in the data, but
    # transform it to the target position and orientation. This can then be used to compute losses
    # for a particular planning goal.
    # the orientation is specified in degrees (not radians) and is [roll, pitch, yaw]
    if box_points is None:
        box_points = load_h5_data('/home/albert/github/robopack/asset/box3_seq_1.h5')['object_pcs'][0][0]   # (N, 3)
    else:
        assert object_masks is not None, "object_masks must be provided if box_points is provided, to consistently sample points for the goal"
    from perception.utils_pc import center_and_rotate_to, farthest_point_sampling_dgl
    start_position = np.mean(box_points, axis=0)
    box_target_position = np.array(box_target_position) + np.array(start_position)
    goal = center_and_rotate_to(box_points, box_target_position, box_target_orientation)
    if object_masks is not None:
        goal = goal[object_masks[0]]
    else:
        goal = farthest_point_sampling_dgl(goal, 20)
    return goal


def packing_cost_function_rodtip(predictions_inhand, predictions_object, goal_points, actions, last_state_only=False):
    """
    Computes planning cost based on rod points with the lowest z-coordinate
    """
    # return the reward for each prediction
    # prediction is a dict containing the following keys:
    # - object_obs: [N_samples, T, N_points, 6]
    # - inhand: [N_samples, T, N_points, 6]
    # - bubble: [N_samples, T, N_points, 6]
    # first 3 channels are position, last 3 channels are visual (RGB)
    if last_state_only:
        states = predictions_inhand[:, -1:, :, :3]
    else:
        states = predictions_inhand[..., :3]

    k = 5    # number of points to represent the tip 
    
    # Sort the points based on the z-coordinate (third column, index 2) within each batch
    sorted_indices = np.argsort(states[:, :, :, 2], axis=2)

    # Get the first 'k' indices within each batch to get the 'k' points with the lowest z-coordinate
    k_lowest_points = np.take_along_axis(states, sorted_indices[:, :, :k, np.newaxis], axis=2)
    
    states = k_lowest_points[..., :2]
    goal_points = goal_points[..., :2]
    
    n_samples, T, n_points, vec_dim = states.shape
    targets = np.tile(goal_points[np.newaxis, np.newaxis], (n_samples, T, 1, 1))
    states, targets = states.reshape(n_samples * T, n_points, vec_dim), \
                      targets.reshape(n_samples * T, -1, vec_dim)
    
    dis_x_to_nearest_y, dis_y_to_nearest_x = Chamfer.compute(torch.from_numpy(states), torch.from_numpy(targets), keep_dim=True)
    costs = -dis_x_to_nearest_y.mean(1).reshape(n_samples, T)
    
    if last_state_only:
        return apply_reversed_exponential_decay(costs, 0.01).sum(-1)
    else:
        exploration = np.clip((actions[:, :1] ** 2).sum(2).sum(1) * 5, 0, 0.001)   # encourage switch action
        total_cost = apply_reversed_exponential_decay(costs, 0.01).sum(-1) + exploration   # encourage switch action
        return total_cost


def packing_cost_function_object(predictions_inhand, predictions_object, goal_points, actions, last_state_only=False):
    """
    Computes planning cost based on objects points.
    It should make sure (1) objects are in the target region, and (2) the objects only occupy a space as small as possible 
    """
    # return the reward for each prediction
    # prediction is a dict containing the following keys:
    # - object_obs: [N_samples, T, N_points, 6]
    # - inhand: [N_samples, T, N_points, 6]
    # - bubble: [N_samples, T, N_points, 6]
    # first 3 channels are position, last 3 channels are visual (RGB)
    if last_state_only:
        states = predictions_object[:, -1:, :, :2]
        predictions_inhand = predictions_inhand[:, -1:]
    else:
        states = predictions_object[..., :2]
    goal_points = goal_points[..., :2]# - np.array([0, 0.02])

    n_samples, T, n_points, vec_dim = states.shape
    targets = np.tile(goal_points[np.newaxis, np.newaxis], (n_samples, T, 1, 1))
    states, targets = states.reshape(n_samples * T, n_points, vec_dim), \
                      targets.reshape(n_samples * T, -1, vec_dim)
    
    # compute the distance from objects points to the nearest targets points (as small as possible)
    # and the distance from targets points to nearest object points   (as large as possible)
    # WARNING: if the input size is too large causing out of memory, it might show "Killed"
    dis_x_to_nearest_y, dis_y_to_nearest_x = Chamfer.compute(torch.from_numpy(states), torch.from_numpy(targets), keep_dim=True)
    costs = -dis_x_to_nearest_y.mean(1).reshape(n_samples, T) + dis_y_to_nearest_x.mean(1).reshape(n_samples, T)
    # Note that the loss range from 0.004-0.005 (not pushable at all) to 0.008-0.009 (completely pushable).
    # penalize if lowest rod points are too high. compute the 5th lowest point's height
    z_values = predictions_inhand[..., 2]
    # reshape to [samples*T, N_points]
    z_values = z_values.reshape(n_samples * T, 20)
    # sort the z values
    sorted_zvalues = np.sort(z_values, axis=1)
    k_lowest_points = sorted_zvalues[:, :3]
    # reshape to [samples, T, 5]
    k_lowest_points = k_lowest_points.reshape(n_samples, T, 3)
    k_lowest_points = k_lowest_points.mean(2)
    # penalize for being 0.1 above the table or higher
    lowpoint_penalty = np.clip(k_lowest_points - 0.11, 0, 0.1) * 100
    low_y_bonus = predictions_inhand[..., 1].mean(-2).mean(-1) * 0.05
    print("lowpoint penalty",   lowpoint_penalty.mean(-1))

    if last_state_only:
        return apply_reversed_exponential_decay(costs, 0.01).sum(-1)
    else:
        exploration = np.clip((actions[:, :1] ** 2).sum(2).sum(1) * 5, 0, 0.001)   # encourage switch action
        # total_cost = apply_reversed_exponential_decay(costs, 0.01).sum(-1) + exploration - lowpoint_penalty.mean(-1)
        chamfer_accum = apply_reversed_exponential_decay(costs, 0.01).sum(-1)
        low_y_bonus[chamfer_accum < 0.007] = 0
        chamfer_accum = np.clip(chamfer_accum, 0, 0.007)
        total_cost = chamfer_accum + exploration - lowpoint_penalty.mean(-1) - low_y_bonus
        return total_cost


def packing_cost_function_inhand_v1(pred_inhand_points, other_object_points, goal_points, actions, last_state_only=False):
        """
        Compute planning cost based on rod points
        """
        cost_dim = 2        # how many dims among x y z are used to compute the cost, e.g., 2 for x y only 
        if last_state_only:
            states = pred_inhand_points[:, -1:, :, :cost_dim]
            other_object_points = other_object_points[:, -1:, :, :cost_dim]
        else:
            states = pred_inhand_points[..., :cost_dim]
            other_object_points = other_object_points[..., :cost_dim]

        goal_points = goal_points[..., :cost_dim]

        n_samples, T, n_points, vec_dim = states.shape
        targets = np.tile(goal_points[np.newaxis, np.newaxis], (n_samples, T, 1, 1))
        states, targets = states.reshape(n_samples * T, n_points, vec_dim), \
                        targets.reshape(n_samples * T, -1, vec_dim)
        
        dis_x_to_nearest_y, dis_y_to_nearest_x = Chamfer.compute(torch.from_numpy(states), torch.from_numpy(targets), keep_dim=True)
        costs = -dis_x_to_nearest_y.mean(1).reshape(n_samples, T)
        
        if last_state_only:
            return apply_reversed_exponential_decay(costs, 0.01).sum(-1)
        else:
            # encourage switch action for exploration. This loss should be at the scale of 1e-2
            # tune the scale if necessary
            exploration = (3 * (actions[:, :1] ** 2).sum(2).sum(1))
            # print("exploration bonus", exploration)
            
            total_cost = apply_reversed_exponential_decay(costs, 0.01).sum(-1) + exploration
            
            return total_cost
    
    
def packing_cost_function_inhand_v2(pred_inhand_points, other_object_points, goal_points, actions, last_state_only=False):
        """
        Compute planning cost based on rod points, 
        plus quite a few exploration terms that encourage row switching. 
        """
        if last_state_only:
            states = pred_inhand_points[:, -1:, :, :2]
            other_object_points = other_object_points[:, -1:, :, :2]
        else:
            states = pred_inhand_points[..., :2]
            other_object_points = other_object_points[..., :2]

        goal_points = goal_points[..., :2]

        n_samples, T, n_points, vec_dim = states.shape
        targets = np.tile(goal_points[np.newaxis, np.newaxis], (n_samples, T, 1, 1))
        states, targets = states.reshape(n_samples * T, n_points, vec_dim), \
                        targets.reshape(n_samples * T, -1, vec_dim)
        
        dis_x_to_nearest_y, dis_y_to_nearest_x = Chamfer.compute(torch.from_numpy(states), torch.from_numpy(targets), keep_dim=True)
        costs = -dis_x_to_nearest_y.max(1).values.reshape(n_samples, T)

        # for each sample, if the sum of action magnitudes is less than M, and the y value of the inhand point is larger than -0.25, set the cost to be -100
        # actions is (num_samples, time, 3)
        action_mags = np.linalg.norm(actions, axis=-1)
        action_mags = action_mags.sum(-1)
        inhand_ys_high = pred_inhand_points[..., -1, :].mean(-2)[..., 1] > -0.25
        action_mags_low_or_inhand_ys_high = (action_mags < 0.1) & inhand_ys_high
        costs[action_mags_low_or_inhand_ys_high] = -100

        # new_costs = torch.zeros_like(costs)
        # new_costs[..., -1] = costs[..., -1] - costs[..., 0]
        # costs = new_costs * 2
        # also, for each object, if it is leaving the box region, add a cost
        # get the minimum y value from the goal points
        min_y = goal_points[..., 1].min(-1)
        # compute set of object points that are below the minimum y value and penalize each one according to the distance to the minimum y value
        # other_object_points is (num_samples, time, 100 points, 2)
        # remove other_object_points that have x values less than 0.1
        # iterate over groups of 20 points. If any of the points in the group has avg x value less than 0.1, remove all points in the group
        print("Initial number of non-inhand points", other_object_points.shape[2])
        filter_objs = []
        for i in range(0, other_object_points.shape[2], 20):
            obj_points = other_object_points[:, :, i:i+20]
            if obj_points[:, :, :, 0].mean() > 0.1:
                filter_objs.append(obj_points)
        other_object_points = np.concatenate(filter_objs, axis=2)
        print("After filtering number of non-inhand points", other_object_points.shape[2])
        y_cost = np.maximum(0, min_y - other_object_points[..., 1])
        # print("Original cost: ", costs)
        y_cost = y_cost.sum(-1) # sum over points
        # print("y cost: ", y_cost)
        costs -= y_cost * 0.7

        if last_state_only:
            return apply_reversed_exponential_decay(costs, 0.01).sum(-1)
        else:
            # encourage switch action for exploration. This loss should be at the scale of 1e-2
            # tune the scale if necessary
            exploration =(1 * (actions[:, :1] ** 2).sum(2).sum(1))
            print("exploration bonus", exploration)
            # breakpoint()
            
            total_cost = apply_reversed_exponential_decay(costs, 0.01).sum(-1) + exploration
            
            return total_cost
    
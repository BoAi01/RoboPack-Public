import numpy as np
import torch
import time
import os
import pdb
import threading
from queue import Queue

from torch_geometric.data import Batch
from perception.utils_cv import find_point_on_line


def convert_state_dict_to_state_list(predictions_dict):
    """
    Convert a dictionary of predictions to a list of predictions, where each prediction is a dictionary
    :param predictions_dict: A dictionary where each key is a string and each value is a numpy array
        of shape [batch_size, ...] or [time, ...]
    :return: A list of dictionaries, where each dictionary has the same keys as the input dictionary and each value
        is a numpy array of shape [...]
    """
    # assert values have the same length
    length = None
    for key, values in predictions_dict.items():
        if length is None:
            length = values.shape[0]
        else:
            assert length == values.shape[0], f'length of values for key {key} is not the same as other keys {length}'

    state_list = []
    for t in range(length):
        state_dict = {k: v[t] for k, v in predictions_dict.items()}   # the first dim is batch
        state_list.append(state_dict)

    return state_list


class MPPIOptimizer:
    def __init__(
        self,
        sampler,
        model,
        objective,
        a_dim,
        horizon,
        num_samples,
        gamma,
        num_iters=3,
        init_std=0.5,
        log_every=1,
    ):
        self.obj_fn = objective
        self.sampler = sampler
        self.model = model
        self.horizon = horizon
        self.a_dim = a_dim
        self.num_samples = num_samples
        self.gamma = gamma
        self.num_iters = num_iters
        self.init_std = np.array(init_std)
        
        if len(self.init_std.shape) == 0:  # a single number
            self.init_std = self.init_std[None].repeat(self.horizon, axis=0)
        elif len(self.init_std.shape) == 1 and self.init_std.shape[0] == 3:
            self.init_std = np.expand_dims(self.init_std, 0).repeat(self.horizon, axis=0)
        else:
            raise NotImplementedError(f"Unknow std shape {self.init_std.shape}")
        
        self.log_every = log_every
        self._model_prediction_times = list()

    def update_dist(self, samples, scores):
        # actions: array with shape [num_samples, time, action_dim]
        # scores: array with shape [num_samples]
        scaled_rews = self.gamma * (scores - np.max(scores))    # all positive
        
        # exponentiated scores
        exp_rews = np.exp(scaled_rews)
        
        # weigh samples by exponentiated scores to bias sampling for future iterations
        softmax_prob = exp_rews / (np.sum(exp_rews, axis=0) + 1e-10)
        mu = np.sum(softmax_prob * samples, axis=0)
        print(f"max prob in the softmax prob list: {softmax_prob.max()}")
        # mu = np.sum(exp_rews * samples, axis=0) / (np.sum(exp_rews, axis=0) + 1e-10)
        # pdb.set_trace()
        
        # prior knowledge: the end effector should not move along z axis
        mu[:, -1] = 0
        
        return mu, self.init_std

    def plan(
            self,
            t,
            log_dir,
            observation_batch,
            action_history,
            goal,
            init_mean=None,
            visualize_top_k=False,
            return_best=False,
    ):
        start_start_time = time.time()
        
        os.makedirs(log_dir, exist_ok=True)
        
        box_points = observation_batch[2][0, :20, -1]   
        box_center_xy, goal_center_xy = box_points.mean(0).cpu().numpy()[:2], goal.mean(0)[:2]
        starting_position = find_point_on_line(goal_center_xy, box_center_xy, 0.1)
        
        # compute a heurisic mean
        guessed_mean_xy = (goal_center_xy - box_center_xy) / np.linalg.norm(goal_center_xy - box_center_xy)  # unit vector
        guessed_mean_xy = guessed_mean_xy * 0.006
        
        if init_mean is not None and len(init_mean) > 0:
            mu = np.zeros((self.horizon, self.a_dim))
            init_mean = init_mean[:self.horizon]    # truncate if the mean is longer than needed
            mu[: len(init_mean)] = init_mean
            mu[len(init_mean) :] = init_mean[-1]
        else:
            mu = np.zeros((self.horizon, self.a_dim))
            mu[:, :2] = guessed_mean_xy[None, :]
        std = self.init_std

        best_action, best_action_prediction = None, None
        self._model_prediction_times = []
        for iter in range(self.num_iters):
            start_time = time.time()

            action_samples = self.sampler.sample_actions(self.num_samples, mu, std)
            action_samples = np.clip(action_samples, -0.006, 0.006)
            
            curr_ee_xy = observation_batch[2][0, 20:40, -1, :2].mean(0).detach().cpu().numpy()      # get the ee xy coordinate by taking the center of the rod
            lead_action = np.zeros((self.num_samples, 1, self.a_dim))
            lead_action[:, :, :2] = starting_position - curr_ee_xy
            action_samples = np.concatenate((lead_action, action_samples), axis=1)
            
            # print(f"Sampling takes {time.time() - start_time}")
            start_time = time.time()

            pred_start_time = time.time()
            with torch.no_grad():
                predictions = self.model.predict_step(observation_batch, action_history, action_samples)
            
            prediction_time = time.time() - pred_start_time
            self._model_prediction_times.append(prediction_time)
            print(
                f"Out of {len(self._model_prediction_times)} iterations, "
                f"median prediction time {np.median(self._model_prediction_times)}"
            )
            
            start_time = time.time()
            rewards = self.obj_fn(predictions, goal, last_state_only=False)[:, None, None] # shape [num_samples, 1, 1]
            top_k = max(visualize_top_k, 1)
            
            best_prediction_inds = np.argsort(-rewards.flatten())[:top_k]
            # best_rewards = [rewards[i] for i in best_prediction_inds]
            # best_actions = [new_action_samples[x] for x in best_prediction_inds]
            # print("best rewards:", best_rewards)
            # print('best actions:', best_actions)
            
            last_state_rewards = self.obj_fn(predictions, goal, last_state_only=True)[:, None, None] # shape [num_samples, 1, 1]
            best_last_state_indices = np.argsort(-last_state_rewards.flatten())[:top_k]
            best_last_state_rewards = [last_state_rewards[i] for i in best_last_state_indices]
            # print(f'last state rewards: {best_last_state_rewards}')
            
            # print(f'Computing rewards takes {time.time() - start_time}')
            
            if iter == self.num_iters - 1:
                end_end_time = time.time()
                
            if t % self.log_every == 0 and iter == self.num_iters - 1 and visualize_top_k > 0:

                log_folder = f"{log_dir}/mppi_step_{t}_iter{iter}_plan"
                    
                self.log_best_plans(
                    log_folder, predictions, goal, best_prediction_inds, best_last_state_rewards, self.sampler.horizon
                )
            
            start_time = time.time()
            mu, std = self.update_dist(action_samples[:, 1:], rewards)
            print(f"mu shape = {mu.shape}, means over all steps = {mu.mean(0)}")
            # print(f'Updating distribution takes {time.time() - start_time}')
            # print(f'total time for the iteration: {time.time() - start_start_time}\n')

            best_action = action_samples[best_prediction_inds[0]]
            best_action_prediction = {k: v[best_prediction_inds[0]] for k, v in predictions.items()}
            best_action_prediction = convert_state_dict_to_state_list(best_action_prediction)

        print(f'Total time for planning: {end_end_time - start_start_time}')

        if return_best:
            return mu, best_action, best_action_prediction
        else:
            return mu

    def log_best_plans(self, filename, predictions, goal, best_prediction_inds, best_rewards, horizon):
        from planning.visualize import create_frames_for_pcs, save_moviepy
        predictions = np.concatenate((
            predictions['object_obs'][..., :3],
            predictions['inhand'][..., :3],
            predictions['bubble'][..., :3],
        ), axis=2)
        total_T = predictions.shape[1]
        observed_his_len = total_T - horizon

        for i, prediction in enumerate(predictions[best_prediction_inds]):
            status_str_lambda = lambda x: "Integrating history.. " if x < observed_his_len else "Planning.."
            vis_frames = create_frames_for_pcs(prediction, goal, multiview=False, title=f"last-state reward = {best_rewards[i]}",
                                               extra_title_func=status_str_lambda)
            vis_frames = [np.asarray(frame) for frame in vis_frames]
            # from utils.visualizer import play_and_save_video
            # play_and_save_video(vis_frames, f"{filename}_{i}.mp4", fps=5)
            save_moviepy(vis_frames, f"{filename}_{i}.mp4", fps=5)
            print("Logging to", filename)


class EEPositionPlanner:
    def __init__(
            self,
            sampler,
            model,
            objective,
            horizon,
            num_samples,
            gamma,
            logging_thread,
            num_iters=3,
            log_every=1,
            theta_std=np.pi/3,
            alpha_std=0.25,
    ):
        self.obj_fn = objective
        self.sampler = sampler
        self.model = model
        self.horizon = horizon
        self.num_samples = num_samples
        self.gamma = gamma
        self.num_iters = num_iters
        self.theta_std = theta_std
        self.alpha_std = alpha_std
        self.logging_thread = logging_thread

        self.log_every = log_every
        self._model_prediction_times = list()
        
    def get_softmax_prob_from_rewards(self, scores):
        scaled_rews = self.gamma * (scores - np.max(scores))  # all positive
        exp_rews = np.exp(scaled_rews)  # exponentiated scores

        # weigh samples by exponentiated scores to bias sampling for future iterations
        softmax_prob = exp_rews / (np.sum(exp_rews, axis=0) + 1e-10)
        
        return softmax_prob

    def update_theta(self, theta_samples, scores):
        # actions: array with shape [num_samples, time, action_dim]
        # scores: array with shape [num_samples]

        softmax_prob = self.get_softmax_prob_from_rewards(scores)

        # update mean of theta
        mu_theta = np.sum(softmax_prob * theta_samples, axis=0)
        print(f"max prob in the softmax prob list: {softmax_prob.max()}")

        return mu_theta, self.theta_std
    
    def update_particle_weights(self, contact_particle_indices, scores):
        # Calculate unique indices and their counts
        counts = np.zeros(20, dtype=int)
        unique_indices, unique_counts = np.unique(contact_particle_indices, return_counts=True)
        counts[unique_indices] = unique_counts
        
        # Update the sum of scores for each index using np.bincount
        score_sum = np.bincount(contact_particle_indices, weights=scores[:, 0, 0])
        score_sum = np.concatenate((score_sum, np.zeros(20 - len(score_sum))), axis=0)    

        # Calculate mean scores
        mean_scores = score_sum / (counts + 1)
        new_softmax_prob = self.get_softmax_prob_from_rewards(mean_scores)
        
        return new_softmax_prob
    
    def update_alpha(self, lead_portions, scores):
        if isinstance(lead_portions, list):
            lead_portions = np.array(lead_portions)
        
        softmax_prob = self.get_softmax_prob_from_rewards(scores)
        
        # update mean of theta
        lead_portion_mu = np.sum(softmax_prob.squeeze(-1) * lead_portions, axis=0)
        
        return lead_portion_mu, self.alpha_std

    def plan(
            self,
            t,
            log_dir,
            observation_batch,
            action_history,
            goal,
            theta_mean=None,
            visualize_top_k=False,
            return_best=False,
    ):
        start_start_time = time.time()

        os.makedirs(log_dir, exist_ok=True)
        
        box_points = observation_batch[2][0, :20, -1]   
        box_center, goal_center = box_points.mean(0).cpu().numpy(), goal.mean(0)
        guessed_theta = np.arctan2(goal_center[1] - box_center[1], goal_center[0] - box_center[0])
        print("guessed_theta:", guessed_theta)

        if theta_mean is not None and len(theta_mean) > 0:
            theta_mu = np.zeros((self.horizon, 1))
            theta_mean = theta_mean[:self.horizon]  # truncate if the mean is longer than needed
            theta_mu[: len(theta_mean)] = theta_mean
            theta_mu[len(theta_mean):] = theta_mean[-1]
        else:
            theta_mu = np.zeros((self.horizon, 1)) + guessed_theta
        theta_std = np.repeat(np.array([self.theta_std])[:, np.newaxis], self.horizon, axis=0)
        
        # initialize as None (uniform distribution by default)
        alpha_mu, alpha_std = None, self.alpha_std

        best_action, best_future = None, None
        particle_weights = np.ones(20) / 20
        self._model_prediction_times = []
        for iteration_index in range(self.num_iters):
            start_time = time.time()

            starting_positions, actions, theta_samples, contact_particle_indices, lead_portions = self.sampler.sample_actions(self.num_samples,
                                                                                                                box_points,
                                                                                                                particle_weights,
                                                                                                                theta_mu, theta_std,
                                                                                                                alpha_mu, alpha_std)

            print(f"Sampling takes {time.time() - start_time}")
            start_time = time.time()

            # add a new action to indicate the first ee movement to reach the init position
            curr_ee_xy = observation_batch[2][0, 20:40, -1, :2].mean(0).detach().cpu().numpy()      # get the ee xy coordinate by taking the center of the rod
            lead_action = starting_positions - curr_ee_xy[None, None, :]
            actions = np.concatenate((lead_action, actions), axis=1)
            actions = np.concatenate((actions, np.zeros((self.num_samples, actions.shape[1], 1))), axis=-1)  # add a zero action at the end

            # predict future
            pred_start_time = time.time()
            with torch.no_grad():
                predictions = self.model.predict_step(observation_batch, action_history, actions)

            prediction_time = time.time() - pred_start_time
            self._model_prediction_times.append(prediction_time)
            print(
                f"Out of {len(self._model_prediction_times)} iterations, "
                f"median prediction time {np.median(self._model_prediction_times)}"
            )

            start_time = time.time()
            rewards = self.obj_fn(predictions, goal, last_state_only=False)[:, None, None]  # shape [num_samples, 1, 1]
            # print(rewards.flatten())
            top_k = max(visualize_top_k, 1)

            best_prediction_inds = np.argsort(-rewards.flatten())[:top_k]
            # best_prediction_inds = np.argsort(-rewards.flatten())[[0, 19, 39]]
            
            # best_rewards = [rewards[i] for i in best_prediction_inds]
            # best_actions = [new_action_samples[x] for x in best_prediction_inds]
            # print("best rewards:", best_rewards)
            # print('best actions:', best_actions)

            last_state_rewards = self.obj_fn(predictions, goal, last_state_only=True)[:, None, None]  # shape [num_samples, 1, 1]
            best_last_state_indices = np.argsort(-last_state_rewards.flatten())[:top_k]  # [[0, 19, 39]] 
            best_last_state_rewards = [last_state_rewards[i] for i in best_last_state_indices]
            # print(f'last state rewards: {best_last_state_rewards}')

            # print(f'Computing rewards takes {time.time() - start_time}')

            if iteration_index == self.num_iters - 1:
                end_end_time = time.time()

            if t % self.log_every == 0 and iteration_index == self.num_iters - 1 and visualize_top_k > 0:
                log_folder = f"{log_dir}/ee_pos_step_{t}_iter{iteration_index}_plan"

                # self.log_best_plans(
                #     log_folder, predictions, goal, best_prediction_inds, best_last_state_rewards, self.sampler.horizon
                # )
                
                # np.save('rewards.npy', rewards)
                # breakpoint()
                
                self.log_best_plans(
                    log_folder, predictions, goal, best_prediction_inds, best_last_state_rewards, self.sampler.num_actions, action_history, actions
                )

            start_time = time.time()
            theta_mu, theta_std = self.update_theta(theta_samples, rewards)
            particle_weights = self.update_particle_weights(contact_particle_indices, rewards)
            alpha_mu, alpha_std = self.update_alpha(lead_portions, rewards)
            # print(particle_weights.max(), particle_weights.min())
            print(f"\tAfter update: theta_mean = {theta_mu}, alpha_mean = {alpha_mu}")
            # print(f'Updating distribution takes {time.time() - start_time}')
            # print(f'total time for the iteration: {time.time() - start_start_time}\n')

            best_action = actions[best_prediction_inds[0]]
            best_future = {k: v[best_prediction_inds[0]] for k, v in predictions.items()}
            best_future = convert_state_dict_to_state_list(best_future)

        print(f'Total time for planning: {end_end_time - start_start_time}')

        if return_best:
            return theta_mu, best_action, best_future[-best_action.shape[0]:]
        else:
            return theta_mu
    def log_best_plans(self, filename, predictions, goal, best_prediction_inds, best_rewards, horizon, action_history,
                       actions):
        from planning.visualize import create_frames_for_pcs, save_moviepy
        predictions = np.concatenate((
            predictions['object_obs'][..., :3],
            predictions['inhand'][..., :3],
            predictions['bubble'][..., :3],
        ), axis=2)
        total_T = predictions.shape[1]
        observed_his_len = total_T - horizon

        for i, prediction in enumerate(predictions[best_prediction_inds]):
            action = actions[i]

            def status_str_lambda(x):
                if x == 0:
                    return "History.."  # no action to show
                elif x < observed_his_len:
                    return "History..."
                    # return f"History.. Action={[f'{num:.3f}' for num in action_history[0, -1, x - 1]]}"  # show the action in history
                else:
                    return f"Planning... Action = {[f'{num:.3f}' for num in action[x - observed_his_len]]}"  # show the planned action
            # status_str_lambda = lambda x: f"{x}: History.. Action = {action_history[0, -1, ].tolist()} " if x < observed_his_len else f"{x}: Planning... Action = {action[x-observed_his_len]}"
            print("Putting logging job")
            self.logging_thread.put({
                'prediction': prediction,
                'goal': goal,
                'filename': f"{filename}_{i}.mp4",
                'title': f"last-state reward = {best_rewards[i].item()}",
                'extra_title_func': True,
                'observed_his_len': observed_his_len,
                'action_history': action_history.cpu(),
                'action': action,
            })

class PackingPushingPlanner:
    def __init__(
            self,
            sampler,
            model,
            objective,
            horizon,
            num_samples,
            # gamma,
            # num_iters=3,
            logging_thread,
            log_every=1,
            # theta_std=np.pi/3,
            # alpha_std=0.25,
    ):
        self.obj_fn = objective
        self.sampler = sampler
        self.model = model
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_iters = 1
        self.logging_thread = logging_thread
        # self.gamma = gamma
        # self.num_iters = num_iters
        # self.theta_std = theta_std
        # self.alpha_std = alpha_std

        self.log_every = log_every
        self._model_prediction_times = list()
        
    def get_softmax_prob_from_rewards(self, scores):
        scaled_rews = self.gamma * (scores - np.max(scores))  # all positive
        exp_rews = np.exp(scaled_rews)  # exponentiated scores

        # weigh samples by exponentiated scores to bias sampling for future iterations
        softmax_prob = exp_rews / (np.sum(exp_rews, axis=0) + 1e-10)
        
        return softmax_prob
    
    @classmethod
    def extract_object_pcs(self, pc_array, return_index=False):
        """
        Given (N, 3), extract objects on the table and in the hand
        """
        n = 20
        object_pcs = pc_array[:-(n * 2)].split(n, dim=0)        # excluding the bubble points 
        object_highest_z = [pc[:, 2].max().item() for pc in object_pcs]
        indices = np.argsort(object_highest_z)      # take the highest 
        inhand_object_pc = object_pcs[indices[-1]]
        table_object_pcs = torch.stack(list(object_pcs[:indices[-1]]) + list(object_pcs[indices[-1]+1:]))
        
        if return_index:
            return inhand_object_pc, table_object_pcs, indices[-1]
        else:
            return inhand_object_pc, table_object_pcs
    
    @classmethod
    def extract_object_pcs_batch(self, pc_array):
        """
        Given (B, N, 3), extract objects on the table and in the hand for each batch element.
        """
        n = 20
        
        # Reshape the input tensor for efficient processing
        batch_size, num_points, _ = pc_array.shape
        pc_array_reshaped = pc_array.view(batch_size, -1, n, 3)
        
        # Calculate the maximum Z values for each point cloud in each batch
        object_max_z = pc_array_reshaped[:, :, :, 2].max(dim=2)[0]
        
        # Find the indices of the objects with the highest Z values
        indices = torch.argsort(object_max_z, dim=1, descending=True)
        
        # Gather the inhand_object_pcs and table_object_pcs
        inhand_object_pc = torch.gather(pc_array_reshaped, 2, indices[:, :1].unsqueeze(2).expand(-1, -1, -1, 3))
        table_object_pcs = torch.gather(pc_array_reshaped, 2, indices[:, 1:].unsqueeze(2).expand(-1, -1, -1, 3))
        
        # Reshape the results to their original shapes
        inhand_object_pc = inhand_object_pc.view(batch_size, -1, 3)
        table_object_pcs = table_object_pcs.view(batch_size, -1, 3)
        
        return inhand_object_pc, table_object_pcs
    
    @classmethod
    def remove_zero_rows(self, input_tensor, return_mask=False):
        input_tensor = input_tensor.clone()
        mask = None
        threshold = 0.1     # points in this neighborhood of the original points are viewed as invalid
        if len(input_tensor.shape) == 2:
            # row_sums = torch.sum(torch.abs(input_tensor), dim=1)
            # mask = row_sums != 0
            mask = ~(input_tensor.abs() < threshold).all(1)
            non_zero_rows = input_tensor[mask]
        elif len(input_tensor.shape) == 3:
            num_objects, num_points, c = input_tensor.shape
            input_tensor = input_tensor.reshape(-1, c)
            # row_sums = torch.sum(torch.abs(input_tensor), dim=1)
            # mask = row_sums != 0
            mask = ~(input_tensor.abs() < threshold).all(1)
            non_zero_rows = input_tensor[mask]
            non_zero_rows = non_zero_rows.reshape(-1, num_points, c)

        if return_mask:
            return non_zero_rows, mask
        else:
            return non_zero_rows
    
    @classmethod
    def infer_bounding_box_from_pc(self, pc):
        """
        Given a point cloud of the initial frame, it tries to infer the goal bounding box region
        """
        nonzero_pc = PackingPushingPlanner.remove_zero_rows(pc.reshape(-1, 3))
        min_x, max_x = nonzero_pc[:, 0].min().item(), nonzero_pc[:, 0].max().item()
        min_y, max_y = nonzero_pc[:, 1].min().item(), nonzero_pc[:, 1].max().item()  # move the region a bit along -y axis

        # expand by 2cm
        min_y -= 0.02
        # Specify the number of points to sample along on axis
        # note the final num of points would be N^3
        N = 10

        # Specify the maximum Z coordinate value
        max_z = 0.1
        
        # maybe even sampling works better
        x_values = np.linspace(min_x, max_x, N)
        y_values = np.linspace(min_y, max_y, N)
        z_values = np.linspace(0, max_z, 1)  # Z values are uniformly sampled from 0 to max_z

        # Create a grid of coordinates using meshgrid
        X, Y, Z = np.meshgrid(x_values, y_values, z_values)

        # Flatten the grid to get the sampled points
        sampled_points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

        # use the points as the goal
        return sampled_points
    
    
    @classmethod
    def infer_bounding_box_from_pc_3d(self, pc):
        """
        Given a point cloud of the initial frame, it tries to infer the goal bounding box region
        """
        nonzero_pc = PackingPushingPlanner.remove_zero_rows(pc.reshape(-1, 3))
        min_x, max_x = nonzero_pc[:, 0].min().item(), nonzero_pc[:, 0].max().item()
        min_y, max_y = nonzero_pc[:, 1].min().item(), nonzero_pc[:, 1].max().item()  # move the region a bit along -y axis
        min_z, max_z = nonzero_pc[:, 2].min().item(), nonzero_pc[:, 2].max().item()
        
        # Specify the number of points to sample along on axis
        # note the final num of points would be N^3
        N = 5

        # # Specify the maximum Z coordinate value
        # max_z = 0.1

        # Generate random (X, Y, Z) coordinates within the specified extents
        # random_x = np.random.uniform(min_x, max_x, N)
        # random_y = np.random.uniform(min_y, max_y, N)
        # random_z = np.random.uniform(0, max_z, N)  # Z values are uniformly sampled from 0 to max_z
                
        # Create an array of (X, Y, Z) coordinates for the sampled points
        # sampled_points = np.column_stack((random_x, random_y, random_z))
        
        # maybe even sampling works better
        x_values = np.linspace(min_x, max_x, N)
        y_values = np.linspace(min_y, max_y, N)
        z_values = np.linspace(0, max_z, N)  # Z values are uniformly sampled from 0 to max_z

        # Create a grid of coordinates using meshgrid
        X, Y, Z = np.meshgrid(x_values, y_values, z_values)

        # Flatten the grid to get the sampled points
        sampled_points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

        # use the points as the goal
        return sampled_points

    def plan(
            self,
            t,
            log_dir,
            observation_batch,
            action_history,
            goal_points, 
            visualize_top_k=False,
    ):
        start_start_time = time.time()

        os.makedirs(log_dir, exist_ok=True)
        
        # extract in-hand pc and table pcs 
        inhand_object_pc, table_object_pcs, inhand_index = self.extract_object_pcs(observation_batch[2][0, :, -1], return_index=True)    # the third axis is time. Use the current frame
        nozero_table_object_pcs, nonzero_mask = self.remove_zero_rows(table_object_pcs, return_mask=True)
        # goal_points = infer_bounding_box_from_pc(table_object_pcs)
        
        # start planning 
        best_action, best_future = None, None
        self._model_prediction_times = []
        for iteration_index in range(self.num_iters):
            start_time = time.time()

            curr_ee_xy = inhand_object_pc.mean(0)[:2].cpu().numpy()
            actions = self.sampler.sample_actions(self.num_samples, nozero_table_object_pcs, curr_ee_xy)

            # print(f"Sampling takes {time.time() - start_time}")
            start_time = time.time()

            # # add a new action to indicate the first ee movement to reach the init position
            # curr_ee_xy = observation_batch[2][0, 20:40, -1, :2].mean(0).detach().cpu().numpy()      # get the ee xy coordinate by taking the center of the rod
            # lead_action = starting_positions - curr_ee_xy[None, None, :]
            # actions = np.concatenate((lead_action, actions), axis=1)
            # actions = np.concatenate((actions, np.zeros((self.num_samples, actions.shape[1], 1))), axis=-1)  # add a zero action at the end

            # predict future
            pred_start_time = time.time()
            max_batch_size = 150
            all_preds = []
            with torch.no_grad():
                # split into max_batch_size chunks
                for i in range(0, self.num_samples, max_batch_size):
                    predictions = self.model.predict_step(observation_batch, action_history, actions[i:i+max_batch_size])
                    all_preds.append(predictions)

            # all_preds is a list of dictionaries containing tensors. Concatenate them along the 0 dimension by key
            predictions = {k: np.concatenate([x[k] for x in all_preds], axis=0) for k in all_preds[0].keys()}
            prediction_time = time.time() - pred_start_time
            self._model_prediction_times.append(prediction_time)
            print(
                f"Out of {len(self._model_prediction_times)} iterations, "
                f"median prediction time {np.median(self._model_prediction_times)}"
            )
            
            start_time = time.time()
            
            predicted_inhand = predictions['object_obs'][:, :, inhand_index * 20: (inhand_index + 1) * 20]
            # get the rest of the points, which don't contain the inhand object
            predicted_others = np.concatenate((predictions['object_obs'][:, :, :inhand_index * 20], predictions['object_obs'][:, :, (inhand_index + 1) * 20:]), axis=2)
            nonzero_predicted_others = predicted_others[:, :, nonzero_mask.cpu().numpy()]  # remove zero objects
            rewards = self.obj_fn(predicted_inhand, nonzero_predicted_others, goal_points, actions, last_state_only=False)[:, None, None]  # shape [num_samples, 1, 1]
            
            # verbose, for our own understanding
            # report best rewards for the two rows
            n_samples = rewards.shape[0]
            row_wise_rewards = rewards.split(n_samples // 2, dim=0)
            print(f'max rewards for each row (higher the better): {[x.max().item() for x in row_wise_rewards]}')
           # compute index of best reward for each row
            row_wise_best_indices = [np.argsort(-x.flatten().cpu().numpy())[:1] for x in row_wise_rewards]
            row_wise_best_indices[1] = row_wise_best_indices[1] + n_samples // 2
            row_wise_best_indices = np.array(row_wise_best_indices).flatten()

            top_k = max(visualize_top_k, 1)
            
            x = input("Intervene, Press 0 or 1. 0 is the left row from sitting in the computer chair, 1 is right row.\n")
            if x == '0':
                best_prediction_inds = row_wise_best_indices[:1]
            elif x == '1':
                best_prediction_inds = row_wise_best_indices[1:]
            else:
                print("no intervention, defaulting.. will visualize the best plan for each row")
                best_prediction_inds = np.argsort(-rewards.flatten().cpu().numpy())[:top_k]
            
            # print("best_prediction_inds", best_prediction_inds)
            # best_prediction_inds = np.argsort(-rewards.flatten())[[0, 19, 39]]
            
            # best_rewards = [rewards[i] for i in best_prediction_inds]
            best_actions = [actions[x] for x in best_prediction_inds]
            rowwise_best_actions = [actions[x] for x in row_wise_best_indices]
            # print("Row-wise best actions:", rowwise_best_actions)
            # print("best rewards:", best_rewards)
            # print('best actions:', best_actions)

            last_state_rewards = self.obj_fn(predicted_inhand, nonzero_predicted_others, goal_points, actions, last_state_only=True)[:, None, None]  # shape [num_samples, 1, 1]
            best_last_state_indices = np.argsort(-last_state_rewards.flatten())[:top_k]  # [[0, 19, 39]] 
            best_last_state_rewards = [last_state_rewards[i] for i in best_last_state_indices]
            best_rowwise_last_state_rewards = [last_state_rewards[i] for i in row_wise_best_indices]
            # print(f'last state rewards: {best_last_state_rewards}')
            # print(f'Computing rewards takes {time.time() - start_time}')
            if iteration_index == self.num_iters - 1:
                end_end_time = time.time()

            if t % self.log_every == 0 and iteration_index == self.num_iters - 1 and visualize_top_k > 0:
                log_folder = f"{log_dir}/ee_pos_step_{t}_iter{iteration_index}_plan"
                
                self.log_best_plans(
                    log_folder, predictions, goal_points, row_wise_best_indices, best_rowwise_last_state_rewards, self.sampler.num_actions, action_history, rowwise_best_actions
                )
                
                # self.log_best_plans(
                #     log_folder, predictions, goal_points, best_prediction_inds, best_last_state_rewards, self.sampler.num_actions, action_history, best_actions
                # )

            start_time = time.time()

            best_action = actions[best_prediction_inds[0]]
            best_future = {k: v[best_prediction_inds[0]] for k, v in predictions.items()}
            best_future = convert_state_dict_to_state_list(best_future)

        print(f'Total time for planning: {end_end_time - start_start_time}')

        return best_action, best_future[-best_action.shape[0]:]

    def log_best_plans(self, filename, predictions, goal, best_prediction_inds, best_rewards, horizon, action_history, actions):
        from planning.visualize import create_frames_for_pcs, save_moviepy
        predictions = np.concatenate((
            predictions['object_obs'][..., :3],
            # predictions['inhand'][..., :3],
            predictions['bubble'][..., :3],
        ), axis=2)
        total_T = predictions.shape[1]
        observed_his_len = total_T - horizon
        
        for i, prediction in enumerate(predictions[best_prediction_inds]):
            action = actions[i]
            
            def status_str_lambda(x):
                if x == 0:
                    return "History.."      # no action to show
                elif x < observed_his_len:
                    return f"History.. Action={[f'{num:.3f}' for num in action_history[0, -1, x-1]]}"        # show the action in history
                else:
                    return f"Planning... Action = {[f'{num:.3f}' for num in action[x-observed_his_len]]}"  # show the planned action
                
            # status_str_lambda = lambda x: f"{x}: History.. Action = {action_history[0, -1, ].tolist()} " if x < observed_his_len else f"{x}: Planning... Action = {action[x-observed_his_len]}"
            print("Putting logging job")
            # self.logging_thread.put({
            #     'prediction': prediction,
            #     'goal': goal,
            #     'filename': f"{filename}_{i}.mp4",
            #     'title': f"last-state reward = {best_rewards[i].item()}",
            #     'extra_title_func': True,
            #     'observed_his_len': observed_his_len,
            #     'action_history': action_history.cpu(),
            #     'action': action,
            # })
            vis_frames = create_frames_for_pcs(prediction, goal, multiview=True,
                                               view_size=5,
                                               title=f"last-state reward = {best_rewards[i].item()}",
                                               extra_title_func=status_str_lambda)
            vis_frames = [np.asarray(frame) for frame in vis_frames]
            save_moviepy(vis_frames, f"{filename}_{i}.mp4", fps=5)
            print("Logging to", filename)


from planning.visualize import create_frames_for_pcs, save_moviepy
import multiprocessing
import queue
class VisualizationLoggingThread(multiprocessing.Process):

    def __init__(self, parameters_queue):
        super(VisualizationLoggingThread, self).__init__()
        self.parameters_queue = parameters_queue

    def run(self):
        while True:
            try:
                parameters = self.parameters_queue.get(block=True)
                vis_frames = create_frames_for_pcs(parameters['prediction'], parameters['goal'], multiview=True,
                                                   view_size=5, title=parameters['title'],
                                                   observed_his_len=parameters['observed_his_len'],
                                                   extra_title_func=parameters['extra_title_func'],
                                                   action_history=parameters['action_history'],
                                                   action=parameters['action'])
                vis_frames = [np.asarray(frame) for frame in vis_frames]
                save_moviepy(vis_frames, parameters['filename'], fps=5)
                print("Logging to", parameters['filename'])
            except queue.Empty:
                # print("Exception in logging thread:", e)
                time.sleep(0.5)
                pass

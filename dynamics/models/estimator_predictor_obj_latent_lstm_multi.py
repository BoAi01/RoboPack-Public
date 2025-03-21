"""
Implementation of our method for the box pushing task.
The implementation implicitly assumes there are multiple objects on the table.
In principle, this should work for box pushing as well, but there are some slight differences.
"""


import pdb
from collections import OrderedDict
import time
import random
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
import torch_geometric as pyg

# from pytorch_memlab import profile, profile_every, MemReporter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dynamics.dataset import connect_edges, connect_edges_batched, compute_slice_indices
from dynamics.models.estimator_predictor_dpi_net_obj_latent_lstm import EstimatorDPINet, PredictorDPINet
from dynamics.loss import PositionLoss, Chamfer, EMDCPU, MSE
from dynamics.models.autoencoder import AutoEncoder

from utils_general import AverageMeter


class DynamicsPredictor(pl.LightningModule):
    def __init__(self, config, stats):
        super().__init__()

        # Extract configuration parameters
        self.config = config
        self.points_per_obj = config["n_points"]
        assert (torch.tensor(self.points_per_obj) == self.points_per_obj[0]).all(), f"currently we only support objects having the same particles but got {self.points_per_obj}"
        self.cumul_points = np.cumsum([0] + self.points_per_obj)
        self.type_feat_len = 1
        self.vis_feat_len = config["feature_dim_vision"]
        self.act_feat_len = config["feature_dim_action"]
        self.pos_len = 3  # vector length of position (xyz)
        self.his_len = config["history_length"]
        assert self.his_len == 1, "history length must be 1 for this model."
        
        self.seq_len = config["sequence_length"]
        # self.tac_feat_len = config["tactile_feat_dim"]
        self.tac_raw_dim = config["tactile_raw_dim"]
        self.tac_feat_dim = config["ae_enc_dim"]
        self.n_object_points = sum(config["n_points"])  # number of points except tool
        self.num_objects = len(config["n_points"])
        self.lr = config["optimizer"]["lr"]
        self.num_bubbles = 2
        self.zero_tactile = config["zero_tactile"]
        self.tactile_use_gt = config["tactile_use_gt"]
        self.pred_physics_params = config["pred_physics_params"]
        self.rigid_only = self.config["rigid_only"]
        
        if self.zero_tactile is True:
            assert not self.tactile_use_gt
        if self.zero_tactile:
            print(f'tactile will be zeroed. ')
        self.visual_blind = config["visual_blind"]
        
        self.recurr_T = config["recurr_T"]
        self.obj_phy_feat_len = config["dpi_net"]["obj_phy_feat_len"]
        self.teacher_forcing_thres = config["teacher_forcing_thres"]

        # tactile-related
        self.autoencoder = AutoEncoder.load_from_checkpoint(input_dim=config["tactile_raw_dim"],
                                                            encoding_dim=config["ae_enc_dim"],
                                                            config=config, stats=stats,
                                                            frozen=True,
                                                            checkpoint_path=ae_checkpoint).to(config.device)
        
        print(f'Autoencoder loaded from {ae_checkpoint}')

        # Initialize the GNN layer
        self.estimator = EstimatorDPINet(config)
        self.predictor = PredictorDPINet(config)

        # Training time: initialize loss functions
        self.position_loss = PositionLoss(
            config["loss_type"],
            config["chamfer_emd_weights"],
            config["loss_by_n_points"],
            self.points_per_obj,
            config["object_weights"], 
        )
        self.pos_loss_weight = config["pos_loss_weight"]

        # Test time: initialize different loss types
        self.chamfer_loss = Chamfer()
        self.emd_loss = EMDCPU()
        self.mse_loss = MSE()
        self.tactile_loss = nn.MSELoss()

        # Test time: initialize placeholders for predicted and ground truth state sequences for visualization
        self.loss_dicts = []
        self.pred_state_seqs = []
        self.gt_tac_seqs = []

        # Save hyperparameters
        self.save_hyperparameters()
        
        # global history
        self.pred_physics = None
        self.total_tac_loss = AverageMeter()
        self.total_emd_loss = AverageMeter()
        self.total_chamfer_loss = AverageMeter()
        self.total_mse_loss = AverageMeter()
        self.pred_physics_list = {}
        self.lstm_states = {}

        # test-time logger
        self.error_seq = []
    
    def update_rod_points(self, prev_pos, predicted_pos, pointwise_action):
        # predicted_pos = predicted_pos.clone()
        # rod_start_idx, rod_end_idx = self.cumul_points[-2], self.cumul_points[-1]   # the last object is the rod
        # predicted_pos[:, rod_start_idx: rod_end_idx] = prev_pos[:, rod_start_idx: rod_end_idx] + \
        #                                            pointwise_action[:, rod_start_idx: rod_end_idx]
        
        num_point_per_obj = self.points_per_obj[0]
        
        for i, objects_pc in enumerate(predicted_pos):
            # find out which is the in-hand object
            # assuming it is the object with the max z value
            objects_pc = objects_pc.clone()   # avoid in-place operations
            object_pcs = objects_pc.split(num_point_per_obj, dim=0)
            max_zs = [pc.mean(0)[2] for pc in object_pcs] 
            max_z_index = max_zs.index(max(max_zs))
            
            # translate the object by action
            predicted_pos[i, max_z_index * num_point_per_obj : (max_z_index + 1) * num_point_per_obj] += pointwise_action[i, -num_point_per_obj:]
            
        return predicted_pos
    
    def compute_losses_dict(self, pred_pos, gt_pos, train):
        loss = {}
        if train:
            position_loss, position_losses = self.position_loss(pred_pos, gt_pos, return_losses=True)
            loss["train_pos"] = position_loss * self.pos_loss_weight
            loss["train_pos_losses"] = position_losses

        else:
            loss["chamfer"] = self.chamfer_loss(pred_pos, gt_pos)
            loss["emd"] = self.emd_loss(pred_pos, gt_pos)
            loss["mse"] = self.mse_loss(pred_pos, gt_pos)
        return loss
    
    def has_contact(self, points, pointwise_action, distance_threashold=0.05):
        post_action_points = points + pointwise_action
        tool_points = post_action_points[:, self.cumul_points[-2]:]     # tool and rod, after action
        object_points = points[:, :self.cumul_points[-2]]   # objects, before action
        tool_to_object_distances, _ = Chamfer.compute(tool_points, object_points, keep_dim=True)
        tool_to_object_distances_min = tool_to_object_distances.min(1).values
        # note: if we split the box points into two, the min distance would be ~0.03
        # thus the threshold should take into account the effect of discretization 
        return tool_to_object_distances_min < distance_threashold
    
    def is_tool_moving_close(self, points, pointwise_action, difference_threshold=0.002):
        pre_action_points = points.clone()
        post_action_points = points + pointwise_action
        
        # compute pre-action chamfer dist
        pre_action_tool_points = pre_action_points[:, self.cumul_points[-2]:]     # tool and rod, before action
        pre_action_object_points = pre_action_points[:, :self.cumul_points[-2]]   # objects, before action
        pre_action_tool_to_object_dist, _ = Chamfer.compute(pre_action_tool_points, pre_action_object_points, keep_dim=True)
        pre_action_tool_to_object_dist_min = pre_action_tool_to_object_dist.min(1).values
        
        # compute post-action chamfer dist
        post_action_tool_points = post_action_points[:, self.cumul_points[-2]:]     # tool and rod, before action
        post_action_object_points = post_action_points[:, :self.cumul_points[-2]]   # objects, before action
        post_action_tool_to_object_dist, _ = Chamfer.compute(post_action_tool_points, post_action_object_points, keep_dim=True) 
        post_action_tool_to_object_dist_min = post_action_tool_to_object_dist.min(1).values
        
        return post_action_tool_to_object_dist_min - pre_action_tool_to_object_dist_min < difference_threshold      # allow some error margin
    
    def is_action_valid(self, pointwise_action, min_threashold=1e-4, max_threashold=0.05):
        tool_actions = pointwise_action[:, self.cumul_points[-1]:]
        large_enough = (torch.abs(tool_actions) > min_threashold).any(1).any(1)   # any of the action is large
        small_enough = (torch.abs(tool_actions) < max_threashold).all(dim=1).all(dim=1)        # all should be smaller than bound
        return large_enough & small_enough
    
    def is_physical_interaction_possible(self, points, pointwise_action):
        # return self.has_contact(points, pointwise_action) \
        #        & self.is_tool_moving_close(points, pointwise_action) \
        #        & self.is_action_valid(pointwise_action)
        return self.is_action_valid(pointwise_action)
               
    def init_physics_params(self, shape):
        """
        Return inital values for physics parameters, given the shape of the array in tuple
        """        
        # physics_params = torch.zeros(shape).float().to(self.device)
        physics_params = torch.normal(0, 0.4, shape).to(self.device)
        physics_params = physics_params.clip(-1, 1)
        return torch.zeros_like(physics_params)

    def estimator_forward(self, data, t, tac_feat_bubbles, pred_pos_prev, pred_physics, train):
        B = data.num_graphs
        N = data.num_nodes // B

        # Step 1: Extract and process pos history, from t to (t + self.his_len - 1)
        node_pos = data.pos.view(B, N, -1, self.pos_len)
        T = node_pos.shape[2]
        node_pos_hist = node_pos[:, :, t: (t + self.his_len)].view(B, N, -1)
        gt_pos = node_pos[
                 :,
                 : self.cumul_points[-1],  # take all the points specified
                 (t + self.his_len): (t + self.his_len + 1)
                 ].squeeze(2)  # ground truth is the pos at the next time step

        # # Step 1.1: Get tactile data
        if self.zero_tactile:
            tac_feat_bubbles = torch.zeros_like(tac_feat_bubbles)
        tactile_feat_objects = torch.zeros(B, tac_feat_bubbles.shape[1],
                                           self.n_object_points,
                                           self.config["ae_enc_dim"]).to(self.device)
        tactile_feat_all_points = torch.cat([tactile_feat_objects, tac_feat_bubbles], dim=2)

        # debugging
        # for i in range(tactile_feat_all_points.shape[1]):
        #     tactile_feat_all_points[:, i] = torch.zeros_like(tactile_feat_all_points[:, i]) + i

        # get the tactile feature history, and the ground truth tactile feature, which
        # is the next reading after the history
        tactile_feat_hist = tactile_feat_all_points[:, t:(t + self.his_len + 1)]  # include current tactile
        tactile_feat_gt = tactile_feat_all_points[:, (t + self.his_len):(t + self.his_len + 1)]
        # shape (B, his_len, N, tac_feat_dim)

        # reshape the tactile feature so that the feats across time steps are stacked along the last dim
        # WARNING: You cannot directly do .view(B, N, -1) because the two dimensions
        # are not adjacent (space not contiguous) and this would not concate the two dimensions along the last dim
        # Instead, do permute and reshape
        tactile_feat_hist = tactile_feat_hist.permute(0, 2, 1, 3).reshape(B, N, -1)
        tactile_feat_gt = tactile_feat_gt.permute(0, 2, 1, 3).reshape(B, N, -1)
        # shape (B, N, his_len * tac_feat_dim)

        # Step 2: Extract and process node feature history
        # for the format, see the dataset class
        # feature is a concat of (particle type, visual features (rgb), and actions)
        node_feature = data.x.view(B, N, -1)
        # assert node_feature.shape[-1] == self.type_feat_len + self.vis_feat_len +
        # (self.his_len + self.seq_len - 1) * self.pos_len, "you need to rebuild dataset"

        action_hist = node_feature[:, :, self.type_feat_len + self.vis_feat_len:].view(B, N, T - 1, -1)
        action_hist = action_hist[:, :, t: (t + self.his_len)].reshape(B, N, -1)

        # node features, concat of type (B, N, 1), visual feature (B, N, 3), action feat (B, N, 3), and tac (B, N, 6)
        node_feature_hist = torch.cat(
            (action_hist, tactile_feat_hist), dim=-1
        )

        # Process input data depending on if it's the first action or not
        if t == 0:
            # Last history state is the previous state
            pos_prev = node_pos_hist[:, : self.cumul_points[-1], -self.pos_len:]

            # Update pos and x in data
            window = data.clone("edge_index", "edge_attr", "y", "batch", "ptr")
            window.pos = node_pos_hist.view(-1, node_pos_hist.shape[-1])
            window.x = node_feature_hist.view(-1, node_feature_hist.shape[-1])
            if pred_physics is None:
                # window.obj_phy_feat = torch.zeros(B, len(self.cumul_points) - 1, self.obj_phy_feat_len).to(self.device)
                window.obj_phy_feat = self.init_physics_params((B, len(self.cumul_points) - 1, self.obj_phy_feat_len))
            else:
                if not train: 
                    # this might be possible at test time 
                    window.obj_phy_feat = pred_physics
                else: 
                    raise AssertionError("pred_physics should be None for now")
        else:
            # Previous predicted state is the previous state
            pos_prev = pred_pos_prev
            pos_prev_tool = node_pos_hist[:, self.cumul_points[-1]:, -self.pos_len:]
            # pred_physics = pred_physics.view(B, len(self.points_per_obj), -1)

            window = data.clone("edge_index", "edge_attr", "y", "batch", "ptr")
            
            if pred_physics is None:
                # pred_physics = torch.zeros(B, len(self.cumul_points) - 1, self.obj_phy_feat_len).to(self.device)
                pred_physics = self.init_physics_params((B, len(self.cumul_points) - 1, self.obj_phy_feat_len))
                
            window.obj_phy_feat = pred_physics
            edge_index, edge_attr = connect_edges_batched(
                self.config,
                pos_prev,
                pos_prev_tool,
                N,
                data.num_graphs,
                self.cumul_points,
            )

            # sort the edge_index by the first row so that we can unbatch them if desired
            # this is optional, but you need it for visualization to compute edge_slice_indices
            edge_index_indices = torch.argsort(edge_index[0])
            edge_index = edge_index[:, edge_index_indices]
            edge_attr = edge_attr[edge_index_indices]

            window.edge_index = edge_index
            window.edge_attr = edge_attr
            window_pos = torch.cat(
                (node_pos_hist[:, :, : (self.his_len - 1) * self.pos_len], torch.cat((pos_prev, pos_prev_tool), dim=1)),
                dim=-1,
            )
            window.pos = window_pos.view(-1, window_pos.shape[-1])
            window.x = node_feature_hist.view(-1, node_feature_hist.shape[-1])

            # compute the first index in edge_index that corresponds to the first edge of each graph
            # this can be determined by finding the first index greater than or equal to N, then 2N, then 3N, etc.
            # this is because the edge_index is sorted by the first row
            edge_slice_indices = compute_slice_indices(window.edge_index[0], N, data.num_graphs)
            window._slice_dict['edge_index'] = edge_slice_indices
            window._slice_dict['edge_attr'] = edge_slice_indices

        # Use DPI network to predict non-rigid and rigid motion norms
        rigid_norm, non_rigid_norm, pred_physics = self.estimator(window)

        # If this model is not supposed to predict physics params, just zero it out 
        if not self.pred_physics_params:
            pred_physics = torch.zeros_like(pred_physics)

        # Denormalize the motion norms and compute the predicted state
        non_rigid = non_rigid_norm * 1

        # Process predictions depending on if we want rigid transformation or not
        rigid = rigid_norm * 1
        pred_pos_rigid = pos_prev + rigid
        pred_pos = (pred_pos_rigid + non_rigid) if not self.config["rigid_only"] else pred_pos_rigid
        
        # # manual update the rod points
        # pred_pos = self.update_rod_points(node_pos_hist, pred_pos, action_hist)
        
        # Calculate loss
        loss = self.compute_losses_dict(pred_pos, gt_pos, train)

        return loss, pred_pos, pred_physics, (gt_pos, tactile_feat_gt)

    def predictor_forward(self, data, t, pred_pos_prev, pred_physics, train):
        B = data.num_graphs
        N = data.num_nodes // B

        # Step 1: Extract and process pos history, from t to (t + self.his_len - 1)
        node_pos = data.pos.view(B, N, -1, self.pos_len)
        T = node_pos.shape[2]
        node_pos_hist = node_pos[:, :, t: (t + self.his_len)].view(B, N, -1)
        gt_pos = node_pos[
                 :,
                 : self.cumul_points[-1],  # take all the points specified
                 (t + self.his_len): (t + self.his_len + 1)
                 ].squeeze(2)  # ground truth is the pos at the next time step

        # Step 2: Extract and process node feature history
        # for the format, see the dataset class
        # feature is a concat of (particle type, visual features (rgb), and actions)
        node_feature = data.x.view(B, N, -1)
        # assert node_feature.shape[-1] == self.type_feat_len + self.vis_feat_len +
        # (self.his_len + self.seq_len - 1) * self.pos_len, "you need to rebuild dataset"

        action_hist = node_feature[:, :, self.type_feat_len + self.vis_feat_len:].view(B, N, T - 1, -1)
        action_hist = action_hist[:, :, t: (t + self.his_len)].reshape(B, N, -1)

        # node features, concat of type (B, N, 1), visual feature (B, N, 3), action feat (B, N, 3), and tac (B, N, 6)
        node_feature_hist = action_hist

        # Process input data depending on if it's the first action or not
        if t == 0:
            # Last history state is the previous state
            pos_prev = node_pos_hist[:, : self.cumul_points[-1], -self.pos_len:]
            
            # Update pos and x in data
            window = data.clone("edge_index", "edge_attr", "y", "batch", "ptr")
            window.pos = node_pos_hist.view(-1, node_pos_hist.shape[-1])
            window.x = node_feature_hist.view(-1, node_feature_hist.shape[-1])
            
            if pred_physics is None:
                # window.obj_phy_feat = torch.zeros(B, len(self.cumul_points) - 1, self.obj_phy_feat_len).to(self.device)
                window.obj_phy_feat = self.init_physics_params((B, len(self.cumul_points) - 1, self.obj_phy_feat_len))
            else:
                if not train:
                    # this might be possible at test time
                    window.obj_phy_feat = pred_physics
                else:
                    raise AssertionError("pred_physics should be None for now")
                
        else:
            # Previous predicted state is the previous state
            pos_prev = pred_pos_prev
            pos_prev_tool = node_pos_hist[:, self.cumul_points[-1]:, -self.pos_len:]
            # pred_physics = pred_physics.view(B, len(self.points_per_obj), -1)
            
            # in case pred_physics is not available, e.g., state estimator not called
            if pred_physics is None:
                # pred_physics = torch.zeros(B, len(self.cumul_points) - 1, self.obj_phy_feat_len).to(self.device)
                pred_physics = self.init_physics_params((B, len(self.cumul_points) - 1, self.obj_phy_feat_len))

            window = data.clone("edge_index", "edge_attr", "y", "batch", "ptr")

            window.obj_phy_feat = pred_physics
            edge_index, edge_attr = connect_edges_batched(
                self.config,
                pos_prev,
                pos_prev_tool,
                N,
                data.num_graphs,
                self.cumul_points,
            )

            # sort the edge_index by the first row so that we can unbatch them if desired
            # this is optional, but you need it for visualization to compute edge_slice_indices
            edge_index_indices = torch.argsort(edge_index[0])
            edge_index = edge_index[:, edge_index_indices]
            edge_attr = edge_attr[edge_index_indices]

            window.edge_index = edge_index
            window.edge_attr = edge_attr
            window_pos = torch.cat(
                (node_pos_hist[:, :, : (self.his_len - 1) * self.pos_len], torch.cat((pos_prev, pos_prev_tool), dim=1)),
                dim=-1,
            )
            window.pos = window_pos.view(-1, window_pos.shape[-1])
            window.x = node_feature_hist.view(-1, node_feature_hist.shape[-1])

            # compute the first index in edge_index that corresponds to the first edge of each graph
            # this can be determined by finding the first index greater than or equal to N, then 2N, then 3N, etc.
            # this is because the edge_index is sorted by the first row
            edge_slice_indices = compute_slice_indices(window.edge_index[0], N, data.num_graphs)
            window._slice_dict['edge_index'] = edge_slice_indices
            window._slice_dict['edge_attr'] = edge_slice_indices

        # Use DPI network to predict non-rigid and rigid motion norms
        rigid_norm, non_rigid_norm = self.predictor(window)

        # Denormalize the motion norms and compute the predicted state
        non_rigid = non_rigid_norm * 1  # self.pos_scale

        # Process predictions depending on if we want rigid transformation or not
        rigid = rigid_norm * 1  # self.pos_scale
        pred_pos_rigid = pos_prev + rigid
        pred_pos = (pred_pos_rigid + non_rigid) if not self.config["rigid_only"] else pred_pos_rigid

        # # manual update the rod points
        # pred_pos = self.update_rod_points(node_pos_hist, pred_pos, action_hist)

        # Calculate loss
        loss = self.compute_losses_dict(pred_pos, gt_pos, train)

        return loss, pred_pos, (gt_pos, None)

    def training_step(self, batch, batch_idx):
        B = batch.num_graphs
        object_cls = batch.object_cls.reshape(B, -1)[:, 0]

        train_pos_loss, tac_loss = 0, 0
        pred_pos, pred_tactile, pred_physics = None, None, None
        
        tac_feat_bubbles = self.autoencoder.encode_structured(batch)
        self.estimator.reset_lstm_state()
        physics_params_t = [] 
        box_losses = torch.zeros(1, self.num_objects)

        # Calculate total loss over the sequence
        updated_recurr_T = random.randint(0, self.seq_len - 1)
        for i in range(self.seq_len):
            
            # before updated_recurr_T, we execute state estimation
            # after that, we execute dynamics prediction
            if i < updated_recurr_T:
                loss, pred_pos, pred_physics, gts = self.estimator_forward(batch, i, tac_feat_bubbles,
                                                                           pred_pos, pred_physics, True)
                gt_pos, gt_tactile = gts
                physics_params_t.append(pred_physics)
                box_losses += torch.tensor(loss["train_pos_losses"])
                train_pos_loss += loss["train_pos"]
            else:
                # shuffle physics params of objects of the same type
                if updated_recurr_T > 0:
                    assert pred_physics is not None, f'pred_physics is should not be None when updated_recurr_T = {updated_recurr_T} > 0'

                loss, pred_pos, gts = self.predictor_forward(batch, i, pred_pos, pred_physics, True)
                gt_pos, gt_tactile = gts
                # physics_params_t.append(pred_physics)
                box_losses += torch.tensor(loss["train_pos_losses"])
                train_pos_loss += loss["train_pos"]

            if loss["train_pos"] > self.teacher_forcing_thres:
                pred_pos = gt_pos

            if not self.visual_blind:
                pred_pos = gt_pos

        # normalize the loss by the sequence length
        train_pos_loss = train_pos_loss / self.seq_len / self.num_objects
        # tac_loss /= self.seq_len
        box_losses /= self.seq_len

        # from torchviz import make_dot
        # make_dot(pred_pos, params=dict(self.named_parameters())).render(
        #     "computation_graph", format="png"
        # )
        # pdb.set_trace()

        # memory_alloc = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()

        for i, param in enumerate(physics_params_t):
            self.log(
                f"train_physics_params_{i}",
                param.view(-1, self.obj_phy_feat_len).std(0).mean(0).item(), 
                prog_bar=False,
                on_epoch=True,
                on_step=True,
                batch_size=batch.num_graphs,
            )
        
        for i, loss in enumerate(box_losses.tolist()):
            self.log(
                f"train_obj_loss_{i}",
                loss[i],
                prog_bar=False,
                on_epoch=True,
                on_step=False,
                batch_size=batch.num_graphs,
            )

        self.log(
            "train_loss",
            train_pos_loss,
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            batch_size=batch.num_graphs,
        )

        self.log(
            "learning_rate",
            self.optimizer.param_groups[0]['lr'],
            prog_bar=False,
            on_epoch=True,
            on_step=False,
            batch_size=batch.num_graphs
        )
        
        # breakpoint()
        # loss.backward()
        # [param.grad.max() if param is not None else None for name, param in self.named_parameters()]

        return train_pos_loss   # IMPORTANT: need to return all trainable losses

    def validation_step(self, batch, batch_idx):
        train_pos_loss, tac_loss = 0, 0
        pred_pos, pred_tactile, pred_physics = None, None, None
        
        tac_feat_bubbles = self.autoencoder.encode_structured(batch)
        self.estimator.reset_lstm_state()
        physics_params_t = [] 
        box_losses = torch.zeros(1, self.num_objects)

        # Calculate total loss over the sequence
        updated_recurr_T = self.recurr_T
        for i in range(self.seq_len):
            
            # before updated_recurr_T, we execute state estimation
            # after that, we execute dynamics prediction
            if i < updated_recurr_T:
                loss, pred_pos, pred_physics, gts = self.estimator_forward(batch, i, tac_feat_bubbles,
                                                                           pred_pos, pred_physics, True)
                gt_pos, gt_tactile = gts
                physics_params_t.append(pred_physics)
                box_losses += torch.tensor(loss["train_pos_losses"])

                train_pos_loss += loss["train_pos"]

            else:
                loss, pred_pos, gts = self.predictor_forward(batch, i, pred_pos, pred_physics, True)
                gt_pos, gt_tactile = gts
                # physics_params_t.append(pred_physics)
                box_losses += torch.tensor(loss["train_pos_losses"])

                train_pos_loss += loss["train_pos"]

            if loss["train_pos"] > self.teacher_forcing_thres:
                pred_pos = gt_pos

            if not self.visual_blind:
                pred_pos = gt_pos

        # normalize the loss by the sequence length
        train_pos_loss = train_pos_loss / self.seq_len / self.num_objects
        # tac_loss /= self.seq_len
        box_losses /= self.seq_len

        for i, param in enumerate(physics_params_t):
            self.log(
                f"val_physics_params_{i}",
                param.view(-1, self.obj_phy_feat_len).std(0).mean(0).item(),
                prog_bar=False,
                on_epoch=True,
                on_step=False,
                batch_size=batch.num_graphs,
            )
        
        for i, loss in enumerate(box_losses.tolist()):
            self.log(
                f"val_obj_loss_{i}",
                loss[i],
                prog_bar=False,
                on_epoch=True,
                on_step=False,
                batch_size=batch.num_graphs,
            )

        self.log(
            "val_loss",  # instead of val_pos_loss in order to be comparable to prev results
            train_pos_loss,
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            batch_size=batch.num_graphs,
        )

    def get_particle_tactile(self, data):
        B = data.num_graphs
        N = data.num_nodes // B

        forces, flows, pressure = data.forces, data.flows, torch.zeros(32, 2) #  data.pressure
        # torch.Size([24, 2, 7]), torch.Size([24, 40, 2]), torch.Size([24, 2])
        n_bubble_points = (N - self.n_object_points) // 2

        # -1 means seq length, e.g., (his_len + seq_len) during training
        # but at test time the value might differ
        forces = forces.view(B, -1, self.num_bubbles, 7)

        flows = flows.view(B, -1, self.num_bubbles, n_bubble_points, 2)
        # flows = (flows - self.flow_mean) / self.flow_scale
        # flows = flows.clip(-2, 2) / 2

        pressure = pressure.view(B, -1, self.num_bubbles)
        # pressure = (pressure - self.pressure_mean) / self.pressure_scale
        # pressure = pressure.clip(-2, 2) / 2

        # Step 1.2: Some processing to unify the shape
        forces = forces.unsqueeze(3).repeat(1, 1, 1, n_bubble_points, 1)
        forces = torch.cat([forces[..., :2], forces[..., -1:]], dim=-1)  # get the three nonzero values
        # forces = (forces - self.force_mean) / self.force_scale
        # forces = forces.clip(-2, 2) / 2

        pressure = pressure.unsqueeze(3).unsqueeze(-1).repeat(1, 1, 1, n_bubble_points, 1)

        return forces, flows, pressure
    
    def denorm_tactile_reading(self, normed_reading, mean, scale, coefficient=2):
        return normed_reading * coefficient * scale + mean

    def test_step(self, batch, batch_idx):
        B = batch.num_graphs
        N = batch.num_nodes // B
        S = batch.pos.shape[-1] // self.pos_len
        n_bubble_points = (N - self.n_object_points) // 2

        # Get the ground truth state
        gt_pos_seq = batch.pos.view(B, N, S, self.pos_len).transpose(1, 2)

        # Get the state history for each object in the batch
        pos_seq_object = [gt_pos_seq[:, : self.his_len, : self.cumul_points[-1]]]

        # set up variables for tactile prediction
        gt_forces, gt_flows, gt_pressure = self.get_particle_tactile(batch)
        gt_forces, gt_flows, gt_pressure = [x.cpu().numpy() for x in [gt_forces, gt_flows, gt_pressure]]
        # every of shape (B, T, 2, n_points_per_bubble, x)
        
        # initialize with a ground truth for the first step
        tac_pred_seqs = [
            dict(
                forces=gt_forces[:, 0],
                flows=gt_flows[:, 0],
                pressure=gt_pressure[:, 0]
            )
        ]
        
        gt_tac_seq = [
            dict(
                forces=gt_forces[:, 0],
                flows=gt_flows[:, 0],
                pressure=gt_pressure[:, 0]
            )
        ]

        # Log initial zero losses for the first H steps
        for i in range(self.his_len):
            self.log_dict(
                {
                    f"chamfer_{i}": 0.0,
                    f"emd_{i}": 0.0,
                    f"mse_{i}": 0.0,
                    f"tactile_mse_{i}": 0
                },
                on_epoch=False,
                on_step=True,
                batch_size=B,
            )

        pred_pos, pred_tactile, self.pred_physics = None, None, None

        tac_feat_bubbles = self.autoencoder.encode_structured(batch)
        
        # For each state in the sequence (S), calculate the predicted state and losses
        self.estimator.reset_lstm_state()
        self.lstm_states[batch_idx] = []
        
        history_reset_idx = -1
        
        # Calculate total loss over the sequence
        updated_recurr_T = self.recurr_T
        for i in range(S - self.his_len):
            
            # before updated_recurr_T, we execute state estimation
            # after that, we execute dynamics prediction
            if i < updated_recurr_T:
                loss, pred_pos, pred_physics, gts = self.estimator_forward(batch, i, tac_feat_bubbles,
                                                                           pred_pos, self.pred_physics, False)
                gt_pos, gt_tactile = gts
                # box_losses += torch.tensor(loss["train_pos_losses"])
                self.pred_physics = pred_physics
                
            else:
                loss, pred_pos, gts = self.predictor_forward(batch, i, pred_pos, self.pred_physics, False)
                gt_pos, gt_tactile = gts
                # box_losses += torch.tensor(loss["train_pos_losses"])

            if not self.visual_blind:
                pred_pos = gt_pos
                
            if i == S - self.his_len - 1:
                # self.total_tac_loss.update(loss["tac"].item())
                self.total_emd_loss.update(loss["emd"].item())
                self.total_chamfer_loss.update(loss["chamfer"].item())
            
            # the hidden state might be none since the estimator is not called when recurr=0
            if self.estimator.lstm_hn_cn is not None and self.recurr_T > 0:
                self.lstm_states[batch_idx].append(list(x.cpu() for x in self.estimator.lstm_hn_cn) + [batch.object_cls[0].item()])

                if i not in self.pred_physics_list:
                    self.pred_physics_list[i] = []
                else:
                    self.pred_physics_list[i].append((pred_physics.cpu(), batch.object_cls[0].item()))

            # obtain the gt
            if i < updated_recurr_T:
                gt_tac_seq.append(
                    dict(
                        forces=gt_forces[:, i + 1],
                        flows=gt_flows[:, i + 1],
                        pressure=gt_pressure[:, 0]
                    )
                )
            
            # Log the losses
            self.log_dict(
                {f"{k}_{i + self.his_len}": v for k, v in loss.items()},
                on_epoch=False,
                on_step=True,
                batch_size=B,
            )

            # Append the predicted state to the object state sequence
            pos_seq_object.append(pred_pos[:, None])
            
        print(f'count: {self.total_tac_loss.count}, tac loss avg: {self.total_tac_loss.avg}, '
              f'emd loss avg: {self.total_emd_loss.avg}, chamfer loss avg: {self.total_chamfer_loss.avg}')
        
        # Get the pos sequence
        pos_seq_object = torch.cat(pos_seq_object, dim=1)

        # Get the vision feature sequence
        vision_feature_object = batch.x[..., self.type_feat_len: self.type_feat_len + self.vis_feat_len].view(
            B, 1, N, self.vis_feat_len
        )[:, :, : self.cumul_points[-1]]

        state_seq_object = torch.cat(
            (
                pos_seq_object,
                vision_feature_object.repeat(1, S, 1, 1),
            ),
            dim=-1,
        )

        pos_seq_tool = gt_pos_seq[:, :, self.cumul_points[-1]:]

        red = torch.tensor([1.0, 0.0, 0.0], device=gt_pos_seq.device)
        state_seq_tool = torch.cat(
            (
                pos_seq_tool,
                red.repeat(B, S, N - self.cumul_points[-1], 1),
            ),
            dim=-1,
        )
        pred_state_seq = [
            dict(
                object_obs=state_seq_object[0, i, : self.cumul_points[-2]].detach().cpu().numpy(),
                inhand=state_seq_object[0, i, self.cumul_points[-2]:].detach().cpu().numpy(),
                bubble=state_seq_tool[0, i].detach().cpu().numpy(),
            )
            for i in range(S)
        ]

        self.pred_state_seqs.append(pred_state_seq)
        self.gt_tac_seqs.append(gt_tac_seq)

    def state_estimator_run_forward(self, action, particles, tac_state, pred_physics):
        B = action.shape[0]
        N = particles.shape[1]
        
        node_pos_hist = particles

        # Step 1.1: Get tactile data
        # tac_feat_bubbles = torch.zeros_like(tac_feat_bubbles)
        if self.zero_tactile:
            tac_state = torch.zeros_like(tac_state)
        
        node_feature_hist = torch.cat(
            (action, tac_state), dim=-1
        )
        # shape (B, N, (his_len + seq_len) * (sum of all feat dim))
        
        if pred_physics is None:
            # pred_physics = torch.zeros(B, len(self.cumul_points) - 1, self.obj_phy_feat_len).to(self.device)
            pred_physics = self.init_physics_params((B, len(self.cumul_points) - 1, self.obj_phy_feat_len))

        # Previous predicted state is the previous state
        particles = particles[:, :self.cumul_points[-1]]
        pos_prev_tool = node_pos_hist[:, self.cumul_points[-1]:, -self.pos_len:]

        # window = data.clone("edge_index", "edge_attr", "y", "batch", "ptr")
        window = pyg.data.Batch(
            x=None,
            edge_index=None,
            edge_attr=None,
            # y=target_state,
            pos=None,        # ground truth for loss computation
            forces=None,
            flows=None,
            # pressure=pressure,
            object_cls=None,
            # rand_index = torch.cat([torch.from_numpy(d['rand_index']) for d in state_list], dim=0)
        )
        
        window.obj_phy_feat = pred_physics
        start_time = time.time()
        edge_index, edge_attr = connect_edges_batched(
            self.config,
            particles,
            pos_prev_tool,
            N,
            B, # data.num_graphs,
            self.cumul_points,
        )
        # print(f'\tConstructing graph takes {time.time() - start_time} seconds')

        # sort the edge_index by the first row so that we can unbatch them if desired
        # this is optional, but you need it for visualization to compute edge_slice_indices
        edge_index_indices = torch.argsort(edge_index[0])
        edge_index = edge_index[:, edge_index_indices]
        edge_attr = edge_attr[edge_index_indices]

        window.edge_index = edge_index
        window.edge_attr = edge_attr
        window_pos = torch.cat(
            (node_pos_hist[:, :, : (self.his_len - 1) * self.pos_len], torch.cat((particles, pos_prev_tool), dim=1)),
            dim=-1,
        )
        window.pos = window_pos.view(-1, window_pos.shape[-1])
        window.x = node_feature_hist.view(-1, node_feature_hist.shape[-1])

        # compute the first index in edge_index that corresponds to the first edge of each graph
        # this can be determined by finding the first index greater than or equal to N, then 2N, then 3N, etc.
        # this is because the edge_index is sorted by the first row
        edge_slice_indices = compute_slice_indices(window.edge_index[0], N, B)

        # manually create a dict atttribute. this is a hack 
        window._slice_dict = {}
        window._slice_dict['edge_index'] = edge_slice_indices
        window._slice_dict['edge_attr'] = edge_slice_indices
        window._num_graphs = B

        start_time = time.time()
        
        # Use DPI network to predict non-rigid and rigid motion norms
        rigid_norm, non_rigid_norm, pred_physics = self.estimator(window)
        
        # If this model is not supposed to predict physics params, just zero it out
        # print(f'\tDPI takes {time.time() - start_time} seconds: {window}')

        if not self.pred_physics_params:
            pred_physics = torch.zeros_like(pred_physics)

        # Denormalize the motion norms and compute the predicted state
        non_rigid = non_rigid_norm * 1  # self.pos_scale
        rigid = rigid_norm  # * self.pos_scale

        # Process predictions if physical interaction is possible by pre-specified criteria
        is_zero_motion = ~self.is_physical_interaction_possible(node_pos_hist, action)
        non_rigid[is_zero_motion], rigid[is_zero_motion] = 0, 0
        
        pred_pos_rigid = particles + rigid
        pred_pos = (pred_pos_rigid + non_rigid) if not self.config["rigid_only"] else pred_pos_rigid
        
        # # manual update the rod points
        # pred_pos = self.update_rod_points(node_pos_hist, pred_pos, action)
        
        # compute new tool points
        pos_curr_tool = pos_prev_tool + action[:, self.n_object_points:]
        # print(pos_curr_tool[..., -1].mean())
        # breakpoint()
        pred_pos = torch.cat([pred_pos, pos_curr_tool], dim=1)

        return pred_pos, pred_physics
    
    def dynamics_prediction_run_forward(self, action, pos_prev, pred_physics, is_null_action):
        B = action.shape[0]
        N = pos_prev.shape[1]
        
        node_pos_hist = pos_prev
        
        node_feature_hist = action
        # shape (B, N, (his_len + seq_len) * (sum of all feat dim))
        
        if pred_physics is None:
                # pred_physics = torch.zeros(B, len(self.cumul_points) - 1, self.obj_phy_feat_len).to(self.device)
                pred_physics = self.init_physics_params((B, len(self.cumul_points) - 1, self.obj_phy_feat_len))

        # Previous predicted state is the previous state
        pos_prev = pos_prev[:, :self.cumul_points[-1]]
        pos_prev_tool = node_pos_hist[:, self.cumul_points[-1]:, -self.pos_len:]

        LARGEST_VALID_ACTION = 0.015
        # window = data.clone("edge_index", "edge_attr", "y", "batch", "ptr")
        window = pyg.data.Batch(
            x=None,
            edge_index=None,
            edge_attr=None,
            # y=target_state,
            pos=None,  # ground truth for loss computation
            forces=None,
            flows=None,
            # pressure=pressure,
            object_cls=None,
            # rand_index = torch.cat([torch.from_numpy(d['rand_index']) for d in state_list], dim=0)
        )

        window.obj_phy_feat = pred_physics
        start_time = time.time()
        edge_index, edge_attr = connect_edges_batched(
            self.config,
            pos_prev,
            pos_prev_tool,
            N,
            B,  # data.num_graphs,
            self.cumul_points,
        )
        # print(f'\tConstructing graph takes {time.time() - start_time} seconds')

        # sort the edge_index by the first row so that we can unbatch them if desired
        # this is optional, but you need it for visualization to compute edge_slice_indices
        edge_index_indices = torch.argsort(edge_index[0])
        edge_index = edge_index[:, edge_index_indices]
        edge_attr = edge_attr[edge_index_indices]

        window.edge_index = edge_index
        window.edge_attr = edge_attr
        window_pos = torch.cat(
            (node_pos_hist[:, :, : (self.his_len - 1) * self.pos_len], torch.cat((pos_prev, pos_prev_tool), dim=1)),
            dim=-1,
        )
        window.pos = window_pos.view(-1, window_pos.shape[-1])
        window.x = node_feature_hist.view(-1, node_feature_hist.shape[-1])

        # compute the first index in edge_index that corresponds to the first edge of each graph
        # this can be determined by finding the first index greater than or equal to N, then 2N, then 3N, etc.
        # this is because the edge_index is sorted by the first row
        edge_slice_indices = compute_slice_indices(window.edge_index[0], N, B)

        # manually create a dict atttribute. this is a hack
        window._slice_dict = {}
        window._slice_dict['edge_index'] = edge_slice_indices
        window._slice_dict['edge_attr'] = edge_slice_indices
        window._num_graphs = B

        # start_time = time.time()

        # Use DPI network to predict non-rigid and rigid motion norms
        rigid_norm, non_rigid_norm = self.predictor(window)

        # If this model is not supposed to predict physics params, just zero it out
        # print(f'\tDPI takes {time.time() - start_time} seconds: {window}')

        # Denormalize the motion norms and compute the predicted state
        non_rigid = non_rigid_norm * 1  # self.pos_scale

        # Process predictions depending on if we want rigid transformation or not
        rigid = rigid_norm
        
        # record the indices of invalid actions for verbose 
        invalid_indices = []
        for i, one_action in enumerate(action):
            if (one_action.abs() > LARGEST_VALID_ACTION).any() or is_null_action:
                invalid_indices.append(i)
                rigid[i], non_rigid[i] = 0, 0
        if len(invalid_indices) > 0:
            print(f'Invalid action received, predicting zero motion for instances: {invalid_indices}')
        
        # Process predictions if physical interaction is possible by pre-specified criteria
        is_zero_motion = ~self.is_physical_interaction_possible(node_pos_hist, action)
        non_rigid[is_zero_motion], rigid[is_zero_motion] = 0, 0
        
        pred_pos_rigid = pos_prev + rigid
        pred_pos = (pred_pos_rigid + non_rigid) if not self.config["rigid_only"] else pred_pos_rigid
        
        if len(invalid_indices) > 0:
            # manual update the rod points for this step 
            pred_pos = self.update_rod_points(node_pos_hist, pred_pos, action)
        
        # compute new tool points
        pos_curr_tool = pos_prev_tool + action[:, self.n_object_points:]
        # print(pos_curr_tool[..., -1].mean())
        # breakpoint()
        pred_pos = torch.cat([pred_pos, pos_curr_tool], dim=1)
        # print(pos_prev_tool[0, 20:40].mean(0), pos_curr_tool[0, 20:40].mean(0))
        
        return pred_pos

    def predict_step(self, observation_batch, action_history, new_action_samples):
        new_action_samples = torch.from_numpy(new_action_samples).to(self.device)
        
        object_type_feat, vision_feature, node_pos, tac_feat_bubbles, tactile_feat_all_points = observation_batch

        B = new_action_samples.shape[0]
        N = node_pos.shape[1]
        observed_his_len = node_pos.shape[2]
        
        self.estimator.reset_lstm_state()
        pred_physics = None
        
        start_time = time.time()
        
        pos_seq_object = []

        # initial particle state
        particle_pos = node_pos[:, :, 0]

        # start integrating history, using offline data
        for t in range(observed_his_len - 1):
            # only action and tactile are "ground-truth" and need to be given for every step
            action = action_history[:, :, t].float()
            tactile_state = tactile_feat_all_points[:, t:t+2]
            tactile_state = tactile_state.permute(0, 2, 1, 3).reshape(1, N, -1)
            
            particle_pos, pred_physics = self.state_estimator_run_forward(action, particle_pos, tactile_state, 
                                                                          pred_physics)
            if t not in self.pred_physics_list:
                self.pred_physics_list[t] = []
            self.pred_physics_list[t].append((pred_physics.cpu().clone(), 0))

            pos_seq_object.append(particle_pos.repeat(B, 1, 1))
        
        print(f'State estimation completed, taken {time.time() - start_time}s for history length {observed_his_len}')
        
        # convert many things to batches
        # object_type_feat = object_type_feat.repeat(B, 1, 1)
        # vision_feature = vision_feature.repeat(B, 1, 1)
        node_pos = node_pos.repeat(B, 1, 1, 1)
        # tactile_feat_all_points = tactile_feat_all_points.repeat(B, 1, 1, 1)
        # node_pos[0, -20:, :, -1].mean(0)
        
        if observed_his_len > 1:
            pred_physics = pred_physics.repeat(B, 1, 1)
        
        # use the last state in history as the first observation 
        particle_pos = node_pos[:, :, -1, :] 
        # tactile_state = tactile_feat_all_points[:, -1]  # use the last tactile reading 

        start_time = time.time()
        for t in range(new_action_samples.shape[1]):
            action = new_action_samples[:, t].float()  # shape (N, 3)
            non_tool_action = torch.zeros(B, self.n_object_points, 3).to(self.device)
            # non_tool_action[:, 20:40, :] = action.unsqueeze(1).repeat(1, self.points_per_obj[-1], 1)    # the rod
            tool_action = action.unsqueeze(1).repeat(1, N - self.n_object_points, 1)
            node_action_feat = torch.cat([non_tool_action, tool_action], dim=1)
            particle_pos_new = self.dynamics_prediction_run_forward(node_action_feat, particle_pos, pred_physics, t == 0)
            
            # diff = (particle_pos_new - particle_pos)[0][-1]
            # print(f'diff - action = {diff - action[0]}')
            # pdb.set_trace()
            
            particle_pos = particle_pos_new
            pos_seq_object.append(particle_pos)
            # tactile_states.append(tactile_state)

        print(f'Prediction completed, taken {time.time() - start_time}s, '
              f'for sequence length {new_action_samples.shape[1]}')
        
        pos_seq_object = torch.stack(pos_seq_object, dim=1)
        # tactile_states = torch.stack(tactile_states, dim=1)
        pos_seq_object = pos_seq_object  # torch.cat([pos_seq_object, tactile_states], dim=-1)
        pred_state_seq = dict(
            object_obs=pos_seq_object[:, :, :self.cumul_points[-1]].detach().cpu().numpy(),
            # inhand=None, # pos_seq_object[:, :, self.cumul_points[-2]:self.cumul_points[-1]].detach().cpu().numpy(),
            bubble=pos_seq_object[:, :, self.cumul_points[-1]:].detach().cpu().numpy(),
        )
        
        # object_obs=state_seq_object[0, i, : self.cumul_points[-2]].detach().cpu().numpy(),
        #         inhand=state_seq_object[0, i, self.cumul_points[-2]:].detach().cpu().numpy(),
        #         bubble=state_seq_tool[0, i].detach().cpu().numpy(),
        
        return pred_state_seq
    
    def configure_optimizers(self):
        # Freeze the weights of the AE
        for name, param in self.autoencoder.named_parameters():
            param.requires_grad = False
        print(f'Autoencoder frozen. ')

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, "min", factor=0.8, patience=3, verbose=True)
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler,
            "monitor": "val_loss",
        }

    def to_device(self, device):
        self.autoencoder = self.autoencoder.to(device)
        self.autoencoder.set_statistics(device)
        self.config.device = device
        self = self.to(device)
        return self
    
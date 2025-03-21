"""
Baseline method for learning particle dynamics. No state estimation.
"""

from collections import OrderedDict

import time 
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
import torch_geometric as pyg

# from pytorch_memlab import profile, profile_every, MemReporter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dynamics.dataset import connect_edges, connect_edges_batched, compute_slice_indices
from dynamics.models.dpi_net import DPINet
from dynamics.loss import PositionLoss, Chamfer, EMDCPU, MSE
from dynamics.models.autoencoder import AutoEncoder

from utils_general import AverageMeter


class DynamicsPredictor(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        # Extract configuration parameters
        self.config = config
        self.points_per_obj = config["n_points"]
        self.cumul_points = np.cumsum([0] + self.points_per_obj)
        self.type_feat_len = 1
        self.vis_feat_len = config["feature_dim_vision"]
        self.act_feat_len = config["feature_dim_action"]
        self.pos_len = 3  # vector length of position (xyz)
        self.his_len = config["history_length"]
        self.seq_len = config["sequence_length"]
        self.T = self.his_len + self.seq_len
        self.tac_feat_dim = config["ae_enc_dim"]
        self.n_object_points = sum(config["n_points"])  # number of points except tool
        self.lr = config["optimizer"]["lr"]
        self.num_bubbles = 2
        self.zero_tactile = config["zero_tactile"]
        self.tactile_use_gt = config["tactile_use_gt"]
        
        if self.zero_tactile is True:
            assert not self.tactile_use_gt
        if self.zero_tactile:
            print(f'tactile will be zeroed. ')
        self.visual_blind = config["visual_blind"]
        
        if self.visual_blind:
            assert self.his_len == 1, f'if visual is blind, his_len should be 1 for this model, but got {self.his_len}'
        
        self.recurr_T = config["recurr_T"]
        self.teacher_forcing_thres = config["teacher_forcing_thres"]

        # # Load normalization constants
        # self.pos_mean = torch.tensor(
        #     stats["position_mean"], dtype=torch.float32, device=config.device
        # )
        # self.pos_scale = torch.tensor(
        #     stats["position_scale"], dtype=torch.float32, device=config.device
        # )
        
        # check if current user is aibo or albert (extreme hackiness)
        import os, pwd
        user = pwd.getpwuid(os.getuid()).pw_name
        if user == "albert":
            # replace the path to the tactile data
            ae_checkpoint = config["ae_checkpoint"].replace("/home/aibo/intern", "/home/albert/github")
            ae_checkpoint = "/home/albert/github/robopack/dynamics/pretrained_ae/v24_5to5_epoch=101-step=70482_corrected.ckpt"
            print("AE checkpoint path is ", ae_checkpoint)
            device = 'cuda:0'
        else:
            ae_checkpoint = config["ae_checkpoint"]
            ae_checkpoint = "/svl/u/boai/robopack/dynamics/pretrained_ae/v24_5to5_epoch=101-step=70482_corrected.ckpt"
            # device = 'cuda:0' # config.device
        # config.device = device        # may cause error during training

        # tactile-related
        self.autoencoder = AutoEncoder.load_from_checkpoint(frozen=True,
                                                            checkpoint_path=ae_checkpoint,
                                                            map_location='cuda:0')
        print(f'Autoencoder loaded from {config["ae_checkpoint"]}')

        # Initialize the GNN layer
        self.layers = DPINet(config)

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
        # self.loss_dicts = []
        self.error_seq = []
        self.pred_state_seqs = []
        # self.pred_tac_seqs = []
        # self.gt_tac_seqs = []
        self.total_tac_loss, self.total_emd_loss, self.total_chamfer_loss, self.total_mse_loss = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        
        # Save hyperparameters
        self.save_hyperparameters()

    # @profile
    def forward(self, data, t, tac_feat_bubbles, pred_pos_prev, pred_tac_prev, train):
        # forward_start_t = time.time() 
        # DataBatch(x=(B * N), edge_index=(2, E), edge_attr=(E, 4), y=[B * N_p, 3],
        # pos=[B * N, (his_len + seq_len) * pos_len], batch=[B * N], ptr=[?])
        B = data.num_graphs
        N = data.num_nodes // B

        # Step 1: Extract and process pos history
        # the history is from t to (t + self.his_len - 1)
        node_pos = data.pos.view(B, N, -1, self.pos_len)
        T = node_pos.shape[2]
        node_pos_hist = node_pos[:, :, t: (t + self.his_len)].view(B, N, -1)
        # Step 3: Extract target state for loss computation
        gt_pos = node_pos[
                 :,
                 : self.cumul_points[-1],  # take all the points specified
                 (t + self.his_len) : (t + self.his_len + 1) 
                 ].squeeze(2)  # ground truth is the pos at the next time step

        # Step 1.1: Get tactile data
        if self.zero_tactile:
            tac_feat_bubbles = torch.zeros_like(tac_feat_bubbles)
            # tac_feat_bubbles = torch.normal(torch.zeros_like(tac_feat_bubbles), torch.zeros_like(tac_feat_bubbles) + 0.6)

        tactile_feat_objects = torch.zeros(B, tac_feat_bubbles.shape[1],
                                           self.n_object_points,
                                           self.config["ae_enc_dim"]).to(self.device)
        tactile_feat_all_points = torch.cat([tactile_feat_objects, tac_feat_bubbles], dim=2)

        # debugging
        # for i in range(tactile_feat_all_points.shape[1]):
        #     tactile_feat_all_points[:, i] = torch.zeros_like(tactile_feat_all_points[:, i]) + i

        # get the tactile feature history, and the ground truth tactile feature, which
        # is the next reading after the history
        tactile_feat_hist = tactile_feat_all_points[:, t:(t + self.his_len)]
        tactile_feat_gt = tactile_feat_all_points[:, (t + self.his_len):(t + self.his_len + 1)]

        # replace the last reading with the predicted reading, if available 
        if t > 0 and (not self.tactile_use_gt):
            tactile_feat_hist[:, -1] = pred_tac_prev
            # the math should be correct here, -1 corresponds to (t-1) + self.his_len
            # TODO: Not only the last one, but all previous history tactile feat should be predicted, not gt. 

        # reshape the tactile feature so that the feats across time steps are stacked along the last dim
        # WARNING: You cannot directly do .view(B, N, -1) because the two dimensions
        # are not adjacent (space not contiguous) and this would not concate the two dimensions along the last dim
        # Instead, do permute and reshape
        tactile_feat_hist = tactile_feat_hist.permute(0, 2, 1, 3).reshape(B, N, -1)
        tactile_feat_gt = tactile_feat_gt.permute(0, 2, 1, 3).reshape(B, N, -1)

        # Step 2: Extract and process node feature history
        # for the format, see the dataset flass
        # feature is a concat of (particle type, visual features (rgb), and actions)
        node_feature = data.x.view(B, N, -1)
        # assert node_feature.shape[-1] == self.type_feat_len + self.vis_feat_len +
        # (self.his_len + self.seq_len - 1) * self.pos_len, "you need to rebuild dataset"

        type_hist = node_feature[
                    ...,
                    : self.type_feat_len
                    ]

        vision_feature_hist = node_feature[
                              ...,
                              self.type_feat_len: self.type_feat_len + self.vis_feat_len,
                              ]
        vision_feature_hist = torch.zeros_like(vision_feature_hist)

        action_hist = node_feature[:, :, self.type_feat_len + self.vis_feat_len :].view(B, N, T - 1, -1)
        action_hist = action_hist[:, :, t: (t + self.his_len)].reshape(B, N, -1)

        # node features, concat of type (B, N, 1), visual feature (B, N, 3), action feat (B, N, 3), and tac (B, N, 6)
        node_feature_hist = torch.cat(
            (type_hist, vision_feature_hist, action_hist, tactile_feat_hist), dim=-1
        )

        # Process input data depending on if it's the first action or not
        if t == 0:
            # Last history state is the previous state
            pos_prev = node_pos_hist[:, : self.cumul_points[-1], -self.pos_len:]

            # Update pos and x in data
            window = data.clone("edge_index", "edge_attr", "y", "batch", "ptr")
            window.pos = node_pos_hist.view(-1, node_pos_hist.shape[-1])
            window.x = node_feature_hist.view(-1, node_feature_hist.shape[-1])
        else:
            # else_start_t = time.time() 
            # Previous predicted state is the previous state
            pos_prev = pred_pos_prev
            pos_prev_tool = node_pos_hist[:, self.cumul_points[-1]:, -self.pos_len:]
            # import time
            # start = time.time()
            # window_list = []
            # for batch_idx, (pred_pos, tool_pos) in enumerate(
            #         zip(pos_prev, pos_prev_tool)
            # ):
            #     assert pred_pos.shape[0] == self.cumul_points[-1], \
            #         f"self.cumul_points[-1] = {self.cumul_points[-1]} should be the total number of points predicted, " \
            #         f"but got {pred_pos.shape[0]} points instead"
            #
            #     # Prepare particle positions for graph construction
            #     # TODO: check if this should be modified to fit multiple objects
            #     pos_dict = OrderedDict(
            #         object_obs=(0, pred_pos[self.cumul_points[0]: self.cumul_points[1]]),
            #         inhand=(self.cumul_points[1], pred_pos[self.cumul_points[1]:self.cumul_points[2]]),
            #         bubble=(self.cumul_points[2], tool_pos),
            #     )
            #
            #     # IMPORTANT NOTE: edge building also happens during training (here)
            #     # compounding error issue must be taken into consideration
            #     edge_index, edge_attr = connect_edges(
            #         self.config,
            #         pos_dict,
            #     )
            #
            #     # concat the tool pose (action) and the predicted object pose,
            #     # as well as history poses
            #     pos_all = torch.cat((pred_pos, tool_pos), dim=0)
            #     window_pos = torch.cat(
            #         (node_pos_hist[batch_idx, :, : (self.his_len - 1) * self.pos_len], pos_all),
            #         dim=-1,
            #     )
            #
            #     # WARNING: the below is toxic. Wrong! Need to reshap back before assigning values.
            #     # replace the tactile reading of the prev step
            #     # with the tactile reading predicted at the last step
            #     # this works regardless of self.his_len values
            #     # node_feature_hist[batch_idx, :, -self.tac_feat_dim:] = pred_tac_feat[batch_idx]
            #     # print(pred_tac_feat[batch_idx].mean(), pred_tac_feat[batch_idx].std(), pred_tac_feat[batch_idx].max())
            #
            #     # Construct the Data object
            #     window = data[batch_idx].clone("y", "batch", "ptr")
            #     # Data(x=[80, 19], edge_index=[2, 2058], edge_attr=[2058, 4], pos=[80, 18], forces=[6, 2, 7],
            #          # flows=[6, 40, 2], object_cls=[6])
            #     window.edge_index = edge_index
            #     window.edge_attr = edge_attr
            #     window.pos = window_pos
            #     window.x = node_feature_hist[batch_idx]
            #     # Data(x=[80, 12], edge_index=[2, 1576], edge_attr=[1576, 4], pos=[80, 3], forces=[6, 2, 7],
            #     #      flows=[6, 40, 2], object_cls=[6])
            #     window_list.append(window)
            # # Construct the data batch
            # window = pyg.data.Batch.from_data_list(window_list)
            # print(f'window construction time: {time.time() - start}')
            #

            window = data.clone("edge_index", "edge_attr", "y", "batch", "ptr")
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

            # assert that for all edges in all_edges, they are somewhere in window.edge_index, not necessarily in the same order.
            # also, for each edge in window.edge_index, it should be somewhere in all_edges
            # for i in range(window2.edge_index.shape[1]):
            #     assert torch.any(torch.all(torch.permute(window.edge_index, (1, 0)) == window2.edge_index[:, i], dim=1)), f'edge {window2.edge_index[:, i]} not found in window.edge_index'
            # for i in range(window.edge_index.shape[1]):
            #     assert torch.any(torch.all(torch.permute(window2.edge_index, (1, 0)) == window.edge_index[:, i], dim=1)), f'edge {window.edge_index[:, i]} not found in all_edges'
            
        # Visualize the graph
        # import matplotlib.pyplot as plt
        # import networkx as nx
        # from datetime import datetime
        #
        # for batch_idx in range(B):
        #     print(t, window[batch_idx])
        #     g = pyg.utils.to_networkx(window[batch_idx], to_undirected=True)
        #     pos_dict = {
        #         t: pos for t, pos in enumerate(window[batch_idx].pos[:, :2].tolist())
        #     }
        #     nx.draw_networkx(g, pos_dict, with_labels=False, node_size=10)
        #     plt.savefig(
        #         f'scratch/{datetime.now().strftime("%d-%b-%Y-%H_%M_%S_%f")}_s{t}_b{batch_idx}.png'
        #     )
        #     plt.close()

        # # Normalize the position in the state
        # window.pos = (window.pos - self.pos_mean.repeat(self.his_len)) / self.pos_scale
        
        # Use DPI network to predict non-rigid and rigid motion norms
        rigid_norm, non_rigid_norm, pred_tac_feat = self.layers(window)

        # Denormalize the motion norms and compute the predicted state
        non_rigid = non_rigid_norm # * self.pos_scale

        # Process predictions depending on if we want rigid transformation or not
        rigid = rigid_norm   # * self.pos_scale
        pred_pos_rigid = pos_prev + rigid
        pred_pos = (pred_pos_rigid + non_rigid) if not self.config["rigid_only"] else pred_pos_rigid

        # Calculate loss
        loss = {}
        if train:
            position_loss, position_losses = self.position_loss(pred_pos, gt_pos, return_losses=True)
            loss["train_pos"] = position_loss * self.pos_loss_weight
            loss["train_pos_losses"] = position_losses

            if (not self.tactile_use_gt):
                loss["train_tac"] = self.tactile_loss(pred_tac_feat[:, self.cumul_points[-1]:],
                                                      tactile_feat_gt[:, self.cumul_points[-1]:])
            else:
                loss["train_tac"] = 0
                # loss["train_tac1"] = self.tactile_loss(pred_raw_tac[:, self.cumul_points[-1]:, :3],
                #                                       gt_tactile[:, self.cumul_points[-1]:, :3])
                # loss["train_tac2"] = self.tactile_loss(pred_raw_tac[:, self.cumul_points[-1]:, 3:5],
                #                                        gt_tactile[:, self.cumul_points[-1]:, 3:5])
                # loss["train_tac3"] = self.tactile_loss(pred_raw_tac[:, self.cumul_points[-1]:, 5:],
                #                                        gt_tactile[:, self.cumul_points[-1]:, 5:])

        else:
            loss["chamfer"] = self.chamfer_loss(pred_pos, gt_pos)
            loss["emd"] = self.emd_loss(pred_pos, gt_pos)
            loss["mse"] = self.mse_loss(pred_pos, gt_pos)
            if (not self.tactile_use_gt):
                loss["tac"] = self.tactile_loss(pred_tac_feat[:, self.cumul_points[-1]:],
                                                tactile_feat_gt[:, self.cumul_points[-1]:])
            else:
                loss["tac"] = 0

        # forward_end_t = time.time()
        # if t > 0:
        #     print(f'forward takes {forward_end_t - forward_start_t} seconds, in which the else block takes {else_end_t - else_start_t} seconds. ')

        return loss, pred_pos, pred_tac_feat, (gt_pos, tactile_feat_gt)

    def dynamics_prediction_run_forward(self, action, pos_prev, tac_prev):
        B = action.shape[0]
        N = pos_prev.shape[1]

        node_pos_hist = pos_prev
        # shape (B, N, (his_len + seq_len) * (sum of all feat dim))

        # Previous predicted state is the previous state
        pos_prev = pos_prev[:, :self.cumul_points[-1]]
        pos_prev_tool = node_pos_hist[:, self.cumul_points[-1]:, -self.pos_len:]

        # should have shape (B, N, (his_len) * (sum of all feat dim))
        tactile_feat_hist = tac_prev
        object_points_list = list(self.points_per_obj) + [pos_prev_tool.shape[1]]
        type_hist = torch.cat(
            [torch.full((n, 1), i, dtype=torch.int)
             for i, n in enumerate(object_points_list)],
            dim=0,
        ).unsqueeze(0).repeat(B, 1, 1).to(self.device)
        vision_feature_hist = torch.zeros(B, N, self.vis_feat_len).to(self.device)
        action_hist = action

        node_feature_hist = torch.cat((type_hist, vision_feature_hist, action_hist, tactile_feat_hist), dim=-1)

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
        rigid_norm, non_rigid_norm, pred_tac_feat = self.layers(window)

        # If this model is not supposed to predict physics params, just zero it out
        # print(f'\tDPI takes {time.time() - start_time} seconds: {window}')

        # Denormalize the motion norms and compute the predicted state
        non_rigid = non_rigid_norm * 1  # self.pos_scale

        # Process predictions depending on if we want rigid transformation or not
        rigid = rigid_norm

        # record the indices of invalid actions for verbose
        invalid_indices = []
        for i, one_action in enumerate(action):
            if (one_action.abs() > LARGEST_VALID_ACTION).any():
                invalid_indices.append(i)
                rigid[i], non_rigid[i] = 0, 0
        if len(invalid_indices) > 0:
            print(f'Invalid action received, predicting zero motion for instances: {invalid_indices}')

        # Process predictions if physical interaction is possible by pre-specified criteria
        # is_zero_motion = ~self.is_physical_interaction_possible(node_pos_hist, action)
        # non_rigid[is_zero_motion], rigid[is_zero_motion] = 0, 0

        pred_pos_rigid = pos_prev + rigid
        pred_pos = (pred_pos_rigid + non_rigid) if not self.config["rigid_only"] else pred_pos_rigid

        # manual update the rod points
        # print(pred_pos[0, 20:40].mean(0), self.update_rod_points(node_pos_hist, pred_pos, action)[0, 20:40].mean(0),
        # action[0, 20:40].mean(0))
        pred_pos = self.update_rod_points(node_pos_hist, pred_pos, action)

        # compute new tool points
        pos_curr_tool = pos_prev_tool + action[:, self.n_object_points:]
        pred_pos = torch.cat([pred_pos, pos_curr_tool], dim=1)
        # print(pos_prev_tool[0, 20:40].mean(0), pos_curr_tool[0, 20:40].mean(0))
        # pdb.set_trace()

        return pred_pos, pred_tac_feat

    def update_rod_points(self, prev_pos, predicted_pos, pointwise_action):
        predicted_pos = predicted_pos.clone()
        rod_start_idx, rod_end_idx = self.cumul_points[-2], self.cumul_points[-1]  # the last object is the rod
        predicted_pos[:, rod_start_idx: rod_end_idx] = prev_pos[:, rod_start_idx: rod_end_idx] + \
                                                       pointwise_action[:, rod_start_idx: rod_end_idx]
        return predicted_pos
    def training_step(self, batch, batch_idx):
        train_pos_loss, tac_loss = 0, 0
        pred_pos, pred_tactile = None, None
        
        tac_feat_bubbles = self.autoencoder.encode_structured(batch)
        box_losses = torch.tensor([0, 0], dtype=torch.float32)

        # Calculate total loss over the sequence
        for i in range(self.seq_len):
            loss, pred_pos, pred_tactile, gts = self.forward(batch, i, tac_feat_bubbles, pred_pos, pred_tactile, True)
            train_pos_loss += loss["train_pos"]
            box_losses += torch.tensor(loss["train_pos_losses"])

            if (not self.tactile_use_gt):
                tac_loss += loss["train_tac"]

            gt_pos, gt_tactile = gts
            if loss["train_pos"] > self.teacher_forcing_thres:
                pred_pos = gt_pos
                pred_tactile = gt_tactile
  
        # normalize the loss by the sequence length
        train_pos_loss /= self.seq_len
        tac_loss /= self.seq_len
        box_losses /= self.seq_len

        # from torchviz import make_dot
        # make_dot(pred_pos, params=dict(self.named_parameters())).render(
        #     "computation_graph", format="png"
        # )
        # pdb.set_trace()

        # memory_alloc = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()

        self.log(
            "train_loss",
            train_pos_loss,
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            batch_size=batch.num_graphs,
        )

        for i, loss in enumerate(box_losses.tolist()):
            self.log(
                f"train_obj_loss_{i}",
                loss,
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

        if (not self.tactile_use_gt):
            self.log(
                "tac_loss",
                tac_loss,
                prog_bar=True,
                on_epoch=True,
                on_step=True,
                batch_size=batch.num_graphs,
            )

        self.log(
            "total_loss",
            train_pos_loss + tac_loss,
            prog_bar=False,
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

        # self.log(
        #     "memory_alloc",
        #     memory_alloc,
        #     prog_bar=True,
        #     on_epoch=False,
        #     on_step=True,
        #     batch_size=batch.num_graphs,
        # )

        return train_pos_loss + tac_loss  # IMPORTANT: need to return all trainable losses

    def validation_step(self, batch, batch_idx):
        train_pos_loss, tac_loss = 0, 0
        pred_pos, pred_tactile = None, None

        tac_feat_bubbles = self.autoencoder.encode_structured(batch)
        box_losses = torch.tensor([0, 0], dtype=torch.float32)

        # Calculate total loss over the sequence
        for i in range(self.seq_len):
            loss, pred_pos, pred_tactile, gts = self.forward(batch, i, tac_feat_bubbles, pred_pos, pred_tactile, True)
            train_pos_loss += loss["train_pos"]
            box_losses += torch.tensor(loss["train_pos_losses"])

            if (not self.tactile_use_gt):
                tac_loss += loss["train_tac"]

            gt_pos, gt_tactile = gts
            if loss["train_pos"] > self.teacher_forcing_thres:
                pred_pos = gt_pos
                pred_tactile = gt_tactile

        # normalize the loss by the sequence length
        train_pos_loss /= self.seq_len
        tac_loss /= self.seq_len
        box_losses /= self.seq_len
        
        for i, loss in enumerate(box_losses.tolist()):
            self.log(
                f"val_obj_loss_{i}",
                loss,
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

        if (not self.tactile_use_gt):
            self.log(
                "val_tac_loss",  
                tac_loss,
                prog_bar=True,
                on_epoch=True,
                on_step=True,
                batch_size=batch.num_graphs,
            )

    def get_particle_tactile(self, data):
        B = data.num_graphs
        N = data.num_nodes // B

        forces, flows, pressure = data.forces, data.flows, data.pressure
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

        # # set up variables for tactile prediction
        # gt_forces, gt_flows, gt_pressure = self.get_particle_tactile(batch)
        # gt_forces, gt_flows, gt_pressure = [x.cpu().numpy() for x in [gt_forces, gt_flows, gt_pressure]]
        # # every of shape (B, T, 2, n_points_per_bubble, x)
        #
        # # initialize with a ground truth for the first step
        # tac_pred_seqs = [
        #     dict(
        #         forces=np.concatenate([gt_forces[:, 0][..., :2], gt_forces[:, 0][..., -1:]], axis=-1),
        #         flows=gt_flows[:, 0],
        #         pressure=gt_pressure[:, 0]
        #     )
        # ]

        # # Log initial zero losses for the first H steps
        # for i in range(self.his_len):
        #     self.log_dict(
        #         {
        #             f"chamfer_{i}": 0.0,
        #             f"emd_{i}": 0.0,
        #             f"mse_{i}": 0.0,
        #             f"tactile_mse_{i}": 0
        #         },
        #         on_epoch=False,
        #         on_step=True,
        #         batch_size=B,
        #     )

        pred_pos, pred_tactile = None, None
        seq_tac_loss_logger = AverageMeter()

        tac_feat_bubbles = self.autoencoder.encode_structured(batch)
        # For each state in the sequence (S), calculate the predicted state and losses
        for i in range(S - self.his_len):
            loss, pred_pos, pred_tactile, gts = self.forward(batch, i, tac_feat_bubbles, pred_pos, pred_tactile, False)
            # tactile_feat, tactile_raw = pred_tactile

            # post-process predicted tactile
            # pred_forces = tactile_raw[:, self.cumul_points[-1]:, :3].view(B, self.num_bubbles, n_bubble_points, -1)
            # pred_flows = tactile_raw[:, self.cumul_points[-1]:, 3:5].view(B, self.num_bubbles, n_bubble_points, -1)
            # pred_pressure = tactile_raw[:, self.cumul_points[-1]:, 5:].view(B, self.num_bubbles, n_bubble_points, -1)
            #
            # pred_forces = self.denorm_tactile_reading(pred_forces, self.force_mean, self.force_scale)
            # pred_flows = self.denorm_tactile_reading(pred_flows, self.flow_mean, self.flow_scale)
            # pred_pressure = self.denorm_tactile_reading(pred_pressure, self.pressure_mean, self.pressure_scale)

            # tac_pred_seqs.append(
            #     dict(
            #         forces=pred_forces,
            #         flows=pred_flows,
            #         pressure=pred_pressure
            #     )
            # )

            if (not self.tactile_use_gt):
                seq_tac_loss_logger.update(loss["tac"].item())
                self.total_tac_loss.update(loss["tac"].item())
            self.total_emd_loss.update(loss["emd"].item())
            self.total_chamfer_loss.update(loss["chamfer"].item())
            self.total_mse_loss.update(loss["mse"].item())

            # Log the losses
            self.log_dict(
                {f"{k}_{i + self.his_len}": v for k, v in loss.items()},
                on_epoch=False,
                on_step=True,
                batch_size=B,
            )

            # Append the predicted state to the object state sequence
            pos_seq_object.append(pred_pos[:, None])

        print(f'last-state error: tactile = {loss["tac"].item()}, mse = {loss["mse"].item()}')
        print(
            f'count: {self.total_tac_loss.count}, seq tac loss avg: {seq_tac_loss_logger.avg}, mse loss avg: {self.total_mse_loss.avg} '
            f'total tac loss avg: {self.total_tac_loss.avg}. ')
        
        self.error_seq.append((loss["mse"].item(), seq_tac_loss_logger.avg))

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
                object_obs=state_seq_object[0, i, : self.cumul_points[1]].detach().cpu().numpy(),
                inhand=state_seq_object[0, i, self.cumul_points[1]:].detach().cpu().numpy(),
                bubble=state_seq_tool[0, i].detach().cpu().numpy(),
            )
            for i in range(S)
        ]

        self.pred_state_seqs.append(pred_state_seq)
        # self.pred_tac_seqs.append(tac_pred_seqs)


    def predict_step(self, observation_batch, action_history, new_action_samples):
        new_action_samples = torch.from_numpy(new_action_samples).to(self.device)
        object_type_feat, vision_feature, node_pos, tac_feat_bubbles, tactile_feat_all_points = observation_batch

        B = new_action_samples.shape[0]
        N = node_pos.shape[1]
        observed_his_len = node_pos.shape[2]

        start_time = time.time()

        pos_seq_object = []
        tactile_states = []

        # initial tactile state
        # convert many things to batches
        # object_type_feat = object_type_feat.repeat(B, 1, 1)
        # vision_feature = vision_feature.repeat(B, 1, 1)
        node_pos = node_pos.repeat(B, 1, 1, 1)
        tactile_feat_all_points = tactile_feat_all_points.repeat(B, 1, 1, 1)

        # use the last state in history as the first observation
        particle_pos = node_pos[:, :, -1, :]
        tac_prev = tactile_feat_all_points[:, -1]  # use the last tactile reading

        start_time = time.time()
        for t in range(new_action_samples.shape[1]):
            action = new_action_samples[:, t].float()  # shape (N, 3)
            non_tool_action = torch.zeros(B, self.n_object_points, 3).to(self.device)
            non_tool_action[:, 20:40, :] = action.unsqueeze(1).repeat(1, self.points_per_obj[-1], 1)    # the rod
            tool_action = action.unsqueeze(1).repeat(1, N - self.n_object_points, 1)
            node_action_feat = torch.cat([non_tool_action, tool_action], dim=1)

            particle_pos_new, pred_tac_feat = self.dynamics_prediction_run_forward(node_action_feat, particle_pos, tac_prev)

            # diff = (particle_pos_new - particle_pos)[0][-1]
            # print(f'diff - action = {diff - action[0]}')
            # pdb.set_trace()

            particle_pos = particle_pos_new
            tac_prev = pred_tac_feat

            pos_seq_object.append(particle_pos.clone())
            tactile_states.append(tac_prev.clone())

        print(f'Prediction completed, taken {time.time() - start_time}s, '
              f'for sequence length {new_action_samples.shape[1]}')

        pos_seq_object = torch.stack(pos_seq_object, dim=1)
        tactile_states = torch.stack(tactile_states, dim=1)
        pos_seq_object = pos_seq_object  # torch.cat([pos_seq_object, tactile_states], dim=-1)

        pred_state_seq = dict(
            object_obs=pos_seq_object[:, :, : self.cumul_points[1]].detach().cpu().numpy(),
            inhand=pos_seq_object[:, :, self.cumul_points[1]:self.cumul_points[2]].detach().cpu().numpy(),
            bubble=pos_seq_object[:, :, self.cumul_points[2]:].detach().cpu().numpy(),
        )

        return pred_state_seq


    def predict_step_v1(self, batch, batch_idx):
        B = batch.num_graphs
        N = batch.num_nodes // B
        S = batch.pos.shape[-1] // self.pos_len
        n_bubble_points = (N - self.n_object_points) // 2

        # Get the ground truth state
        gt_pos_seq = batch.pos.view(B, N, S, self.pos_len).transpose(1, 2)

        # Get the state history for each object in the batch
        pos_seq_object = [gt_pos_seq[:, : self.his_len, : self.cumul_points[-1]]]

        pred_pos, pred_tactile = None, None
        seq_tac_loss_logger = AverageMeter()

        tac_feat_bubbles = self.autoencoder.encode_structured(batch)
        # For each state in the sequence (S), calculate the predicted state and losses
        for i in range(S - self.his_len):
            loss, pred_pos, pred_tactile, gts = self.forward(batch, i, tac_feat_bubbles, pred_pos, pred_tactile, False)

            seq_tac_loss_logger.update(loss["tac"].item())
            self.total_tac_loss.update(loss["tac"].item())
            self.total_emd_loss.update(loss["emd"].item())
            self.total_chamfer_loss.update(loss["chamfer"].item())
            self.total_mse_loss.update(loss["mse"].item())
            # Append the predicted state to the object state sequence
            pos_seq_object.append(pred_pos[:, None])

        # print(f'last-state error: tactile = {loss["tac"].item()}, mse = {loss["mse"].item()}')
        # print(
        #     f'count: {self.total_tac_loss.count}, seq tac loss avg: {seq_tac_loss_logger.avg}, mse loss avg: {self.total_mse_loss.avg} '
        #     f'total tac loss avg: {self.total_tac_loss.avg}. ')

        self.error_seq.append((loss["mse"].item(), seq_tac_loss_logger.avg))

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

        pred_state_seq = dict(
            object_obs=state_seq_object[:, :, : self.cumul_points[1]].detach().cpu().numpy(),
            inhand=state_seq_object[:, :, self.cumul_points[1]:].detach().cpu().numpy(),
            bubble=state_seq_tool[:, :].detach().cpu().numpy(),
        )

        # self.pred_tac_seqs.append(tac_pred_seqs)
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
            "monitor": "total_loss",
        }

    def to_device(self, device):
        self.autoencoder = self.autoencoder.to(device)
        self.autoencoder.set_statistics(device)
        self.config.device = device
        self = self.to(device)
        return self
    
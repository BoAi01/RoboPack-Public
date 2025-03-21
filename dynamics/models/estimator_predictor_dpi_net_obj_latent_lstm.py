"""
PyG implementation of an improved DPI-Net for learning particle dynamics using our RoboPack framework.
This module is core to predicting particle motions in our proposed method.

Compared to the original DPI-Net, we have two separate models, estimator and predictor that process
the graph differently.
"""

import itertools
import pdb

import numpy as np
import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing

from dynamics.utils_dynamics import MLP, rigid_params_to_motion


class EstimatorDPINet(MessagePassing):
    def __init__(self, config):
        super().__init__(aggr="add")

        # Configurations
        self.config = config
        self.N_list = config["n_points"]
        self.N_cum = np.cumsum([0] + self.N_list)
        self.F_t = 1
        self.F_v = config["feature_dim_vision"]
        self.F_a = config["feature_dim_action"]
        # self.F_tac = config["tactile_feat_dim"]
        self.P = 3
        self.H = config["history_length"]
        assert self.H == 1, "history length must be 1 for this recurrent GNN"
        self.has_rigid_motion = config["has_rigid_motion"]
        self.euler_angle_bound = config["euler_angle_bound"]
        self.rigid_dim = config["rigid_dim"]
        self.tactile_dim = config["ae_enc_dim"]
        self.obj_phy_feat_len = config["dpi_net"]["obj_phy_feat_len"]
        self.history_before_rigid = config["dpi_net"]["history_before_rigid"]

        self.p_step = config["dpi_net"]["propgation_step"]
        hidden_size_node = config["dpi_net"]["hidden_size_node"]
        hidden_size_edge = config["dpi_net"]["hidden_size_edge"]
        hidden_size_effect = config["dpi_net"]["hidden_size_effect"]
        n_layers = config["dpi_net"]["n_layers"]

        # NodeEncoder
        node_feature_dim = self.H * (self.F_a + self.P + self.tactile_dim * 2) + self.obj_phy_feat_len

        self.node_encoder = MLP(
            [node_feature_dim, hidden_size_node, hidden_size_effect],
            n_layers=n_layers,
            last_relu=True,
        )

        # EdgeEncoder
        self.edge_encoder = MLP(
            [2 * node_feature_dim + 3 + 1, hidden_size_edge, hidden_size_effect],
            n_layers=n_layers,
            last_relu=True,
        )

        # Propagators for nodes and edges
        self.node_propagator = nn.Sequential(
            nn.Linear(hidden_size_effect * 2, hidden_size_effect),
            nn.ReLU(),
        )
        self.edge_propagator = nn.Sequential(
            nn.Linear(hidden_size_effect * 3, hidden_size_effect),
            nn.ReLU(),
        )

        # Prediction heads
        # rigid: (euler, translation)
        self.rigid_predictor = MLP(
            [hidden_size_effect, hidden_size_effect, self.rigid_dim], n_layers=n_layers
        )

        # non_rigid motion
        self.non_rigid_predictor = MLP(
            [hidden_size_effect, hidden_size_effect, self.P], n_layers=n_layers
        )

        self.physics_predictor = MLP(
            [hidden_size_effect, hidden_size_effect, self.obj_phy_feat_len], n_layers=n_layers
        )

        # memory module 
        self.lstm = nn.LSTM(hidden_size_effect, hidden_size_effect, 2, batch_first=True)
        self.lstm_hn_cn = None 

    # @profile
    def forward(self, data):
        # Extract the number of graphs and nodes
        B = data.num_graphs
        N = data.num_nodes // B

        # Compute offsets
        # basically normalize objects points with respect to each its center
        pos_expand = data.pos.view(B, N, self.H, self.P)
        offsets = []
        obj_indices = self.N_cum.tolist() + [N]
        for i in range(len(obj_indices) - 1):
            object_points = pos_expand[:, obj_indices[i] : obj_indices[i + 1]]
            object_center = torch.mean(object_points, dim=1, keepdim=True)
            offset_pos = object_points - object_center
            offsets.append(offset_pos)
        offset_pos = torch.cat(offsets, dim=1)

        # node_input: (B * N, particle_input_dim)
        data.obj_phy_feat = data.obj_phy_feat.view(B, len(self.N_list), self.obj_phy_feat_len)
        physics_params = []
        for i in range(len(self.N_list)):
            physics_params.append(data.obj_phy_feat[:, i, :].unsqueeze(1).repeat(1, self.N_list[i], 1))
        physics_params.append(torch.zeros(B, N - sum(self.N_list), self.obj_phy_feat_len, device=physics_params[0].device))
        physics_params = torch.cat(physics_params, dim=1)

        node_phy_feat = physics_params.view(-1, self.obj_phy_feat_len)
        node_input = torch.cat((data.x, offset_pos.view(-1, self.H * self.P), node_phy_feat), dim=-1)

        # node_encode: (B * N, hidden_size_effect)
        node_encode = self.node_encoder(node_input)

        # Process edge inputs
        # sender/receiver_input: (B * R, particle_input_dim)
        # edge_attr: (B * R, 3 + 1)
        receiver_input = node_input[data.edge_index[0]]
        sender_input = node_input[data.edge_index[1]]
        # edge_input: (B * R, 2 * particle_input_dim + 3 + 1)
        edge_input = torch.cat(
            [
                receiver_input,
                sender_input,
                data.edge_attr,
            ],
            dim=-1,
        )
        edge_encode = self.edge_encoder(edge_input)

        # Particle effect after encoding
        node_effect = node_encode

        # stack of GNN layers
        for i in range(self.p_step):
            edge_effect = self.propagate(
                edge_index=data.edge_index, x=node_effect, edge_attr=edge_encode
            )

            node_effect = self.node_propagator(
                torch.cat([node_encode, edge_effect], dim=-1)
            )

        # post-processing, predicting quantities of interest
        # predict non-rigid motions: (B, N_p, D)
        node_effect_object = node_effect.view(B, N, -1)[:, : self.N_cum[-1], :]
        non_rigid = self.non_rigid_predictor(node_effect_object)

        # predict rigid motions
        if self.has_rigid_motion:
            # obtain the instance-level feature for each object: (B, N_inst, node effect size)
            instance_effect_object = torch.cat(
                [
                    torch.mean(
                        node_effect_object[:, self.N_cum[i] : self.N_cum[i + 1]],
                        keepdim=True,
                        dim=1,
                    )
                    for i in range(len(self.N_cum) - 1)
                ],
                dim=1,
            )

            if self.history_before_rigid:
                # predict physical params
                instance_effect_object = instance_effect_object.view(B * len(self.N_list), -1)
                instance_effect_object, self.lstm_hn_cn = self.lstm(instance_effect_object.unsqueeze(1), self.lstm_hn_cn)
                instance_effect_object = instance_effect_object.squeeze(1).view(B, len(self.N_list), -1)
                physics_params = self.physics_predictor(instance_effect_object) + data.obj_phy_feat     # only predict residual 

                # rigid_params: (B, N_inst, 6)
                rigid_params = self.rigid_predictor(instance_effect_object)

                # rigid_motion: B x N_p x 3
                rigid = rigid_params_to_motion(
                    rigid_params, self.euler_angle_bound, self.N_cum, offset_pos
                )

            else:
                # rigid_params: (B, N_inst, 6)
                rigid_params = self.rigid_predictor(instance_effect_object)

                # rigid_motion: B x N_p x 3
                rigid = rigid_params_to_motion(
                    rigid_params, self.euler_angle_bound, self.N_cum, offset_pos
                )

                # predict physical params
                instance_effect_object = instance_effect_object.view(B * len(self.N_list), -1)
                instance_effect_object, self.lstm_hn_cn = self.lstm(instance_effect_object.unsqueeze(1), self.lstm_hn_cn)
                instance_effect_object = instance_effect_object.squeeze(1).view(B, len(self.N_list), -1)
                physics_params = self.physics_predictor(instance_effect_object) + data.obj_phy_feat     # only predict residual 

                # TODO: Remove ablation study
                # physics_params = torch.zeros((B, 2, 16)).cuda()
                
        else:
            raise NotImplementedError("has_rigid_motion has to be true")

        return rigid, non_rigid, physics_params

    def message(self, x_i, x_j, edge_attr):
        # Computes messages for each edge in edge_index
        edge_messages = torch.cat([edge_attr, x_i, x_j], dim=-1)
        return self.edge_propagator(edge_messages)

    def reset_lstm_state(self):
        self.lstm_hn_cn = None


class PredictorDPINet(MessagePassing):
    def __init__(self, config):
        super().__init__(aggr="add")

        # Configurations
        self.config = config
        self.N_list = config["n_points"]
        self.N_cum = np.cumsum([0] + self.N_list)
        self.F_t = 1
        self.F_v = config["feature_dim_vision"]
        self.F_a = config["feature_dim_action"]
        # self.F_tac = config["tactile_feat_dim"]
        self.P = 3
        self.H = config["history_length"]
        assert self.H == 1, "history length must be 1 for this recurrent GNN"
        self.has_rigid_motion = config["has_rigid_motion"]
        self.euler_angle_bound = config["euler_angle_bound"]
        self.rigid_dim = config["rigid_dim"]
        self.tactile_dim = config["ae_enc_dim"]
        self.obj_phy_feat_len = config["dpi_net"]["obj_phy_feat_len"]
        self.history_before_rigid = config["dpi_net"]["history_before_rigid"]

        self.p_step = config["dpi_net"]["propgation_step"]
        hidden_size_node = config["dpi_net"]["hidden_size_node"]
        hidden_size_edge = config["dpi_net"]["hidden_size_edge"]
        hidden_size_effect = config["dpi_net"]["hidden_size_effect"]
        n_layers = config["dpi_net"]["n_layers"]

        # NodeEncoder
        node_feature_dim = self.H * (self.F_a + self.P) + self.obj_phy_feat_len

        self.node_encoder = MLP(
            [node_feature_dim, hidden_size_node, hidden_size_effect],
            n_layers=n_layers,
            last_relu=True,
        )

        # EdgeEncoder
        self.edge_encoder = MLP(
            [2 * node_feature_dim + 3 + 1, hidden_size_edge, hidden_size_effect],
            n_layers=n_layers,
            last_relu=True,
        )

        # Propagators for nodes and edges
        self.node_propagator = nn.Sequential(
            nn.Linear(hidden_size_effect * 2, hidden_size_effect),
            nn.ReLU(),
        )
        self.edge_propagator = nn.Sequential(
            nn.Linear(hidden_size_effect * 3, hidden_size_effect),
            nn.ReLU(),
        )

        # Prediction heads
        # rigid: (euler, translation)
        self.rigid_predictor = MLP(
            [hidden_size_effect, hidden_size_effect, self.rigid_dim], n_layers=n_layers
        )

        # non_rigid motion
        self.non_rigid_predictor = MLP(
            [hidden_size_effect, hidden_size_effect, self.P], n_layers=n_layers
        )

        # self.physics_predictor = MLP(
        #     [hidden_size_effect, hidden_size_effect, self.obj_phy_feat_len], n_layers=n_layers
        # )

        # # memory module
        # self.lstm = nn.LSTM(hidden_size_effect, hidden_size_effect, 2, batch_first=True)
        # self.lstm_hn_cn = None

    def forward(self, data):
        # Extract the number of graphs and nodes
        B = data.num_graphs
        N = data.num_nodes // B

        # Compute offsets of every point from its object center
        # basically normalize objects points with respect to each its center
        pos_expand = data.pos.view(B, N, self.H, self.P)
        offsets = []
        obj_indices = self.N_cum.tolist() + [N]
        for i in range(len(obj_indices) - 1):
            object_points = pos_expand[:, obj_indices[i]: obj_indices[i + 1]]
            object_center = torch.mean(object_points, dim=1, keepdim=True)
            offset_pos = object_points - object_center
            offsets.append(offset_pos)
        offset_pos = torch.cat(offsets, dim=1)

        # node_input: (B * N, particle_input_dim)
        obj_phy_point_feat = data.obj_phy_feat.view(B, len(self.N_list), self.obj_phy_feat_len)
        physics_params = []
        for i in range(len(self.N_list)):
            physics_params.append(obj_phy_point_feat[:, i, :].unsqueeze(1).repeat(1, self.N_list[i], 1))
        physics_params.append(
            torch.zeros(B, N - sum(self.N_list), self.obj_phy_feat_len, device=physics_params[0].device))
        physics_params = torch.cat(physics_params, dim=1)

        node_phy_feat = physics_params.view(-1, self.obj_phy_feat_len)
        node_input = torch.cat((data.x, offset_pos.view(-1, self.H * self.P), node_phy_feat), dim=-1)

        # node_encode: (B * N, hidden_size_effect)
        node_encode = self.node_encoder(node_input)

        # Process edge inputs
        # sender/receiver_input: (B * R, particle_input_dim)
        # edge_attr: (B * R, 3 + 1)
        receiver_input = node_input[data.edge_index[0]]
        sender_input = node_input[data.edge_index[1]]
        # edge_input: (B * R, 2 * particle_input_dim + 3 + 1)
        edge_input = torch.cat(
            [
                receiver_input,
                sender_input,
                data.edge_attr,
            ],
            dim=-1,
        )
        edge_encode = self.edge_encoder(edge_input)

        # Particle effect after encoding
        node_effect = node_encode

        # stack of GNN layers
        for i in range(self.p_step):
            edge_effect = self.propagate(
                edge_index=data.edge_index, x=node_effect, edge_attr=edge_encode
            )

            node_effect = self.node_propagator(
                torch.cat([node_encode, edge_effect], dim=-1)
            )

        # post-processing, predicting quantities of interest
        # predict non-rigid motions: (B, N_p, D)
        node_effect_object = node_effect.view(B, N, -1)[:, : self.N_cum[-1], :]
        non_rigid = self.non_rigid_predictor(node_effect_object)

        # predict rigid motions
        if self.has_rigid_motion:
            # obtain the instance-level feature for each object: (B, N_inst, node effect size)
            instance_effect_object = torch.cat(
                [
                    torch.mean(
                        node_effect_object[:, self.N_cum[i]: self.N_cum[i + 1]],
                        keepdim=True,
                        dim=1,
                    )
                    for i in range(len(self.N_cum) - 1)
                ],
                dim=1,
            )

            # rigid_params: (B, N_inst, 6)
            rigid_params = self.rigid_predictor(instance_effect_object)

            # rigid_motion: B x N_p x 3
            rigid = rigid_params_to_motion(
                rigid_params, self.euler_angle_bound, self.N_cum, offset_pos
            )

        else:
            raise NotImplementedError("has_rigid_motion has to be true")

        return rigid, non_rigid

    def message(self, x_i, x_j, edge_attr):
        # Computes messages for each edge in edge_index
        edge_messages = torch.cat([edge_attr, x_i, x_j], dim=-1)
        return self.edge_propagator(edge_messages)

    def reset_lstm_state(self):
        self.lstm_hn_cn = None

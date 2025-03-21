"""
PyG implementation of DPI-Net for learning particle dynamics.
This module is core to predicting particle motions in our baselines.
"""

import itertools
import pdb

import numpy as np
import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing

from dynamics.utils_dynamics import MLP, rigid_params_to_motion


class DPINet(MessagePassing):
    def __init__(self, config):
        super(DPINet, self).__init__(aggr="add")

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
        self.has_rigid_motion = config["has_rigid_motion"]
        self.euler_angle_bound = config["euler_angle_bound"]
        self.rigid_dim = config["rigid_dim"]
        self.tactile_dim = config["ae_enc_dim"]

        self.p_step = config["dpi_net"]["propgation_step"]
        hidden_size_node = config["dpi_net"]["hidden_size_node"]
        hidden_size_edge = config["dpi_net"]["hidden_size_edge"]
        hidden_size_effect = config["dpi_net"]["hidden_size_effect"]
        n_layers = config["dpi_net"]["n_layers"]

        # NodeEncoder
        node_feature_dim = self.F_t + self.F_v + self.H * (self.F_a + self.P + self.tactile_dim)

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

        # tactile predictor
        self.tactile_predictor = nn.Sequential(
            MLP(
                [hidden_size_effect, hidden_size_effect, self.tactile_dim], n_layers=n_layers
            ), 
            nn.Tanh()
        )

    # @profile
    def forward(self, data):
        # Extract the number of graphs and nodes
        B = data.num_graphs
        N = data.num_nodes // B

        # Compute offsets
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
        node_input = torch.cat((data.x, offset_pos.view(-1, self.H * self.P)), dim=-1)
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

        # post-processing
        node_effect_object = node_effect.view(B, N, -1)[:, : self.N_cum[-1], :]
        # non_rigid: (B, N_p, D)
        non_rigid = self.non_rigid_predictor(node_effect_object)

        if self.has_rigid_motion:
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

            # rigid_params: (B, N_inst, 6)
            rigid_params = self.rigid_predictor(instance_effect_object)

            # rigid_motion: B x N_p x 3
            rigid = rigid_params_to_motion(
                rigid_params, self.euler_angle_bound, self.N_cum, offset_pos
            )
        else:
            rigid = None

        # predict raw tactile signals
        node_effect_tactile = node_effect.view(B, N, -1)    # [:, self.N_cum[-1]:, :]
        tactile = self.tactile_predictor(node_effect_tactile) 

        return rigid, non_rigid, tactile

    def message(self, x_i, x_j, edge_attr):
        # Computes messages for each edge in edge_index
        edge_messages = torch.cat([edge_attr, x_i, x_j], dim=-1)
        return self.edge_propagator(edge_messages)

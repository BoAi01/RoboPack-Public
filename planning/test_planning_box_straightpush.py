import os
import sys
import scipy
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import numpy as np
import pdb
import torch
import pytorch_lightning as pl
from planning.samplers import CorrelatedNoiseSampler, StraightLinePushSampler
from planning.planner import MPPIOptimizer, EEPositionPlanner
from planning.cost_functions import pointcloud_cost_function
from dynamics.dataset import DynamicsDataModule
from dynamics.config_parser import ConfigParser
from dynamics.models.estimator_predictor_obj_latent_lstm import DynamicsPredictor
from utils.utils import *
from utils.visualizer import *

torch.set_float32_matmul_precision("medium")

# device='cuda'

a_dim = 3
act_feat_len = 3
# box_target_position = [0.5, -0.2, 0]  # np.array([0.4, -0.2, 0])
# box_target_orientation = [0, 0, 60]

box_target_position = [0.6, 0.2, 0]  # np.array([0.4, -0.2, 0])
box_target_orientation = [0, 0, 90]

# np.array([0.5 - 0.10, -0.1 + 0.4, 0])
# np.array([0.5 - 0.20, -0.1 + 0.3, 0])
# np.array([0.5 + 0.15, -0.1 + 0.3, 0])
# np.array([0.5 - 0.05, -0.1 + 0.5, 0])


def test_planning(initial_points):
    pretrained_path = "/svl/u/boai/robopack/dynamics/training/train_tb_logs/v24_estimator_predictor/version_0/checkpoints/epoch=117-step=81538-val_loss=0.00216-val_tac_loss=0.00000.ckpt"
    # pretrained_path = "/home/aibo/intern/robopack/dynamics/training/train_tb_logs/v23_estimator_predictor/version_6/checkpoints/epoch=52-step=26182-val_loss=0.00175-val_tac_loss=0.00000.ckpt"
    model = DynamicsPredictor.load_from_checkpoint(pretrained_path)
    device = torch.device("cuda:0")
    model = model.to_device(device)
    config = ConfigParser(dict())
    config.update_from_json("../dynamics/model_configs/estimator_predictor_tac_boxes.json")
    folder_name_prefix = "v23_idx8_0.6m_his2_translation_repro2"
    assert config["test_batch_size"] == 1, "test batch size must be 1"
    
    data_module = DynamicsDataModule(config)
    data_module.prepare_data()
    data_module.setup('test')
    dataloader = data_module.test_dataloader()

    # test model
    # get a batch of data
    data_iter = iter(dataloader)
    data_idx = 0
    for i in range(data_idx):
        next(data_iter)
    
    batch = next(data_iter).to(device)
    B = batch.num_graphs
    N = batch.num_nodes // B
    seq_len = batch.pos.view(B, N, -1).shape[2] // a_dim
    
    observed_his_len = 2
    
    node_pos = batch.pos.view(B, N, seq_len, -1)
    tac_feat_bubbles = model.autoencoder.encode_structured(batch)
    node_feature = batch.x.view(B, N, -1)
    action_hist = node_feature[:, :, model.type_feat_len + model.vis_feat_len :].view(B, N, seq_len - 1, -1)
    
    # get the history needed
    object_type_feat = node_feature[..., :model.type_feat_len] # should be zero
    vision_feature = torch.zeros(B, N, model.vis_feat_len).to(object_type_feat.device)
    
    node_pos = node_pos[:, :, :observed_his_len, :]
    tac_feat_bubbles = tac_feat_bubbles[:, :observed_his_len, :, :]
    tactile_feat_objects = torch.zeros(B, tac_feat_bubbles.shape[1],
                                            model.n_object_points,
                                            model.config["ae_enc_dim"]).to(model.device)
    tactile_feat_all_points = torch.cat([tactile_feat_objects, tac_feat_bubbles], dim=2)
    action_hist = action_hist[:, :, :(observed_his_len - 1), :]
    
    # # set history to zero
    # node_pos = node_pos[:, :, -1:]
    # tac_feat_bubbles = tac_feat_bubbles[:, -1:]
    # tactile_feat_all_points = tactile_feat_all_points[:, -1:]
    # action_hist = action_hist[:, :, :0]
    
    # action_idx_offset = model.type_feat_len + model.vis_feat_len
    # num_hist = model.his_len
    # num_object_points = model.n_object_points
    
    horizon = 80
    
    # build a planner
    # first build a sampler
    # sampler = CorrelatedNoiseSampler(a_dim=a_dim, beta=0.6, horizon=horizon, num_repeat=10)
    sampler = StraightLinePushSampler(horizon=1, push_distance=0.6, action_size=0.005)
    
    # then build a planner
    # planner = MPPIOptimizer(
    #     sampler=sampler,
    #     model=model,
    #     objective=pointcloud_cost_function,
    #     a_dim=a_dim,
    #     horizon=horizon,
    #     num_samples=200,
    #     gamma=100,      # the larger the more exploitation
    #     num_iters=5,
    #     init_std=np.array([0.0143777, 0.02258945, 0.000]) 
    # )
    planner = EEPositionPlanner(sampler, model, objective=pointcloud_cost_function, horizon=1, num_samples=200, 
                                gamma=100, num_iters=2, log_every=1, theta_std=np.pi/3)
    
    # obtain the goal
    box_points = load_h5_data('../asset/box3_seq_1.h5')['object_pcs'][0][0]   # (N, 3)
    from perception.utils_pc import center_and_rotate_to, farthest_point_sampling_dgl
    goal = center_and_rotate_to(box_points, box_target_position, box_target_orientation)
    goal = farthest_point_sampling_dgl(goal, 20)

    # then plan
    best_actions = planner.plan(
        t=1,
        log_dir="./{}_planning_log_test_sample_{}".format(folder_name_prefix, data_idx),
        observation_batch=(object_type_feat,
                            vision_feature,
                            node_pos, 
                            tac_feat_bubbles,
                            tactile_feat_all_points),
        action_history=action_hist,
        goal=goal,
        visualize_top_k=2
    )

    return best_actions


if __name__ == '__main__':
    # from utils_general import load_dictionary_from_hdf5
    # data_dict = load_dictionary_from_hdf5("/svl/u/boai/robopack/data/v19_1003_parsed_anno/validation/box3/seq_1.h5")
    # dict_keys(['bubble_pcs', 'forces', 'inhand_object_pcs', 'object_cls', 'object_pcs'])
    # import pdb; pdb.set_trace()
    # timestep = 100
    # initial_points = dict(
    #     real_box_points=data_dict["object_pcs"][timestep, 0, :20, :3],
    #     real_rod_points=data_dict["inhand_object_pcs"][timestep, :20, :3],
    #     real_bubble_points=data_dict["bubble_pcs"][timestep][:, :20, :3].reshape(40, -1)
    # )
    initial_points = dict(
        real_box_points=None,
        real_rod_points=None,
        real_bubble_points=None
    )
    test_planning(initial_points)
    
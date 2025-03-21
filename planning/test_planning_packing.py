import os
import sys
import scipy
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import numpy as np
import pdb
import torch
import pytorch_lightning as pl
from planning.samplers import PackingStraightLinePushSampler
from planning.planner import PackingPushingPlanner
from planning.cost_functions import packing_cost_function_inhand_v1, packing_cost_function_inhand_v2, packing_cost_function_rodtip, packing_cost_function_object
from dynamics.dataset import DynamicsDataModule
from dynamics.config_parser import ConfigParser
from dynamics.models.estimator_predictor_obj_latent_lstm_multi import DynamicsPredictor
from utils.utils import *
from utils.visualizer import *

torch.set_float32_matmul_precision("medium")

a_dim = 3
act_feat_len = 3


def test_planning(initial_points):
    # pretrained_path = "/svl/u/boai/robopack/dynamics/training/train_tb_logs/v2_0127_packing_v1_anno/version_4/checkpoints/epoch=32-step=20163-val_loss=0.00021-val_tac_loss=0.00000.ckpt"
    pretrained_path = "/svl/u/boai/robopack/dynamics/training/train_tb_logs/v2_0127_packing_v1_anno/version_5/checkpoints/epoch=36-step=22607-val_loss=0.00020-val_tac_loss=0.00000.ckpt"
    # pretrained_path = "/svl/u/boai/robopack/dynamics/training/train_tb_logs/v2_0127_packing_v1_anno/version_3/checkpoints/epoch=29-step=20430-val_loss=0.00012-val_tac_loss=0.00000.ckpt"
    # pretrained_path = "/svl/u/boai/robopack/dynamics/training/train_tb_logs/v2_0127_packing_v1_anno/version_6/checkpoints/epoch=52-step=36093-val_loss=0.00012-val_tac_loss=0.00000.ckpt"
    model = DynamicsPredictor.load_from_checkpoint(pretrained_path)
    device = torch.device("cuda:0")
    model = model.to_device(device)
    config = ConfigParser(dict())
    config.update_from_json("../dynamics/model_configs/estimator_predictor_notac_packing.json")
    folder_name_prefix = "v2_packing_obj_model5_e36"
    assert config["test_batch_size"] == 1, "test batch size must be 1"
    
    data_module = DynamicsDataModule(config)
    data_module.prepare_data()
    data_module.setup('test')
    dataloader = data_module.test_dataloader()
    
    # test model
    # get a batch of data
    data_iter = iter(dataloader)
    # for v1 packing data: not pushable: 70，40  pushable: 90
    # for v2: pushable: 147, 155, 160, 165    not pushable: 34, 70， (90, 105), 199, 220, 258, 283
    data_idx = 155    
    for i in range(data_idx):
        next(data_iter)
    
    batch = next(data_iter).to(device)
    B = batch.num_graphs
    N = batch.num_nodes // B
    seq_len = batch.pos.view(B, N, -1).shape[2] // a_dim
    
    observed_his_len = 15
    
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
    
    # extract target point cloud from current observation
    # infer the goal point cloud 
    inhand_object_pc, table_object_pcs = PackingPushingPlanner.extract_object_pcs(PackingPushingPlanner.remove_zero_rows(node_pos[0, :, 0]))    # the third axis is time. Use the first frame
    goal_points = PackingPushingPlanner.infer_bounding_box_from_pc(table_object_pcs)

    # build a planner
    # first build a sampler
    sampler = PackingStraightLinePushSampler(num_actions=80)
    
    # then build a planner
    import multiprocessing
    from planner import VisualizationLoggingThread
    logging_queue = multiprocessing.Queue()
    logging_process = VisualizationLoggingThread(
        parameters_queue=logging_queue,
    )
    logging_process.start()
    
    planner = PackingPushingPlanner(sampler, model, objective=packing_cost_function_object, horizon=1, num_samples=200, log_every=1, logging_thread=logging_queue)

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
        visualize_top_k=2,
        goal_points=goal_points
    )

    return best_actions


if __name__ == '__main__':
    initial_points = dict(
        real_box_points=None,
        real_rod_points=None,
        real_bubble_points=None
    )
    test_planning(initial_points)
    
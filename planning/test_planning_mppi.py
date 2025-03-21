import os
import sys

from planning.cost_functions import pointcloud_cost_function

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from planning.samplers import CorrelatedNoiseSampler
from planning.planner import MPPIOptimizer
from dynamics.dataset import DynamicsDataModule
from dynamics.config_parser import ConfigParser
from dynamics.models.dynamics_obj_latent_lstm import DynamicsPredictor
from utils.utils import *
from utils.visualizer import *

torch.set_float32_matmul_precision("medium")

# device='cuda'

a_dim = 3
act_feat_len = 3
# box_target_position = np.array([0.5 - 0.1, -0.1 + 0.3, 0]) # np.array([0.35, 0., 0])       # the origin is (0.5, 0)
box_target_position = [0.4 + 0.2, 0, 0]  # np.array([0.4, -0.2, 0])
box_target_orientation = [0, 0, 45]
# np.array([0.5 - 0.10, -0.1 + 0.4, 0])
# np.array([0.5 - 0.20, -0.1 + 0.3, 0])
# np.array([0.5 + 0.15, -0.1 + 0.3, 0])
# np.array([0.5 - 0.05, -0.1 + 0.5, 0])


def test_planning(initial_points):
    # pretrained_path = "/home/aibo/intern/robopack/dynamics/training/train_tb_logs/v18_0922_parsed_v3_anno/version_0/checkpoints/epoch=49-step=30650-val_loss=0.00613-val_tac_loss=0.00621.ckpt"
    # pretrained_path = "/home/aibo/intern/robopack/dynamics/training/train_tb_logs/v18_0922_parsed_v3_anno/version_1/checkpoints/epoch=82-step=48721-val_loss=0.00330-val_tac_loss=0.00367.ckpt"
    # pretrained_path = "/home/aibo/intern/robopack/dynamics/training/train_tb_logs/v18_0922_parsed_v3_anno/version_3/checkpoints/epoch=116-step=71721-val_loss=0.00618-val_tac_loss=0.00651.ckpt"
    pretrained_path = "/home/aibo/intern/robopack/dynamics/training/train_tb_logs/v18_0922_parsed_v4_negative/version_32/checkpoints/epoch=97-step=25774-val_loss=0.00207-val_tac_loss=0.00330.ckpt"
    model = DynamicsPredictor.load_from_checkpoint(pretrained_path)
    # model.config.device = torch.device("cuda:1")
    device = model.config.device
    model = model.to(device)
    config = ConfigParser(dict())
    config.update_from_json("../dynamics/model_configs/v13_study/v13_latent_lstm_tactile_after.json")
    assert config["test_batch_size"] == 1, "test batch size must be 1"
    
    data_module = DynamicsDataModule(config)
    data_module.prepare_data()
    data_module.setup('test')
    dataloader = data_module.test_dataloader()

    # test model
    # get a batch of data
    data_iter = iter(dataloader)
    for i in range(60):
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
    
    # action_idx_offset = model.type_feat_len + model.vis_feat_len
    # num_hist = model.his_len
    # num_object_points = model.n_object_points
    
    horizon = 30
    
    # build a planner
    # first build a sampler
    sampler = CorrelatedNoiseSampler(a_dim=a_dim, beta=0.3, horizon=horizon, num_repeat=5)
    
    # then build a planner
    planner = MPPIOptimizer(
        sampler=sampler,
        model=model,
        objective=pointcloud_cost_function,
        a_dim=a_dim,
        horizon=horizon,
        num_samples=100,
        gamma=100,      # the larger the more exploitation
        num_iters=10,
        init_std= np.array([0.0143777, 0.02258945, 0.000]) 
    )

    # obtain the goal
    box_points = load_h5_data('../asset/box3_seq_1.h5')['object_pcs'][0][0]   # (N, 3)
    from perception.utils_pc import center_and_rotate_to, farthest_point_sampling_dgl
    goal = center_and_rotate_to(box_points, box_target_position, box_target_orientation)
    goal = farthest_point_sampling_dgl(goal, 20)

    # # then plan
    best_actions = planner.plan(
        t=1,
        log_dir="./planning_log_test",
        observation_batch=(object_type_feat, 
                            vision_feature,
                            node_pos, 
                            tac_feat_bubbles,
                            tactile_feat_all_points),
        action_history=action_hist,
        goal=goal,
        visualize_top_k=1
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
    # test_planning_v0(initial_points)
    test_planning(initial_points)
    
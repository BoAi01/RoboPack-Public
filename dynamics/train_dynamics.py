import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
# os.environ['CUDA_VISIBLE_DEVICES'] = "2"

import importlib
import collections

import pytorch_lightning as pl
from pytorch_lightning.callbacks import *
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from tqdm import tqdm

from dynamics.config_parser import ConfigParser
from dynamics.dataset import DynamicsDataModule
from dynamics.models.autoencoder import AutoEncoder

from utils.utils import *
from utils.visualizer import *
from perception.utils_cv import concatenate_images_side_by_side


torch.set_float32_matmul_precision("medium")


tb_log_dir = 'training/train_tb_logs/v7_inner0.1_outer0.1_p20_lr5e-4_rigid_split/version_0'


DynamicsPredictor = None 
def import_dynamics_class(module_name, class_name='DynamicsPredictor'):
    global DynamicsPredictor
    module_path = f'models.{module_name}'
    module = importlib.import_module(module_path)
    DynamicsPredictor = getattr(module, class_name)
    print(f'imported {class_name} from {module_path}: {DynamicsPredictor}')


def train(config, stats, save_dir):
    if config['pretrained_path']:      # not null string
        model = DynamicsPredictor.load_from_checkpoint(config['pretrained_path'], map_location='cuda:0')
        # model.__init__(config, stats)       # reset config
        print(f'Training resumes from: {config["pretrained_path"]}')
    else:
        model = DynamicsPredictor(config, stats)
    
    # very ad-hoc way of loading AE checkpoint
    # to override the autoencoder weights contained in the DynamicsPredictor weights, which might be the wrong one 
    # For example, when we initialize the model with a previous checkpoint, the old auto-encoder will also be loaded
    # then we need to override it by loading the AE with the desired weight manually. 
    ae_checkpoint = config["ae_checkpoint"] # "/svl/u/boai/robopack/dynamics/pretrained_ae/v24_5to5_epoch=101-step=70482_corrected.ckpt"
    model.autoencoder = AutoEncoder.load_from_checkpoint(frozen=True, checkpoint_path=ae_checkpoint, map_location=config.device)
    
    # put model to the right device
    model = model.to_device(config.device)

    data_module = DynamicsDataModule(config)

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=2, 
                                          filename='{epoch}-{step}-{val_loss:.5f}-{val_tac_loss:.5f}')
    # periodic_checkpoint_callback = ModelCheckpoint(every_n_epochs=20)

    # Create a CSV Logger
    csv_logger = CSVLogger(os.path.join(save_dir, "train_csv_logs"), name="")

    # Create a TensorBoard Logger
    tensorboard_logger = TensorBoardLogger(
        os.path.join(save_dir, "train_tb_logs"), name=config["log_name"]
    )
    global tb_log_dir
    tb_log_dir = tensorboard_logger.log_dir
    print(f'tensorboard logging director: {tb_log_dir}')

    # wandb_logger = WandbLogger(project="RoboPack", name=tb_log_dir.split('/')[-2] + '/' + tb_log_dir.split('/')[-1])

    # train model
    trainer = pl.Trainer(
        max_epochs=config["optimizer"]["max_epoch"],
        accelerator="gpu",
        devices=-1,
        callbacks=[
            TQDMProgressBar(refresh_rate=20),
            EarlyStopping(monitor="val_loss", mode="min", patience=50),  
            # this should be enough. If val loss is low but prediction is bad, 
            # that would mean your loss function is too easy (seq_len?)
            checkpoint_callback,
            # periodic_checkpoint_callback
        ],
        default_root_dir=save_dir,
        logger=[tensorboard_logger, csv_logger],
        num_sanity_val_steps=0,
        deterministic=True,
        # limit_val_batches=0
    )

    trainer.fit(model=model, datamodule=data_module)

    # subprocess.run(
    #     [
    #         "tensorboard",
    #         "--logdir",
    #         os.path.join(save_dir, "tb_logs"),
    #     ]
    # )

    return checkpoint_callback.best_model_path


def test(config, save_dir, best_model_path, stats):
    model = DynamicsPredictor.load_from_checkpoint(best_model_path, map_location='cuda:0')
    
    # # very ad-hoc way of loading AE checkpoint
    # # to override the autoencoder weights contained in the DynamicsPredictor weights, which might be the wrong one 
    # ae_checkpoint = "/svl/u/boai/robopack/dynamics/pretrained_ae/v24_5to5_epoch=101-step=70482.ckpt"
    # model.autoencoder = AutoEncoder.load_from_checkpoint(input_dim=config["tactile_raw_dim"],
    #                                                         encoding_dim=config["ae_enc_dim"],
    #                                                         config=config, stats=stats,
    #                                                         frozen=True,
    #                                                         checkpoint_path=ae_checkpoint,
    #                                                         map_location=config.device)
    # breakpoint()
    
    model = model.to_device(config.device)  # put model to the right device
    data_module = DynamicsDataModule(config)
    
    # in case we want to use a different name for visualizations/statistics for a specific run 
    saving_suffix = ""

    # Create a CSV Logger
    stats_folder_name = f"testing_stats{saving_suffix}"
    csv_logger = CSVLogger(tb_log_dir, name="", version=stats_folder_name)
    tensorboard_logger = TensorBoardLogger(
        tb_log_dir, name="", version=stats_folder_name
    )
    # wandb_logger = WandbLogger(project="RoboPack")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        default_root_dir=save_dir,
        logger=[csv_logger, tensorboard_logger],
        # log_every_n_steps=1,
        num_sanity_val_steps=0,
        deterministic=True,
        # gradient_clip_val=0.5, gradient_clip_algorithm="value"
    )

    trainer.test(model, data_module)

    anim_path = Path(f"{tb_log_dir}/visualizations{saving_suffix}")    # Path(os.path.join(save_dir, "anim"))
    anim_path.mkdir(parents=True, exist_ok=True)

    # assert len(model.pred_state_seqs) == len(model.pred_tac_seqs)
    for i in tqdm(range(len(model.pred_state_seqs))):
        # if i == 41 or i == 42:
        #     suffix = "packing"
        #     pred_pos_seq = model.pred_state_seqs[i]
        #     gt_pos_seq = data_module.test.datasets[0].gt_state_seqs[i]
        #     folder_name = "dynamics_qualitative_vis_notac_packing"
        #     os.makedirs(folder_name, exist_ok=True)
        #     np.save(f'{folder_name}/pred_dict_{i}_{suffix}.npy', pred_pos_seq, allow_pickle=True)
        #     np.save(f'{folder_name}/gt_dict_{i}_{suffix}.npy', gt_pos_seq, allow_pickle=True)
        #     breakpoint()
        # else:
        #     continue

        # if data_module.test is of type ConcatDataset, then it will have
        # an attribute called datasets, which is a list of dataset.Subset
        gt_pos_seq = data_module.test.datasets[0].gt_state_seqs[i]

        # [(20, 6), (20, 6), (40, 6)] -- [(20, 6), (20, 6), (40, 9), (2, 7), (40, 2), (2,)]
        if "state_estimator_obj_latent_lstm" in config["module_name"]:
            pred_pos_seq = model.pred_state_seqs[i]
            # pred_tac_seq = model.pred_tac_seqs[i]
            gt_tac_seq = model.gt_tac_seqs[i]
            
            # tac_frames = visualize_pred_tactile(
            #     gt_tac_seq,
            #     gt_tac_seq,       # pred_tac_seq,
            #     path=anim_path / f"tac_test_{str(i).zfill(3)}"
            # )
            
            vis_frame = visualize_pred_gt_pos_simple(
                config=config["visualizer"],
                title_list=["Prediction", "Ground Truth"],
                pred_gt_pos_seqs=[pred_pos_seq, gt_pos_seq],
                path=None, # anim_path / f"test_{str(i).zfill(3)}",
                num_skip_frames=config["history_length"], # max(config["history_length"], config["recurr_T"]),
                multiview=True
            )
            
            # for j in range(len(vis_frame) - len(tac_frames)):
            #     tac_frames += [Image.fromarray(np.zeros_like(np.array(tac_frames[0])))]
            
            # combined_frames = [concatenate_images_side_by_side(x, y) for x, y in zip(vis_frame, tac_frames)]
            combined_frames = vis_frame
            play_and_save_video(combined_frames, f"{str(anim_path)}/vis_tac_{i}.mp4", 2)
            
            print(f'visualization saved to {anim_path}')
        
        else:
            pred_pos_seq = model.pred_state_seqs[i]
            
            vis_frame = visualize_pred_gt_pos_simple(
                config=config["visualizer"],
                title_list=["Prediction", "Ground Truth"],
                pred_gt_pos_seqs=[pred_pos_seq, gt_pos_seq],
                path= anim_path / f"test_{str(i).zfill(3)}",
                num_skip_frames=config["history_length"], # max(config["history_length"], config["recurr_T"]),
                multiview=True
            )
            
            print(f'visualization saved to {anim_path}')
        
    # save_tuples_to_textfile(model.error_seq, ["last-step mse", "avg tac loss"], anim_path / f"test_error.txt")
    if config["module_name"] != "dynamics_ae":
        np.save(f'{tb_log_dir}/physics_params{saving_suffix}.npy', model.pred_physics_list, allow_pickle=True)
        # np.save(f'{tb_log_dir}/physics_params.npy', [{'data': x.cpu().detach().tolist(), 'label': y} for x, y in model.pred_physics_list], allow_pickle=True)
        np.save(f'{tb_log_dir}/lstm_states{saving_suffix}.npy', model.lstm_states, allow_pickle=True)
        print(f'physics params saved to {tb_log_dir}/physics_params{saving_suffix}.npy')


def save_tuples_to_textfile(data_list, column_names, filename):
    try:
        with open(filename, 'w') as file:
            # Write column names to the first line
            file.write(f"{column_names[0]}, {column_names[1]}\n")

            # Write each tuple with an index as a line in the file
            for index, (x, y) in enumerate(data_list):
                file.write(f"{index}, {x}, {y}\n")

        print(f"Data saved to {filename}")
    except Exception as e:
        print(f"Error: {e}")


import numpy as np
from scipy.interpolate import RegularGridInterpolator

def resize_image(original_image, H, W):
    h, w, _ = original_image.shape

    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, h - 1, h)
    new_x = np.linspace(0, w - 1, W)
    new_y = np.linspace(0, h - 1, H)

    interpolators = [RegularGridInterpolator((y, x), original_image[..., channel]) for channel in range(2)]

    new_image = np.zeros((H, W, 2), dtype=original_image.dtype)
    new_coords = np.array(np.meshgrid(new_y, new_x)).T.reshape(-1, 2)

    for channel in range(2):
        new_values = interpolators[channel](new_coords)
        new_image[..., channel] = new_values.reshape((H, W))

    return new_image


def visualize_pred_tactile(gt_tac, pred_tac, path):
    from perception.utils_cv import visualize_raw_flow, force_arrow_visualize
    pred_frames, gt_frames = [], [] 
    frames = []
    # import pdb; pdb.set_trace()
    for gt, pred in zip(gt_tac, pred_tac):
        flow_pred = pred['flows'][0, 0].reshape(4, 5, 2)
        flow_pred = resize_image(flow_pred, 200, 250)
        flow_pred_vis = visualize_raw_flow(flow_pred, scale=5)
        # import pdb; pdb.set_trace()
        arrow_pred = pred['forces'][0, 0, 0]
        flow_pred_vis = force_arrow_visualize(flow_pred_vis, arrow_pred, None)

        flow_gt = gt['flows'][0, 0].reshape(4, 5, 2)
        flow_gt = resize_image(flow_gt, 200, 250)
        flow_gt_vis = visualize_raw_flow(flow_gt, scale=5) 
        arrow_gt = gt['forces'][0, 0, 0]
        flow_gt_vis = force_arrow_visualize(flow_gt_vis, arrow_gt, None)
        
        # plt.imsave('pred_tac.png', flow_pred_vis / 255)
        # plt.imsave('pred_gt.png', flow_gt_vis / 255)
    
        # pred_frames.append(Image.fromarray(flow_pred_vis))
        # gt_frames.append(Image.fromarray(flow_gt_vis))

        # print(abs(flow_pred - flow_gt).mean())
        
        frames.append(concatenate_images_side_by_side(flow_pred_vis.astype(np.uint8), flow_gt_vis.astype(np.uint8)))

    # play_and_save_video(frames, str(path) + '.mp4', 2)
    return frames


def main():
    parser = argparse.ArgumentParser(description="dynamics")
    parser.add_argument(
        "-c",
        "--config",
        default=os.path.join(DYNAMICS_DIR, "dynamics_config.json"),
        type=str,
        help="config file path (default: dynamics_config.json)",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="0",
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    
    # parser.add_argument(
    #     "-d",
    #     "--prefix",
    #     default=None,
    #     type=str,
    #     help="indices of GPUs to enable (default: all)",
    # )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        # CustomArgs(["--exp_name"], type=str, target="exp_name"),
        # CustomArgs(["--dpi_layers"], type=int, target="dpi_net.n_layers"),
        # CustomArgs(["--gnn_type"], type=str, target="gnn_type"),
        # CustomArgs(["--loss_by_n_points"], type=int, target="loss_by_n_points"),
        # CustomArgs(["--pt_layers"], type=int, target="point_transformer.n_layers"),
        # CustomArgs(["--p_step"], type=int, target="dpi_net.propgation_step"),
    ]
    config = ConfigParser.from_dynamics_args(parser, options=options)

    import_dynamics_class(config["module_name"])

    # configure logging module
    logging.getLogger("lightning.pytorch").setLevel(config["logging_level"])

    pl.seed_everything(0)

    stats = load_h5_data(os.path.join(DATA_DIR, config["data_dir_prefix"], "train", "stats.h5"))

    save_dir = 'training'
    assert os.path.exists(save_dir), f"path does not exist: {save_dir}"

    if config["test_only"]:
        best_model_path = config["pretrained_path"]
        global tb_log_dir
        tb_log_dir = os.path.dirname(os.path.dirname(best_model_path))
        print(f'Testing only mode. Model path: {best_model_path} \n\tSetting tb log path to: {tb_log_dir}')
    else:
        best_model_path = train(config, stats, save_dir)
        print(f'Training completed. Best model path: {best_model_path}')

    # save_dir = os.path.join(*Path(best_model_path).parts[:-4])
    # config.update_from_json(
    #     os.path.join(
    #         save_dir, "code", f"dynamics_obs_config_{os.path.basename(save_dir)}.json"
    #     )
    # )

    test(config, save_dir, best_model_path, stats)


# pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
if __name__ == "__main__":
    main()

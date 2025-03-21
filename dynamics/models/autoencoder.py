"""
Model implementation of the encoder for tactile signals.
"""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import pdb
import warnings

import torch
import numpy as np 
import torch.nn as nn
import torch.utils.data as data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import *
import cv2

from torch.optim.lr_scheduler import ReduceLROnPlateau
from perception.utils_cv import visualize_raw_flow

from perception.utils_cv import visualize_raw_flow, force_arrow_visualize
# from train import resize_image
import matplotlib.pyplot as plt
from utils.visualizer import from_ax_to_pil_img, play_and_save_video

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


class AutoEncoder(pl.LightningModule):
    def __init__(self, input_dim, encoding_dim, stats, config):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim),
            # nn.ReLU(),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Tanh()
        )
        self.loss_function = nn.MSELoss()

        self.num_bubbles = 2
        self.n_object_points = sum(config["n_points"])      # number of points except tool, use 120 for packing
        # self.n_object_points = 120

        self.stats = stats
        self.set_statistics(config.device)

        # for inference
        self.warning_count = 0
        self.total_count = 0
        self.recon_error_tolerance = 0.01  # error will throw if loss > this value
        
        # for sanity check
        self.particles_per_obj = config["particles_per_obj"]
        
        # IMPORTANT: it should save the parameters when checkpoint was saved
        # this can minimize the chance of mistakenly modifying the params, such as stats, which can incur hard-to-spot bugs. 
        self.save_hyperparameters()
        
        # to log testing
        self.test_losses = [] 
        
    def set_statistics(self, device):
        self.force_mean = torch.tensor(self.stats["force_mean"], dtype=torch.float32, device=device)
        self.force_scale = torch.tensor(self.stats["force_scale"], dtype=torch.float32, device=device)
        self.flow_mean = torch.tensor(self.stats["flow_mean"], dtype=torch.float32, device=device)
        self.flow_scale = torch.tensor(self.stats["flow_scale"], dtype=torch.float32, device=device)
        # self.pressure_mean = torch.tensor(self.stats["pressure_mean"], dtype=torch.float32, device=device)
        # self.pressure_scale = torch.tensor(self.stats["pressure_scale"], dtype=torch.float32, device=device)

    def preprocess_data(self, data, recover_dim=False, n_object_points=None):
        B = data.num_graphs
        N = data.num_nodes // B
        # forces, flows, pressure = data.forces, data.flows, data.pressure
        forces, flows = data.forces, data.flows

        if n_object_points is None:
            n_bubble_points = (N - self.n_object_points) // 2       # self.n_object_points is the value from checkpoint which might not be accurate
        else:
            n_bubble_points = (N - n_object_points) // 2
        assert n_bubble_points == self.particles_per_obj, \
            f"the bubble should have {self.particles_per_obj} points, got {n_bubble_points} instead"

        # -1 means seq length, e.g., (his_len + seq_len) during training
        # but at test time the value might differ
        forces = forces.view(B, -1, self.num_bubbles, 7)

        flows = flows.view(B, -1, self.num_bubbles, n_bubble_points, 2)
        flows = (flows - self.flow_mean) / self.flow_scale
        flows = flows.clip(-2, 2) / 2

        # pressure = pressure.view(B, -1, self.num_bubbles)
        # pressure = (pressure - self.pressure_mean) / self.pressure_scale
        # pressure = pressure.clip(-2, 2) / 2

        # Step 1.2: Some processing to unify the shape
        forces = forces.unsqueeze(3).repeat(1, 1, 1, n_bubble_points, 1)
        forces = torch.cat([forces[..., :2], forces[..., -1:]], dim=-1)  # get the three nonzero values
        forces = (forces - self.force_mean) / self.force_scale
        forces = forces.clip(-2, 2) / 2

        # pressure = pressure.unsqueeze(3).unsqueeze(-1).repeat(1, 1, 1, n_bubble_points, 1)

        # tactile_raw_bubbles = torch.cat([forces, flows, pressure], dim=-1)  # the last dim is 6
        tactile_raw_bubbles = torch.cat([forces, flows], dim=-1)  # the last dim is 5

        if recover_dim:
            tactile_raw_bubbles = tactile_raw_bubbles.view(B, -1,
                                                           self.num_bubbles * n_bubble_points, self.input_dim)

        return tactile_raw_bubbles

    def forward(self, x):        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode_structured(self, batch, n_object_points=None):
        """
        Extract the tactile reading from a dynamics model data object.
        The method makes assumption on the data structure.
        It returns the reshaped features to the dynamics model.
        :param batch: Data in the format supported by the dynamics model
        :return: The extracted latent features of the tactile readings
        """
        self.total_count += 1
        data = self.preprocess_data(batch, recover_dim=True, n_object_points=n_object_points)
        # ST: Not sure if this is correct
        data = data.float()
        
        # # computing the temporal difference, instead of using the absolute values
        # data = data[:, 1:] - data[:, :-1]
        # data = torch.cat([torch.zeros_like(data[:, :1]), data], dim=1)
        
        # return data 
        
        encoded = self.encoder(data)
        decoded = self.decoder(encoded)
        loss = self.loss_function(data, decoded)
        
        # check if the reconstruction error is too large
        if loss.item() > self.recon_error_tolerance:
            self.warning_count += 1
            if self.warning_count / self.total_count > 0.2:
                warnings.warn(f"Reconstruction loss of AE is larger than {self.recon_error_tolerance} for more than 0.2 of the times "
                              f"(N = {self.total_count}). Current loss = = {loss.item():.3f}")
                # warnings.warn(f"Current loss = {loss.item():.3f}")
        
        # print("Loss of AE: ", loss.item())
        # print("warning counts", self.warning_count / self.total_count)
        
        return encoded

    def training_step(self, batch, batch_idx):
        # data = batch[0].flatten(1)
        data = self.preprocess_data(batch)
        reconstructions = self(data)
        loss = self.loss_function(reconstructions, data)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data = self.preprocess_data(batch)
        reconstructions = self(data)
        loss = self.loss_function(reconstructions, data)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        # raise NotImplementedError()
        data = self.preprocess_data(batch, recover_dim=True)
        
        # # computing the temporal difference, instead of using the absolute values
        # data = data[:, 1:] - data[:, :-1]
        # data = torch.cat([torch.zeros_like(data[:, :1]), data], dim=1)
        
        reconstructions = self(data)
        loss = self.loss_function(reconstructions, data)
        # self.log('test_loss', loss, prog_bar=True, on_epoch=True)
        self.test_losses.append(loss.item())

        # save encoding
        encoded = self.encode_structured(batch)
        # data, reconstructions = encoded, encoded
        # np.save(f'tactile_encoding_{batch_idx}.npy', reconstructions.cpu().detach().numpy())

        # compute loss for every sample 
        losses = torch.nn.functional.mse_loss(data, reconstructions, reduction='none')
        losses = losses.mean(dim=-1).mean(dim=-1)[0]        # shaped (T)
        # print(f'The loss is {self.loss_function(reconstructions[..., :3], data[..., :3]).item():.3f}')
        # print(f'The loss is {self.loss_function(reconstructions[..., 3:5], data[..., 3:5]).item():.3f}')
        # print(f'The loss is {self.loss_function(reconstructions[..., 5:], data[..., 5:]).item():.3f}')

        # reconstructions[:, :, :] = torch.tensor([0.3978, -0.1538, 0.4585, 0.0455, -0.2205, 0.3283], device='cuda:0')
        # data[:, :, :] = torch.tensor([ 0.1523, -0.9452,  0.1965,  0.0667,  0.1823,  0.7319], device='cuda:0')

        # visualize
        gt_flows = data[0, ..., 3:5] * 2 * self.flow_scale + self.flow_mean
        pred_flows = reconstructions[0, ..., 3:5] * 2 * self.flow_scale + self.flow_mean
        gt_forces = data[0, ..., :3] * 2 * self.force_scale + self.force_mean
        pred_forces = reconstructions[0, ..., :3] * 2 * self.force_scale + self.force_mean
        # gt_forces[..., :2] *= 200  # adjust the scale for visualization
        # pred_forces[..., :2] *= 200
        # gt_flows *= 2
        # pred_flows *= 2
        
        frames = []
        for i, (gt_flow, pred_flow, gt_force, pred_force) in enumerate(zip(gt_flows, pred_flows, gt_forces, pred_forces)):
            flow_gt = resize_image(gt_flow[:20].cpu().numpy().reshape(4, 5, 2), 200, 250)
            flow_pred = resize_image(pred_flow[:20].cpu().numpy().reshape(4, 5, 2), 200, 250)
            force_gt = gt_force[0].cpu().numpy()
            force_pred = pred_force[0].cpu().numpy()
            
            flow_gt_vis = visualize_raw_flow(flow_gt, scale=5) 
            flow_gt_vis = force_arrow_visualize(flow_gt_vis, force_gt, None, scale=5e2) / 255
            
            flow_pred_vis = visualize_raw_flow(flow_pred, scale=5) 
            flow_pred_vis = force_arrow_visualize(flow_pred_vis, force_pred, None, scale=5e2) / 255

            # Create a subplot with 1 row and 2 columns
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            # Display the ground truth image with its subtitle
            axes[0].imshow(flow_gt_vis)  # Convert BGR to RGB for correct colors
            axes[0].set_title("Ground Truth")
            axes[0].axis('off')

            # Display the predicted image with its subtitle
            axes[1].imshow(flow_pred_vis)
            axes[1].set_title(f"Reconstructed (loss: {losses[i].item():.5f})")
            axes[1].axis('off')

            # Adjust layout and display the plot
            plt.tight_layout()
            # plt.show()

            frames.append(from_ax_to_pil_img(fig))

        play_and_save_video(frames, f'./ae_vis_test_v24_box4/ae_{batch_idx}.mp4', 1)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.scheduler = ReduceLROnPlateau(self.optimizer, "min", factor=0.8, patience=3, verbose=True)
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler,
            "monitor": "train_loss",
        }


if __name__ == '__main__':

    # # Simulated data
    # train_data = torch.rand(1000, 32)
    # val_data = torch.rand(200, 32)
    #
    # # Create DataLoader instances for train and validation data
    # train_loader = data.DataLoader(train_data, batch_size=64, shuffle=True)
    # val_loader = data.DataLoader(val_data, batch_size=64)

    from torch.utils.data import DataLoader, random_split
    from torchvision import transforms
    from torchvision.datasets import MNIST

    # MNIST data loading and preprocessing
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = MNIST(root='/media/albert/ExternalHDD/scratch/datasets', train=True, transform=transform,
                        download=True)
    mnist_val = MNIST(root='/media/albert/ExternalHDD/scratch/datasets', train=False, transform=transform)

    # Create DataLoader instances for train and validation data
    train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
    val_loader = DataLoader(mnist_val, batch_size=64)

    # Instantiate the AutoEncoder model
    in_dim = 28 * 28
    enc_dim = 10
    autoencoder = AutoEncoder(input_dim=in_dim, encoding_dim=enc_dim)

    # Initialize the Lightning Trainer
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
    trainer = pl.Trainer(max_epochs=10,
                         callbacks=[
                             TQDMProgressBar(refresh_rate=20),
                             EarlyStopping(monitor="val_loss", mode="min", patience=2),
                             checkpoint_callback
                         ]
                         )

    # Train the model and save checkpoints
    trainer.fit(autoencoder, train_loader, val_loader)

    # Load a checkpoint for verification
    # checkpoint = pl.load_checkpoint(checkpoint_callback.best_model_path)
    best_model_path = checkpoint_callback.best_model_path
    print(f'best model path = {best_model_path}')
    loaded_model = AutoEncoder.load_from_checkpoint(input_dim=in_dim, encoding_dim=enc_dim,
                                                    checkpoint_path=best_model_path)

    # Perform inference with the loaded model (optional)
    import matplotlib.pyplot as plt

    # Take some images from the validation set
    num_images_to_visualize = 5
    sample_images, _ = next(iter(val_loader))
    sample_images = sample_images[:num_images_to_visualize].view(num_images_to_visualize, -1)

    # Encode and decode the sample images using the trained autoencoder
    reconstructed_images = autoencoder(sample_images)
    # Calculate reconstruction errors for the test images
    reconstruction_errors = torch.mean((reconstructed_images - sample_images) ** 2)
    print(f'reconstruct errors: {reconstruction_errors}')

    reconstructed_images = reconstructed_images.view(num_images_to_visualize, 1, 28, 28)

    # Plot the original and reconstructed images
    fig, axes = plt.subplots(nrows=2, ncols=num_images_to_visualize, figsize=(12, 5))

    for i in range(num_images_to_visualize):
        axes[0, i].imshow(sample_images[i].view(28, 28), cmap='gray')
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')

        axes[1, i].imshow(reconstructed_images[i].view(28, 28).detach().numpy(), cmap='gray')
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

    print("Checkpoint loaded and model verified.")


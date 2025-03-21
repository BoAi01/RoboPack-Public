import itertools
import pdb

import numpy as np
import torch
import torch.nn as nn
from pytorch3d.transforms import quaternion_to_matrix


# MLP with the option of relu activation and layer normalization
class MLP(nn.Module):
    def __init__(
            self,
            layer_sizes,
            n_layers,
            last_relu=False,
            layer_norm=False,
    ):
        input_size, hidden_size, output_size = layer_sizes

        super(MLP, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            *itertools.chain.from_iterable(
                (nn.Linear(hidden_size, hidden_size), nn.ReLU())
                for i in range(n_layers - 2)
            ),
            nn.Linear(hidden_size, output_size),
        )

        if last_relu:
            self.layers.append(nn.ReLU())

        if layer_norm:
            self.layers.append(nn.LayerNorm(output_size))

    def forward(self, x):
        return self.layers(x)


def rigid_params_to_motion(rigid_params, euler_angle_bound, N_cum, offset_pos, translation_bound=0.05):
    B = rigid_params.shape[0]

    # euler angle representation for rotation, which works better in practice
    assert rigid_params.shape[-1] == 6, f'rigid_params should be of shape (..., 6), but got {rigid_params.shape}'
    rotation_params = torch.tanh(rigid_params[..., :3]) * (euler_angle_bound / 180 * np.pi)
    rotation_matrix = euler_to_rot_matrix(rotation_params.view(-1, 3)).view(B, -1, 3, 3)
    
    # use quaternion to as the rotation representation 
    # the way we have been using 
    # assert rigid_params.shape[-1] == 7, f'rigid_params should be of shape (..., 7), but got {rigid_params.shape}'
    # quat_offset = torch.tensor([1, 0, 0, 0]).to(rigid_params.device)
    # rotation_matrix = quaternion_to_matrix(rigid_params[..., :4] + quat_offset)
    # translation_vector = rigid_params[..., -3:].view(B, -1, 1, 3)
    
    # the actually more correct way of doing quaternion, which has NOT been used for deployment 
    # quat_offset = torch.tensor([1, 0, 0, 0]).to(rigid_params.device)
    # rigid_params[..., 0][rigid_params[..., 0] < 0] *= -1  # set scalar component to positive
    # rotation_quat = rigid_params[..., :4] + quat_offset     # add residual 
    # rotation_quat = rotation_quat / rotation_quat.norm(dim=-1, keepdim=True)
    # rotation_matrix = quaternion_to_matrix(rotation_quat)
    
    # bound translation so that the model can better predict small values 
    translation_params = torch.tanh(rigid_params[..., -3:]) * translation_bound
    translation_vector = translation_params.view(B, -1, 1, 3)
    # translation_vector = rigid_params[..., -3:].view(B, -1, 1, 3)
    
    # print(is_rotation_matrix(rotation_matrix.view(-1, 3, 3)))
    rigid_list = []
    for i in range(len(N_cum) - 1):
        instance_pos_prev = offset_pos[:, N_cum[i]: N_cum[i + 1], -1]
        instance_pos = (
                torch.bmm(instance_pos_prev, rotation_matrix[:, i])
                + translation_vector[:, i]
        )
        rigid_list.append(instance_pos - instance_pos_prev)

    # rigid_motion: B x N_p x 3
    rigid = torch.cat(rigid_list, dim=1)

    return rigid


def is_rotation_matrix(R):
    """
    Checks if a matrix is a valid rotation matrix.
    """
    Rt = torch.transpose(R, 1, 2)
    shouldBeIdentity = torch.matmul(Rt, R)
    I = torch.eye(3).expand_as(shouldBeIdentity).to(R.device)
    maxError = torch.max(torch.abs(shouldBeIdentity - I))

    return torch.allclose(R.det(), torch.tensor(1.0).to(R.device)) and maxError < 1e-6


def euler_to_rot_matrix(euler_angles):
    B = euler_angles.shape[0]

    # Roll, pitch and yaw
    roll, pitch, yaw = euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]

    cos_roll = torch.cos(roll)
    sin_roll = torch.sin(roll)
    cos_pitch = torch.cos(pitch)
    sin_pitch = torch.sin(pitch)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    # Create the rotation matrix
    R = torch.zeros((B, 3, 3), device=euler_angles.device)

    # Populate rotation matrix
    R[:, 0, 0] = cos_yaw * cos_pitch
    R[:, 0, 1] = cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll
    R[:, 0, 2] = cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll
    R[:, 1, 0] = sin_yaw * cos_pitch
    R[:, 1, 1] = sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll
    R[:, 1, 2] = sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll
    R[:, 2, 0] = -sin_pitch
    R[:, 2, 1] = cos_pitch * sin_roll
    R[:, 2, 2] = cos_pitch * cos_roll

    return R

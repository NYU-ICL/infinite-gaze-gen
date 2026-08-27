import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FinePositionalEncoding2d(nn.Module):
    def __init__(self, d_model: int, height: int, width: int):
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4")

        pe = torch.zeros(d_model, height, width)
        d_half = d_model // 2
        div_term = torch.exp(torch.arange(0.0, d_half, 2) * -(math.log(10000.0) / d_half))
        pos_w = torch.arange(0.0, width).unsqueeze(1)
        pos_h = torch.arange(0.0, height).unsqueeze(1)

        pe[0:d_half:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_half:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_half::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_half + 1 :: 2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        self.register_buffer("pe2d", pe.unsqueeze(0))

    def get_pos_enc_for_traj(self, traj: torch.Tensor) -> torch.Tensor:
        height, width = self.pe2d.shape[2], self.pe2d.shape[3]
        traj_indices = traj.long().clone()
        traj_indices[:, 0, :] = traj_indices[:, 0, :].clamp(0, width - 1)
        traj_indices[:, 1, :] = traj_indices[:, 1, :].clamp(0, height - 1)

        batch = traj_indices.shape[0]
        pe_grid = self.pe2d.expand(batch, -1, -1, -1)
        flattened = pe_grid.view(batch, pe_grid.shape[1], -1)
        flat_indices = traj_indices[:, 1, :] * width + traj_indices[:, 0, :]
        flat_indices = flat_indices.unsqueeze(1).expand(-1, pe_grid.shape[1], -1)
        return torch.gather(flattened, dim=2, index=flat_indices)

    def get_pos_enc_for_patch(self, patch_height: int, patch_width: int) -> torch.Tensor:
        return F.interpolate(self.pe2d, size=(patch_height, patch_width), mode="bilinear", align_corners=False)

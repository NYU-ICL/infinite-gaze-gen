import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pos_enc import FinePositionalEncoding2d


def get_time_embedding(time_steps: torch.Tensor, temb_dim: int) -> torch.Tensor:
    if temb_dim % 2 != 0:
        raise ValueError("time_emb_dim must be divisible by 2")
    factor = 10000 ** (
        torch.arange(0, temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2)
    )
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    return torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)


class CustomAttention(nn.Module):
    def __init__(self, target_dim: int, source_dim: int, embed_size: int, num_heads: int, attention_type: str):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.attention_type = attention_type
        self.query = nn.Linear(target_dim, embed_size * num_heads)
        self.key = nn.Linear(source_dim, embed_size * num_heads)
        self.value = nn.Linear(source_dim, embed_size * num_heads)
        self.fc_out = nn.Linear(embed_size * num_heads, target_dim)

    def forward(self, target: torch.Tensor, source: torch.Tensor | None = None):
        if self.attention_type == "self":
            source = target
        elif source is None:
            raise ValueError("Cross-attention requires source conditioning.")

        q = self.query(target)
        k = self.key(source)
        v = self.value(source)

        q = q.view(q.size(0), -1, self.num_heads, self.embed_size).permute(0, 2, 1, 3)
        k = k.view(k.size(0), -1, self.num_heads, self.embed_size).permute(0, 2, 1, 3)
        v = v.view(v.size(0), -1, self.num_heads, self.embed_size).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.embed_size)
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)
        bsz, _, seq_len, _ = out.shape
        out = out.permute(0, 2, 1, 3).reshape(bsz, seq_len, self.num_heads * self.embed_size)
        return self.fc_out(out), weights


class CustomAttentionWithResidual(nn.Module):
    def __init__(self, target_dim: int, source_dim: int, embed_size: int, num_heads: int, attention_type: str):
        super().__init__()
        self.attention = CustomAttention(target_dim, source_dim, embed_size, num_heads, attention_type)
        self.norm = nn.LayerNorm(target_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, target: torch.Tensor, source: torch.Tensor | None = None):
        out, weights = self.attention(target, source)
        out = self.norm(target + self.dropout(out))
        return out, weights


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        t_emb_dim: int,
        down_sample: bool,
        num_heads: int,
        num_layers: int,
        crss_attn_thru_unet: bool,
        num_context_dim: int,
        attn_embed_size: int,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.crss_attn_thru_unet = crss_attn_thru_unet
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv1d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, padding=1),
                )
                for i in range(num_layers)
            ]
        )
        self.t_emb_layers = nn.ModuleList(
            [nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, out_channels)) for _ in range(num_layers)]
        )
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                )
                for _ in range(num_layers)
            ]
        )
        self.self_attentions = nn.ModuleList(
            [
                CustomAttentionWithResidual(out_channels, out_channels, attn_embed_size, num_heads, "self")
                for _ in range(num_layers)
            ]
        )
        self.cross_attentions = nn.ModuleList(
            [
                CustomAttentionWithResidual(out_channels, num_context_dim, attn_embed_size, num_heads, "cross")
                for _ in range(num_layers)
            ]
        )
        self.residual_input_conv = nn.ModuleList(
            [nn.Conv1d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1) for i in range(num_layers)]
        )
        self.down_sample_conv = nn.Conv1d(out_channels, out_channels, kernel_size=4, stride=2, padding=1) if down_sample else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, conditioning: torch.Tensor):
        out = x
        for i in range(self.num_layers):
            residual = out
            out = self.resnet_conv_first[i](out)
            out = out + self.t_emb_layers[i](t_emb)[:, :, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](residual)
            out, _ = self.self_attentions[i](out.permute(0, 2, 1))
            out = out.permute(0, 2, 1)
            if self.crss_attn_thru_unet:
                out, _ = self.cross_attentions[i](out.permute(0, 2, 1), conditioning)
                out = out.permute(0, 2, 1)
        return self.down_sample_conv(out)


class MidBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        t_emb_dim: int,
        num_heads: int,
        num_layers: int,
        crss_attn_thru_unet: bool,
        num_context_dim: int,
        attn_embed_size: int,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.crss_attn_thru_unet = crss_attn_thru_unet
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv1d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, padding=1),
                )
                for i in range(num_layers + 1)
            ]
        )
        self.t_emb_layers = nn.ModuleList(
            [nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, out_channels)) for _ in range(num_layers + 1)]
        )
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                )
                for _ in range(num_layers + 1)
            ]
        )
        self.self_attentions = nn.ModuleList(
            [
                CustomAttentionWithResidual(out_channels, out_channels, attn_embed_size, num_heads, "self")
                for _ in range(num_layers)
            ]
        )
        self.cross_attentions = nn.ModuleList(
            [
                CustomAttentionWithResidual(out_channels, num_context_dim, attn_embed_size, num_heads, "cross")
                for _ in range(num_layers)
            ]
        )
        self.residual_input_conv = nn.ModuleList(
            [nn.Conv1d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1) for i in range(num_layers + 1)]
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, conditioning: torch.Tensor):
        out = x
        residual = out
        out = self.resnet_conv_first[0](out)
        out = out + self.t_emb_layers[0](t_emb)[:, :, None]
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](residual)

        for i in range(self.num_layers):
            out, _ = self.self_attentions[i](out.permute(0, 2, 1))
            out = out.permute(0, 2, 1)
            if self.crss_attn_thru_unet:
                out, _ = self.cross_attentions[i](out.permute(0, 2, 1), conditioning)
                out = out.permute(0, 2, 1)
            residual = out
            out = self.resnet_conv_first[i + 1](out)
            out = out + self.t_emb_layers[i + 1](t_emb)[:, :, None]
            out = self.resnet_conv_second[i + 1](out)
            out = out + self.residual_input_conv[i + 1](residual)
        return out


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        t_emb_dim: int,
        up_sample: bool,
        num_heads: int,
        num_layers: int,
        crss_attn_thru_unet: bool,
        num_context_dim: int,
        attn_embed_size: int,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.crss_attn_thru_unet = crss_attn_thru_unet
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv1d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, padding=1),
                )
                for i in range(num_layers)
            ]
        )
        self.t_emb_layers = nn.ModuleList(
            [nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, out_channels)) for _ in range(num_layers)]
        )
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                )
                for _ in range(num_layers)
            ]
        )
        self.self_attentions = nn.ModuleList(
            [
                CustomAttentionWithResidual(out_channels, out_channels, attn_embed_size, num_heads, "self")
                for _ in range(num_layers)
            ]
        )
        self.cross_attentions = nn.ModuleList(
            [
                CustomAttentionWithResidual(out_channels, num_context_dim, attn_embed_size, num_heads, "cross")
                for _ in range(num_layers)
            ]
        )
        self.residual_input_conv = nn.ModuleList(
            [nn.Conv1d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1) for i in range(num_layers)]
        )
        self.up_sample_conv = (
            nn.ConvTranspose1d(in_channels // 2, in_channels // 2, kernel_size=4, stride=2, padding=1)
            if up_sample
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor, conditioning: torch.Tensor):
        x = self.up_sample_conv(x)
        if x.shape[-1] != skip.shape[-1]:
            diff = skip.shape[-1] - x.shape[-1]
            if diff > 0:
                x = F.pad(x, (0, diff))
            else:
                x = x[:, :, : skip.shape[-1]]
        out = torch.cat([x, skip], dim=1)
        for i in range(self.num_layers):
            residual = out
            out = self.resnet_conv_first[i](out)
            out = out + self.t_emb_layers[i](t_emb)[:, :, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](residual)
            out, _ = self.self_attentions[i](out.permute(0, 2, 1))
            out = out.permute(0, 2, 1)
            if self.crss_attn_thru_unet:
                out, _ = self.cross_attentions[i](out.permute(0, 2, 1), conditioning)
                out = out.permute(0, 2, 1)
        return out


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, embed_size: int, num_positions: int):
        super().__init__()
        self.pos_encoding = nn.Parameter(torch.randn(1, num_positions, embed_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_encoding


class Unet(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        seq_len: int = 45,
        down_channels: list[int] | None = None,
        mid_channels: list[int] | None = None,
        time_emb_dim: int = 128,
        down_sample: list[bool] | None = None,
        num_down_layers: int = 2,
        num_mid_layers: int = 2,
        num_up_layers: int = 2,
        num_self_attn_heads_in_unet: int = 4,
        num_cross_attn_heads_pre_unet: int = 4,
        saliency_patches: int = 24,
        saliency_features: int = 64,
        patches_crss_attn_b4_unet: bool = True,
        patches_crss_attn_thru_unet: bool = True,
        attn_embed_size: int = 64,
        pos_enc_type: str = "sinusoidal_fine",
        should_do_pos_enc: bool = False,
        traj_pos_enc_height: int = 224,
        traj_pos_enc_width: int = 224,
    ):
        super().__init__()
        down_channels = down_channels or [32, 64, 128, 256]
        mid_channels = mid_channels or [256, 256, 128]
        down_sample = down_sample or [True, True, False]

        if mid_channels[0] != down_channels[-1] or mid_channels[-1] != down_channels[-2]:
            raise ValueError("Channel configuration is inconsistent.")
        if len(down_sample) != len(down_channels) - 1:
            raise ValueError("down_sample must have len(down_channels) - 1 entries.")

        self.seq_len = seq_len
        self.saliency_patches = saliency_patches
        self.saliency_features = saliency_features
        self.patches_crss_attn_b4_unet = patches_crss_attn_b4_unet
        self.patches_crss_attn_thru_unet = patches_crss_attn_thru_unet
        self.pos_enc_type = pos_enc_type
        self.should_do_pos_enc = should_do_pos_enc
        self.time_emb_dim = time_emb_dim

        self.t_proj = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.conv_in = nn.Conv1d(in_channels, down_channels[0], kernel_size=3, padding=1)

        context_dim = down_channels[0] if pos_enc_type == "sinusoidal_fine" else saliency_features
        self.downs = nn.ModuleList(
            [
                DownBlock(
                    down_channels[i],
                    down_channels[i + 1],
                    time_emb_dim,
                    down_sample[i],
                    num_self_attn_heads_in_unet,
                    num_down_layers,
                    patches_crss_attn_thru_unet,
                    context_dim,
                    attn_embed_size,
                )
                for i in range(len(down_channels) - 1)
            ]
        )
        self.mids = nn.ModuleList(
            [
                MidBlock(
                    mid_channels[i],
                    mid_channels[i + 1],
                    time_emb_dim,
                    num_self_attn_heads_in_unet,
                    num_mid_layers,
                    patches_crss_attn_thru_unet,
                    context_dim,
                    attn_embed_size,
                )
                for i in range(len(mid_channels) - 1)
            ]
        )
        self.ups = nn.ModuleList(
            [
                UpBlock(
                    down_channels[i] * 2,
                    down_channels[i - 1] if i != 0 else 16,
                    time_emb_dim,
                    down_sample[i],
                    num_self_attn_heads_in_unet,
                    num_up_layers,
                    patches_crss_attn_thru_unet,
                    context_dim,
                    attn_embed_size,
                )
                for i in reversed(range(len(down_channels) - 1))
            ]
        )

        self.norm_out = nn.GroupNorm(8, 16)
        self.conv_out = nn.Conv1d(16, out_channels, kernel_size=3, padding=1)

        if pos_enc_type == "learnable":
            self.saliency_positional_encoder = LearnablePositionalEncoding(saliency_features, saliency_patches)
            self.traj_positional_encoder = LearnablePositionalEncoding(down_channels[0], seq_len)
            self.cross_attn_b4_unet = CustomAttentionWithResidual(
                down_channels[0],
                saliency_features,
                attn_embed_size,
                num_cross_attn_heads_pre_unet,
                "cross",
            )
        elif pos_enc_type == "sinusoidal_fine":
            self.saliency_mlp = nn.Sequential(nn.Linear(saliency_features, down_channels[0]), nn.SiLU())
            self.traj_patch_pos_enc = FinePositionalEncoding2d(
                d_model=down_channels[0],
                height=traj_pos_enc_height,
                width=traj_pos_enc_width,
            )
            self.cross_attn_b4_unet = CustomAttentionWithResidual(
                down_channels[0],
                down_channels[0],
                attn_embed_size,
                num_cross_attn_heads_pre_unet,
                "cross",
            )
        else:
            raise ValueError(f"Unsupported pos_enc_type: {pos_enc_type}")

    def _temporal_pos_emb(self, num_frames: int, dim: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(num_frames, device=device).float()
        half_dim = dim // 2
        div_term = torch.exp(torch.arange(0, half_dim, device=device).float() * (-math.log(10000.0) / max(half_dim, 1)))
        angles = positions[:, None] * div_term[None, :]
        pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if pe.shape[1] < dim:
            pe = torch.cat([pe, torch.zeros((num_frames, dim - pe.shape[1]), device=device)], dim=1)
        return pe[:, :dim]

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        saliency_patches: torch.Tensor,
        history_traj: torch.Tensor | None = None,
        crss_attn_save_params: dict | None = None,
    ):
        del history_traj, crss_attn_save_params
        t_emb = self.t_proj(get_time_embedding(torch.as_tensor(t).long(), self.time_emb_dim))
        out = self.conv_in(x)
        attention_weights = None

        is_temporal = isinstance(saliency_patches, torch.Tensor) and saliency_patches.dim() == 4
        if self.pos_enc_type == "sinusoidal_fine":
            saliency_patches = self.saliency_mlp(saliency_patches)

        if is_temporal:
            _, t_frames, _, feat_dim = saliency_patches.shape
            saliency_patches = saliency_patches + self._temporal_pos_emb(t_frames, feat_dim, saliency_patches.device)[None, :, None, :]

        if self.should_do_pos_enc:
            if self.pos_enc_type == "learnable":
                out = self.traj_positional_encoder(out.permute(0, 2, 1)).permute(0, 2, 1)
                if is_temporal:
                    bsz, t_frames, patches, channels = saliency_patches.shape
                    saliency_patches = saliency_patches.reshape(bsz * t_frames, patches, channels)
                    saliency_patches = self.saliency_positional_encoder(saliency_patches)
                    saliency_patches = saliency_patches.reshape(bsz, t_frames * patches, channels)
                else:
                    saliency_patches = self.saliency_positional_encoder(saliency_patches)
            else:
                out = out + self.traj_patch_pos_enc.get_pos_enc_for_traj(x)
                if is_temporal:
                    bsz, t_frames, patches, channels = saliency_patches.shape
                    grid_h = int(math.sqrt(patches))
                    saliency_patches_2d = einops.rearrange(saliency_patches, "b t (h w) d -> (b t) d h w", h=grid_h, w=grid_h)
                    saliency_patches_2d = saliency_patches_2d + self.traj_patch_pos_enc.get_pos_enc_for_patch(
                        saliency_patches_2d.shape[2], saliency_patches_2d.shape[3]
                    )
                    saliency_patches = einops.rearrange(
                        saliency_patches_2d, "(b t) d h w -> b t (h w) d", b=bsz, t=t_frames
                    )
                    saliency_patches = saliency_patches.reshape(bsz, t_frames * patches, channels)
                else:
                    grid_h = int(math.sqrt(self.saliency_patches))
                    saliency_patches_2d = einops.rearrange(saliency_patches, "b (h w) d -> b d h w", h=grid_h, w=grid_h)
                    saliency_patches_2d = saliency_patches_2d + self.traj_patch_pos_enc.get_pos_enc_for_patch(
                        saliency_patches_2d.shape[2], saliency_patches_2d.shape[3]
                    )
                    saliency_patches = einops.rearrange(saliency_patches_2d, "b d h w -> b (h w) d")

        if isinstance(saliency_patches, torch.Tensor) and saliency_patches.dim() == 4:
            bsz, t_frames, patches, channels = saliency_patches.shape
            saliency_patches = saliency_patches.reshape(bsz, t_frames * patches, channels)

        if self.patches_crss_attn_b4_unet:
            out, attention_weights = self.cross_attn_b4_unet(out.permute(0, 2, 1), saliency_patches)
            out = out.permute(0, 2, 1)

        down_outs = []
        for down in self.downs:
            down_outs.append(out)
            out = down(out, t_emb, saliency_patches)

        for mid in self.mids:
            out = mid(out, t_emb, saliency_patches)

        for up in self.ups:
            out = up(out, down_outs.pop(), t_emb, saliency_patches)

        out = self.conv_out(nn.SiLU()(self.norm_out(out)))
        return out, attention_weights

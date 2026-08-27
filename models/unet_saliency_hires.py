import torch
from torch import Tensor
import torch.nn as nn
import math
import einops
import torch.nn.functional as F
from models.pos_enc import FinePositionalEncoding2d

def get_time_embedding(time_steps, temb_dim):
    r"""
    Convert time steps tensor into an embedding using the
    sinusoidal time embedding formula
    :param time_steps: 1D tensor of length batch size
    :param temb_dim: Dimension of the embedding
    :return: BxD embedding representation of B time steps
    """
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"
    
    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2))
    )
    
    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb

# Positional Encoding - https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding2d(nn.Module):
    def __init__(self, d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                            -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        self.pe_reshaped = einops.arrange(pe, 'd h w -> (h w) d')
        

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        bs = x.size(0)
        pe_reshaped = einops.repeat(pe_reshaped, 'hw d -> bs hw d', bs=bs)
        x = x + self.pe_reshaped.to(x.device)

        pass


# Positional Encoding - https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model) # (num saliency patches, 1, 768) aka (256, 1, 768)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # visualize_saliency_patches_rgb(x[:,0,:], og_img=None)
        visualize_saliency_patches_rgb(x[:,0,:], og_img=None)
        x = x + self.pe[:x.size(0)]
        visualize_saliency_patches_rgb(x[:,0,:], og_img=None)
        return self.dropout(x)

class DownBlock(nn.Module):
    r"""
    Down conv block with attention.
    Sequence of following block
    1. Resnet block with time embedding
    2. Attention block
    3. Downsample using 2x2 average pooling
    """
    def __init__(self, in_channels, out_channels, t_emb_dim,
                 down_sample=True, num_heads=4, num_layers=1, crss_attn_thru_unet=False, num_contex_dim=768, attn_embed_size=64):
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample
        self.crss_attn_thru_unet = crss_attn_thru_unet
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv1d(in_channels if i == 0 else out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for i in range(num_layers)
            ]
        )
        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            )
            for _ in range(num_layers)
        ])
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv1d(out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )
        self.self_attentions = nn.ModuleList(
            [
                CustomAttentionWithResidual(target_dim=out_channels, source_dim=out_channels, embed_size=attn_embed_size, num_heads=num_heads, attention_type="self")
                for _ in range(num_layers)
            ]
        )

        self.cross_attentions = nn.ModuleList(
            [CustomAttentionWithResidual(target_dim=out_channels, source_dim=num_contex_dim, embed_size=attn_embed_size, num_heads=num_heads, attention_type="cross")
                for _ in range(num_layers)]
        )


        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv1d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        self.down_sample_conv = nn.Conv1d(out_channels, out_channels,
                                          4, 2, 1) if self.down_sample else nn.Identity()
    
    def forward(self, x, t_emb, saliency_patches=None, crss_attn_save_params=None, block_idx=None):
        out = x # (bs, channels, seq_len)

        for i in range(self.num_layers):
            # Resnet block of Unet: (bs, channels, seq_len) -> (bs, channels*2, seq_len)
            resnet_input = out # (bs, channels, seq_len)
            out = self.resnet_conv_first[i](out) # (bs, channels, seq_len) -> (bs, channels*2, seq_len)
            out = out + self.t_emb_layers[i](t_emb)[:, :, None] # (bs, channels*2, seq_len) -> (bs, channels*2, seq_len)
            out = self.resnet_conv_second[i](out) # (bs, channels*2, seq_len) -> (bs, channels*2, seq_len)
            out = out + self.residual_input_conv[i](resnet_input) # # (bs, channels*2, seq_len) -> (bs, channels*2, seq_len)

            # Attention block of Unet
            out, attention_weights = self.self_attentions[i](out.permute(0,2,1), out.permute(0,2,1)) # (bs, seq_len, channels*2)
            out = out.permute(0,2,1) # (bs, seq_len, channels*2) -> (bs, channels*2, seq_len)

            # Cross Attention block
            if self.crss_attn_thru_unet:
                out, attention_weights = self.cross_attentions[i](out.permute(0,2,1), saliency_patches) # (bs, seq_len, channels*2)
                out = out.permute(0,2,1) # (bs, seq_len, channels*2) -> (bs, channels*2, seq_len)

        out = self.down_sample_conv(out) # (bs, channels*2, seq_len) -> (bs, channels*2, seq_len/2)
        return out


class MidBlock(nn.Module):
    r"""
    Mid conv block with attention.
    Sequence of following blocks
    1. Resnet block with time embedding
    2. Attention block
    3. Resnet block with time embedding
    """
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads=4, num_layers=1, crss_attn_thru_unet=False, num_contex_dim=768, attn_embed_size=64):
        super().__init__()
        self.num_layers = num_layers
        self.crss_attn_thru_unet = crss_attn_thru_unet
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv1d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1),
                )
                for i in range(num_layers+1)
            ]
        )
        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            )
            for _ in range(num_layers + 1)
        ])
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers+1)
            ]
        )
        
        self.self_attentions = nn.ModuleList(
            [
                CustomAttentionWithResidual(target_dim=out_channels, source_dim=out_channels, embed_size=attn_embed_size, num_heads=num_heads, attention_type="self")
                for _ in range(num_layers)
            ]
        )

        self.cross_attentions = nn.ModuleList(
            [CustomAttentionWithResidual(target_dim=out_channels, source_dim=num_contex_dim, embed_size=attn_embed_size, num_heads=num_heads, attention_type="cross")
                for _ in range(num_layers)]
        )


        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv1d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers+1)
            ]
        )
    
    def forward(self, x, t_emb, saliency_patches=None, crss_attn_save_params=None, block_idx=None):
        out = x # (bs, channels, seq_len)

        # Resnet blocks
        resnet_input = out
        out = self.resnet_conv_first[0](out) # (bs, channels, seq_len) -> (bs, channels, seq_len)
        out = out + self.t_emb_layers[0](t_emb)[:, :, None] # (bs, channels, seq_len) -> (bs, channels, seq_len)
        out = self.resnet_conv_second[0](out) # (bs, channels, seq_len) -> (bs, channels, seq_len)
        out = out + self.residual_input_conv[0](resnet_input) # (bs, channels, seq_len) -> (bs, channels, seq_len)
        
        for i in range(self.num_layers):
            # Attention block
            out, attention_weights = self.self_attentions[i](out.permute(0,2,1), out.permute(0,2,1)) # (bs, seq_len, channels)
            out = out.permute(0,2,1) # (bs, channels, seq_len)
            
            # Cross Attention block
            if self.crss_attn_thru_unet:
                out, attention_weights = self.cross_attentions[i](out.permute(0,2,1), saliency_patches) # (bs, seq_len, channels), (bs, num_heads, seq_len, embed_size)
                out = out.permute(0,2,1) # (bs, seq_len, channels) -> (bs, channels, seq_len)

            # Resnet Block
            resnet_input = out
            out = self.resnet_conv_first[i+1](out) # (bs, channels, seq_len) -> (bs, channels/2, seq_len)
            out = out + self.t_emb_layers[i+1](t_emb)[:, :, None] # (bs, channels/2, seq_len) -> (bs, channels/2, seq_len)
            out = self.resnet_conv_second[i+1](out) # (bs, channels/2, seq_len) -> (bs, channels/2, seq_len)
            out = out + self.residual_input_conv[i+1](resnet_input) # (bs, channels/2, seq_len) -> (bs, channels/2, seq_len)
            
        return out


class UpBlock(nn.Module):
    r"""
    Up conv block with attention.
    Sequence of following blocks
    1. Upsample
    1. Concatenate Down block output
    2. Resnet block with time embedding
    3. Attention Block
    """
    def __init__(self, in_channels, out_channels, t_emb_dim, up_sample=True, num_heads=4, num_layers=1, crss_attn_thru_unet=False, num_contex_dim=768, attn_embed_size=64):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample
        self.crss_attn_thru_unet = crss_attn_thru_unet
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv1d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1),
                )
                for i in range(num_layers)
            ]
        )
        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            )
            for _ in range(num_layers)
        ])
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )
        
        self.self_attentions = nn.ModuleList(
            [
                CustomAttentionWithResidual(target_dim=out_channels, source_dim=out_channels, embed_size=attn_embed_size, num_heads=num_heads, attention_type="self")
                for _ in range(num_layers)
            ]
        )

        self.cross_attentions = nn.ModuleList(
            [CustomAttentionWithResidual(target_dim=out_channels, source_dim=num_contex_dim, embed_size=attn_embed_size, num_heads=num_heads, attention_type="cross")
                for _ in range(num_layers)]
        )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv1d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        self.up_sample_conv = nn.ConvTranspose1d(in_channels // 2, in_channels // 2,
                                                 4, 2, 1) \
            if self.up_sample else nn.Identity()
    
    def forward(self, x, out_down, t_emb, saliency_patches=None, crss_attn_save_params=None, block_idx=None):
        x = self.up_sample_conv(x)
        if x.shape[-1] != out_down.shape[-1]:
            diff = out_down.shape[-1] - x.shape[-1]
            if diff > 0:
                x = F.pad(x, (0, diff))
            else:
                x = x[:, :, :out_down.shape[-1]]
        x = torch.cat([x, out_down], dim=1)
        
        out = x
        for i in range(self.num_layers):
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = out + self.t_emb_layers[i](t_emb)[:, :, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)

            # Attention block of Unet
            out, attention_weights = self.self_attentions[i](out.permute(0,2,1), out.permute(0,2,1))
            out = out.permute(0,2,1)

            if self.crss_attn_thru_unet:
                out, attention_weights = self.cross_attentions[i](out.permute(0,2,1) , saliency_patches) 
                out = out.permute(0,2,1) 

        return out

def cross_attention(Q, K, V, mask=None):
    # Compute the dot products between Q and K, then scale
    d_k = Q.shape[-1]  # embed_size per head
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    # Softmax to normalize scores and get attention weights
    attention_weights = F.softmax(scores, dim=-1)
    # Weighted sum of values
    output = torch.matmul(attention_weights, V)
    bs, num_heads, seq_len, embed_size = output.shape
    output = output.permute(0, 2, 1, 3).reshape(bs, seq_len, num_heads * embed_size)
    return output, attention_weights

class CustomAttention(nn.Module):
    def __init__(self, target_dim, source_dim, embed_size, num_heads=1, attention_type="self"):
        super(CustomAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.attention_type = attention_type
        # Linear layers for Q, K, V
        self.query = nn.Linear(target_dim, embed_size * num_heads)
        self.key = nn.Linear(source_dim, embed_size * num_heads)
        self.value = nn.Linear(source_dim, embed_size * num_heads)
        # Final linear layer after concatenating heads
        self.fc_out = nn.Linear(embed_size * num_heads, target_dim)
    def forward(self, target, source=None, mask=None):
        if self.attention_type == "self":
            Q = self.query(target)
            K = self.key(target)
            V = self.value(target)
        elif self.attention_type == "cross":
            assert source is not None, "Source input required for cross-attention"
            Q = self.query(target)
            K = self.key(source)
            V = self.value(source)
        Q = Q.view(Q.size(0), -1, self.num_heads, self.embed_size).permute(0, 2, 1, 3) # (bs, num_heads, seq_len, embed_size)
        K = K.view(K.size(0), -1, self.num_heads, self.embed_size).permute(0, 2, 1, 3) # (bs, num_heads, seq_len, embed_size)
        V = V.view(V.size(0), -1, self.num_heads, self.embed_size).permute(0, 2, 1, 3) # (bs, num_heads, seq_len, embed_size)
        # Perform attention calculation (self or cross)
        out, attention_weights = cross_attention(Q, K, V, mask)

        return self.fc_out(out), attention_weights
    
class CustomAttentionWithResidual(nn.Module):
    def __init__(self, target_dim, source_dim, embed_size, num_heads=6, attention_type="self"):
        super(CustomAttentionWithResidual, self).__init__()
        self.attention = CustomAttention(target_dim, source_dim, embed_size, num_heads, attention_type)
        self.norm = nn.LayerNorm(target_dim)
        self.dropout = nn.Dropout(0.1)
    def forward(self, target, source=None, mask=None):
        attention_out, attention_weights = self.attention(target, source, mask)
        # Add residual connection and layer normalization
        target = target + self.dropout(attention_out)
        out = self.norm(target)
        return out, attention_weights


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, embed_size, num_patches):
        super(LearnablePositionalEncoding, self).__init__()
        self.pos_encoding = nn.Parameter(torch.randn(1, num_patches, embed_size))

    def forward(self, x):
        return x + self.pos_encoding

class Unet(nn.Module):
    r"""
    Unet model comprising
    Down blocks, Midblocks and Uplocks
    """
    def __init__(
            self, 
            in_channels=2,
            out_channels=2,
            seq_len=720,
            history_len=None,
            pred_len=None,
            data_sample_stride=None,
            down_channels=[32, 64, 128, 256],
            mid_channels=[256, 256, 128],
            time_emb_dim=128,
            down_sample=[True, True, False],
            num_down_layers=2,
            num_mid_layers=2,
            num_up_layers=2,
            num_self_attn_heads_in_unet=4,
            num_cross_attn_heads_pre_unet=4,
            saliency_patches=256,
            saliency_features=768,
            patches_crss_attn_b4_unet=False,
            patches_crss_attn_thru_unet=False,
            attn_embed_size=64,
            pos_enc_type='learnable',
            should_do_pos_enc=True,
            frame_stride=1,
            use_frame_conditioning=True,
            traj_pos_enc_height=224,
            traj_pos_enc_width=224,
            conditioning_grid_height=None,
            conditioning_grid_width=None,
        ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_len = seq_len
        self.history_len = history_len
        self.pred_len = pred_len
        self.data_sample_stride = data_sample_stride
        self.frame_stride = int(frame_stride)
        self.use_frame_conditioning = bool(use_frame_conditioning)
        self.down_channels = down_channels
        self.mid_channels = mid_channels
        self.time_emb_dim = time_emb_dim
        self.down_sample = down_sample
        self.num_down_layers = num_down_layers
        self.num_mid_layers = num_mid_layers
        self.num_up_layers = num_up_layers
        self.num_self_attn_heads_in_unet = num_self_attn_heads_in_unet
        self.num_cross_attn_heads_pre_unet = num_cross_attn_heads_pre_unet
        self.saliency_patches = saliency_patches
        self.saliency_features = saliency_features
        self.patches_crss_attn_b4_unet = patches_crss_attn_b4_unet
        self.patches_crss_attn_thru_unet = patches_crss_attn_thru_unet
        self.pos_enc_type = pos_enc_type
        self.should_do_pos_enc = should_do_pos_enc
        self.traj_pos_enc_height = int(traj_pos_enc_height)
        self.traj_pos_enc_width = int(traj_pos_enc_width)
        self.conditioning_grid_height = None if conditioning_grid_height is None else int(conditioning_grid_height)
        self.conditioning_grid_width = None if conditioning_grid_width is None else int(conditioning_grid_width)
        self.conditioning_grid_hw = self._resolve_conditioning_grid_hw(
            self.saliency_patches,
            self.conditioning_grid_height,
            self.conditioning_grid_width,
        )
        
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-2]
        assert len(self.down_sample) == len(self.down_channels) - 1
        
        # Initial projection from sinusoidal time embedding
        self.t_proj = nn.Sequential(
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim)
        )

        self.up_sample = list(reversed(self.down_sample))
        self.conv_in = nn.Conv1d(self.in_channels, self.down_channels[0], kernel_size=3, padding=(1))
        
        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels)-1):
            if pos_enc_type == 'learnable':
                self.downs.append(DownBlock(self.down_channels[i], self.down_channels[i+1], self.time_emb_dim,
                                            down_sample=self.down_sample[i], num_heads=self.num_self_attn_heads_in_unet, 
                                            num_layers=self.num_down_layers, crss_attn_thru_unet=self.patches_crss_attn_thru_unet, 
                                            num_contex_dim=self.saliency_features, attn_embed_size=attn_embed_size))
            elif pos_enc_type == 'sinusoidal_fine':
                self.downs.append(DownBlock(self.down_channels[i], self.down_channels[i+1], self.time_emb_dim,
                                            down_sample=self.down_sample[i], num_heads=self.num_self_attn_heads_in_unet, 
                                            num_layers=self.num_down_layers, crss_attn_thru_unet=self.patches_crss_attn_thru_unet, 
                                            num_contex_dim=self.down_channels[0], attn_embed_size=attn_embed_size))
        
        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels)-1):
            if pos_enc_type == 'learnable':
                self.mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i+1], self.time_emb_dim,
                                        num_heads=self.num_self_attn_heads_in_unet, num_layers=self.num_mid_layers,
                                        crss_attn_thru_unet=self.patches_crss_attn_thru_unet, num_contex_dim=self.saliency_features, attn_embed_size=attn_embed_size))
            elif pos_enc_type == 'sinusoidal_fine':
                self.mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i+1], self.time_emb_dim,
                                        num_heads=self.num_self_attn_heads_in_unet, num_layers=self.num_mid_layers,
                                        crss_attn_thru_unet=self.patches_crss_attn_thru_unet, num_contex_dim=self.down_channels[0], attn_embed_size=attn_embed_size))
        
        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels)-1)):
            if pos_enc_type == 'learnable':
                self.ups.append(UpBlock(self.down_channels[i] * 2, self.down_channels[i-1] if i != 0 else 16,
                                        self.time_emb_dim, up_sample=self.down_sample[i], 
                                        num_heads=self.num_self_attn_heads_in_unet, num_layers=self.num_up_layers,
                                        crss_attn_thru_unet=self.patches_crss_attn_thru_unet, num_contex_dim=self.saliency_features, attn_embed_size=attn_embed_size))
            elif pos_enc_type == 'sinusoidal_fine':
                self.ups.append(UpBlock(self.down_channels[i] * 2, self.down_channels[i-1] if i != 0 else 16,
                                        self.time_emb_dim, up_sample=self.down_sample[i], 
                                        num_heads=self.num_self_attn_heads_in_unet, num_layers=self.num_up_layers,
                                        crss_attn_thru_unet=self.patches_crss_attn_thru_unet, num_contex_dim=self.down_channels[0], attn_embed_size=attn_embed_size))
        
        self.norm_out = nn.GroupNorm(8, 16)
        self.conv_out = nn.Conv1d(16, self.out_channels, kernel_size=3, padding=1)


        if pos_enc_type == 'learnable':
            self.image_conditioning_feature_dim = int(self.saliency_features)
            self.saliency_positional_encoder = LearnablePositionalEncoding(embed_size=self.saliency_features, num_patches=self.saliency_patches)
            self.traj_positional_encoder = LearnablePositionalEncoding(embed_size=self.down_channels[0], num_patches=self.seq_len)
            self.cross_attn_b4_unet = CustomAttentionWithResidual(target_dim=self.down_channels[0], source_dim=saliency_features, embed_size=attn_embed_size, 
                                                                num_heads=self.num_cross_attn_heads_pre_unet, attention_type="cross")
            self.history_proj = nn.Conv1d(2, self.saliency_features, kernel_size=1)
        
        elif pos_enc_type == 'sinusoidal_fine':
            self.image_conditioning_feature_dim = int(self.down_channels[0])
            self.saliency_mlp = nn.Sequential(
                nn.Linear(self.saliency_features, self.down_channels[0]),
                nn.SiLU(),
            )
            self.traj_patch_pos_enc = FinePositionalEncoding2d(
                d_model=self.down_channels[0],
                height=self.traj_pos_enc_height,
                width=self.traj_pos_enc_width,
            )
            self.cross_attn_b4_unet = CustomAttentionWithResidual(target_dim=self.down_channels[0], source_dim=self.down_channels[0], embed_size=attn_embed_size, 
                                                                num_heads=self.num_cross_attn_heads_pre_unet, attention_type="cross")         
            self.history_proj = nn.Conv1d(2, self.down_channels[0], kernel_size=1)

    def _resolve_conditioning_grid_hw(self, token_count: int, grid_h_override=None, grid_w_override=None):
        if grid_h_override is not None or grid_w_override is not None:
            if grid_h_override is None or grid_w_override is None:
                raise ValueError("conditioning_grid_height and conditioning_grid_width must be set together")
            grid_h = max(1, int(grid_h_override))
            grid_w = max(1, int(grid_w_override))
            expected_tokens = int(grid_h * grid_w)
            if int(token_count) != expected_tokens:
                raise ValueError(
                    f"saliency_patches ({token_count}) must equal conditioning_grid_height * conditioning_grid_width "
                    f"({grid_h} * {grid_w} = {expected_tokens})"
                )
            return grid_h, grid_w
        token_count = max(1, int(token_count))
        grid_h = int(math.floor(math.sqrt(token_count)))
        grid_h = max(1, grid_h)
        while token_count % grid_h != 0 and grid_h > 1:
            grid_h -= 1
        grid_w = int(math.ceil(float(token_count) / float(grid_h)))
        return int(grid_h), int(grid_w)

    def _encode_image_conditioning(self, saliency_patches: torch.Tensor):
        if saliency_patches.dim() not in (4, 5):
            return saliency_patches, False
        if saliency_patches.dim() == 4:
            if saliency_patches.shape[1] != 3 or saliency_patches.shape[2] <= 8 or saliency_patches.shape[3] <= 8:
                return saliency_patches, False
            b, _c, _h, _w = saliency_patches.shape
            feat = self.image_conditioning_stem(saliency_patches.to(torch.float32))
            feat = F.adaptive_avg_pool2d(feat, self.conditioning_grid_hw)
            feat = einops.rearrange(feat, 'b d h w -> b (h w) d')
            return feat, True
        if saliency_patches.shape[2] != 3:
            return saliency_patches, False
        b, t_frames, _c, _h, _w = saliency_patches.shape
        feat = self.image_conditioning_stem(saliency_patches.reshape(b * t_frames, 3, _h, _w).to(torch.float32))
        feat = F.adaptive_avg_pool2d(feat, self.conditioning_grid_hw)
        feat = einops.rearrange(feat, '(b t) d h w -> b t (h w) d', b=b, t=t_frames)
        return feat, True

    def _temporal_pos_emb(self, num_frames: int, dim: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(num_frames, device=device).float()
        half_dim = dim // 2
        if half_dim == 0:
            return torch.zeros((num_frames, dim), device=device)
        div_term = torch.exp(torch.arange(0, half_dim, device=device).float() * (-math.log(10000.0) / half_dim))
        angles = positions[:, None] * div_term[None, :]
        pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if dim % 2 == 1:
            pe = torch.cat([pe, torch.zeros((num_frames, 1), device=device)], dim=1)
        return pe

    def forward(self, x, t, saliency_patches, history_traj=None, crss_attn_save_params=None):
        # x shape: (bs, num_feats, seq_len) -> (bs, 2, 720) Note: 2 represents x,y coordinates
        # t shape: (bs) -> represents the level of noise added to the trajectory. should be between 0 and 1000
        # saliency_patches shape: (bs, num_saliency_patches, num_saliency_feats) -> (bs, 256, 768) Note: 256 patches of 16x16 pixels

        # gets the time embedding and passes it through some linear/non-linear layers
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.time_emb_dim) # (bs) -> (bs, 128)
        t_emb = self.t_proj(t_emb) # (bs, 128) -> (bs, 128)

        # embeds the trajectory
        out = self.conv_in(x) # (bs, 2, 720) -> (bs, 32, 720)
        attention_weights=None

        saliency_patches, is_image_conditioning = self._encode_image_conditioning(saliency_patches)
        is_temporal = isinstance(saliency_patches, torch.Tensor) and saliency_patches.dim() == 4
        if isinstance(saliency_patches, torch.Tensor) and not self.use_frame_conditioning:
            if is_temporal:
                saliency_patches = torch.zeros_like(saliency_patches[:, 0, :, :])
            else:
                saliency_patches = torch.zeros_like(saliency_patches)
            is_temporal = False
        if is_temporal:
            _, t_frames, _, feat_dim = saliency_patches.shape
            temp_pe = self._temporal_pos_emb(t_frames, feat_dim, saliency_patches.device)
            saliency_patches = saliency_patches + temp_pe[None, :, None, :]

        if not is_image_conditioning:
            saliency_patches = self.saliency_mlp(saliency_patches) # (bs, 256, 768) -> (bs, 256, 32)
        if self.should_do_pos_enc:
            if self.pos_enc_type=='learnable':
                out = self.traj_positional_encoder(out.permute(0,2,1)).permute(0,2,1) # (bs, 32, 720)
                if is_temporal:
                    b, t_frames, p, c = saliency_patches.shape
                    saliency_patches = saliency_patches.reshape(b * t_frames, p, c)
                    saliency_patches = self.saliency_positional_encoder(saliency_patches) # (bs*t, 256, 32)
                    saliency_patches = saliency_patches.reshape(b, t_frames * p, c)
                else:
                    saliency_patches = self.saliency_positional_encoder(saliency_patches) # (bs, 256, 768)
            elif self.pos_enc_type=='sinusoidal_fine':
                # add positional embedding to trajectory
                traj_pes = self.traj_patch_pos_enc.get_pos_enc_for_traj(x)
                out = out + traj_pes

                # add positional embedding to saliency patches
                if is_temporal:
                    b, t_frames, p, c = saliency_patches.shape
                    h = int(self.saliency_patches**0.5)
                    saliency_patches_2d = einops.rearrange(saliency_patches, 'b t (h w) d -> (b t) d h w', h=h, w=h)
                    saliency_patches_pes = self.traj_patch_pos_enc.get_pos_enc_for_patch(saliency_patches_2d.shape[2], saliency_patches_2d.shape[3])
                    saliency_patches_2d = saliency_patches_2d + saliency_patches_pes # (b*t, 32, 16, 16)
                    saliency_patches = einops.rearrange(saliency_patches_2d, '(b t) d h w -> b t (h w) d', b=b, t=t_frames)
                    saliency_patches = saliency_patches.reshape(b, t_frames * p, c)
                else:
                    saliency_patches_2d = einops.rearrange(saliency_patches, 'b (h w) d -> b d h w', h=int(self.saliency_patches**0.5), w=int(self.saliency_patches**0.5))
                    saliency_patches_pes = self.traj_patch_pos_enc.get_pos_enc_for_patch(saliency_patches_2d.shape[2], saliency_patches_2d.shape[3])
                    saliency_patches_2d = saliency_patches_2d + saliency_patches_pes # (bs, 32, 16, 16)
                    saliency_patches = einops.rearrange(saliency_patches_2d, 'b d h w -> b (h w) d') # (bs, 32, 256)

        if isinstance(saliency_patches, torch.Tensor) and saliency_patches.dim() == 4:
            b, t_frames, p, c = saliency_patches.shape
            saliency_patches = saliency_patches.reshape(b, t_frames * p, c)

        # CONCAT HISTORY TOKENS TO saliency
        if self.patches_crss_attn_b4_unet: 
            out, attention_weights = self.cross_attn_b4_unet(out.permute(0,2,1), saliency_patches) # (bs, 720, 32)
            out = out.permute(0,2,1) # (bs, 720, 32) -> (bs, 32, 720)

        down_outs = []
        
        for idx, down in enumerate(self.downs):
            down_outs.append(out)
            if crss_attn_save_params:
                out = down(out, t_emb, saliency_patches=saliency_patches, crss_attn_save_params=crss_attn_save_params, block_idx=idx)
            else:
                out = down(out, t_emb, saliency_patches=saliency_patches)
            
        for idx, mid in enumerate(self.mids):
            if crss_attn_save_params:
                out = mid(out, t_emb, saliency_patches=saliency_patches, crss_attn_save_params=crss_attn_save_params, block_idx=idx)
            else:
                out = mid(out, t_emb, saliency_patches=saliency_patches)
        
        for idx, up in enumerate(self.ups):
            down_out = down_outs.pop()
            if crss_attn_save_params:
                out = up(out, down_out, t_emb, saliency_patches=saliency_patches, crss_attn_save_params=crss_attn_save_params, block_idx=idx)
            else:
                out = up(out, down_out, t_emb, saliency_patches=saliency_patches)

        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)

        return out, attention_weights

        # if self.patches_crss_attn_thru_unet or self.patches_crss_attn_b4_unet:
        #     if self.pos_enc_type == 'learnable':
        #         out = self.traj_positional_encoder(out.permute(0,2,1)).permute(0,2,1) # (bs, 32, 720)
        #         saliency_patches = self.saliency_positional_encoder(saliency_patches) # (bs, 256, 768)
        #     elif self.pos_enc_type == 'sinusoidal_fine':
        #         # add positional embedding to trajectory

        #         if self.should_do_pos_enc:
        #             print("Adding positional embedding to trajectory")
        #             traj_pes = self.traj_patch_pos_enc.get_pos_enc_for_traj(out)
        #             out = out + traj_pes

        #         # add positional embedding to saliency patches
        #         saliency_patches = self.saliency_mlp(saliency_patches) # (bs, 256, 768) -> (bs, 256, 32)

                
        #         if self.should_do_pos_enc:
        #             print("Adding positional embedding to saliency patches")
        #             saliency_patches_2d = einops.rearrange(saliency_patches, 'b (h w) d -> b d h w', h=int(self.saliency_patches**0.5), w=int(self.saliency_patches**0.5))
        #             saliency_patches_pes = self.traj_patch_pos_enc.get_pos_enc_for_patch(saliency_patches_2d.shape[2], saliency_patches_2d.shape[3])
        #             saliency_patches_2d = saliency_patches_2d + saliency_patches_pes # (bs, 32, 16, 16)
        #             saliency_patches = einops.rearrange(saliency_patches_2d, 'b d h w -> b (h w) d') # (bs, 32, 256)

        #     if self.patches_crss_attn_b4_unet: 

        #         out, attention_weights = self.cross_attn_b4_unet(out.permute(0,2,1), saliency_patches) # (bs, 720, 32)
        #         out = out.permute(0,2,1) # (bs, 720, 32) -> (bs, 32, 720)



        # # gets the time embedding and passes it through some linear/non-linear layers
        # t_emb = get_time_embedding(torch.as_tensor(t).long(), self.time_emb_dim) # (bs) -> (bs, 128)
        # t_emb = self.t_proj(t_emb) # (bs, 128) -> (bs, 128)

        # down_outs = []
        
        # for idx, down in enumerate(self.downs):
        #     down_outs.append(out)
        #     if crss_attn_save_params:
        #         out = down(out, t_emb, saliency_patches=saliency_patches, crss_attn_save_params=crss_attn_save_params, block_idx=idx)
        #     else:
        #         out = down(out, t_emb, saliency_patches=saliency_patches)
            
        # for idx, mid in enumerate(self.mids):
        #     if crss_attn_save_params:
        #         out = mid(out, t_emb, saliency_patches=saliency_patches, crss_attn_save_params=crss_attn_save_params, block_idx=idx)
        #     else:
        #         out = mid(out, t_emb, saliency_patches=saliency_patches)
        
        # for idx, up in enumerate(self.ups):
        #     down_out = down_outs.pop()
        #     if crss_attn_save_params:
        #         out = up(out, down_out, t_emb, saliency_patches=saliency_patches, crss_attn_save_params=crss_attn_save_params, block_idx=idx)
        #     else:
        #         out = up(out, down_out, t_emb, saliency_patches=saliency_patches)

        # out = self.norm_out(out)
        # out = nn.SiLU()(out)
        # out = self.conv_out(out)

        # return out, attention_weights



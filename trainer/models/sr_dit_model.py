import math
import numpy as np
from typing import Optional, Tuple, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def build_mlp(hidden_size: int, projector_dim: int, z_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# --- Positional Embedding Utilities (Absolute) ---

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid = np.stack(
        np.meshgrid(
            np.arange(grid_size, dtype=np.float32),
            np.arange(grid_size, dtype=np.float32)
        ),
        axis=0
    ).reshape([2, 1, grid_size, grid_size])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)

    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    omega = 1. / 10000 ** (np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.))
    out = np.einsum('m,d->md', pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


# --- Model Components ---

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(t.device)

        args = t[:, None].float() * freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(self.mlp[0].weight.dtype)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + (dropout_prob > 0), hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def forward(self, labels, train, force_drop_ids=None):
        if (train and self.dropout_prob > 0) or force_drop_ids is not None:
            if force_drop_ids is not None:
                drop_ids = force_drop_ids == 1
            else:
                drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            labels = torch.where(drop_ids, self.num_classes, labels)
        return self.embedding_table(labels)


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.RMSNorm,
            use_v1_residual=True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.v1_lambda = nn.Parameter(torch.tensor(0.5)) if use_v1_residual else None
        self.v_last = None

    def forward(self, x, v1=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        self.v_last = v
        q, k = self.q_norm(q), self.k_norm(k)

        if v1 is not None and self.v1_lambda is not None:
            v = self.v1_lambda * v1 + (1.0 - self.v1_lambda) * v

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class ConvMLP(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        groups = 32
        self.conv = nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=groups)
        self.fc2 = nn.Conv2d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x, h, w):
        B, N, C = x.shape
        if N == h * w + 1:
            cls_token = x[:, :1, :]
            spatial_x = x[:, 1:, :]
        else:
            cls_token = None
            spatial_x = x

        spatial_x = spatial_x.transpose(1, 2).reshape(B, C, h, w)
        spatial_x = self.fc1(spatial_x)
        spatial_x = self.act(spatial_x)
        spatial_x = self.conv(spatial_x)
        spatial_x = self.act(spatial_x)
        spatial_x = self.fc2(spatial_x)
        spatial_x = spatial_x.flatten(2).transpose(1, 2)
        spatial_x = self.drop(spatial_x)

        if cls_token is not None:
            return torch.cat([cls_token, spatial_x], dim=1)
        return spatial_x


class SRDiTBlock(nn.Module):
    def __init__(
            self,
            hidden_size,
            num_heads,
            mlp_ratio=4.0,
            use_v1_residual=True,
            context_dim=None,
            **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads,
            qkv_bias=True,
            qk_norm=block_kwargs.get("qk_norm", False),
            use_v1_residual=use_v1_residual
        )

        self.norm2 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            batch_first=True,
            kdim=context_dim,
            vdim=context_dim
        )

        self.norm3 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = ConvMLP(hidden_size, int(hidden_size * mlp_ratio), nn.GELU, 0)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size)
        )

    def forward(self, x, cond, context, v1=None, patch_grid_size=None):
        shift_msa, scale_msa, gate_msa, shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(cond).chunk(9, dim=-1)

        x_norm1 = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_norm1, v1=v1)

        x_norm2 = modulate(self.norm2(x), shift_ca, scale_ca)
        x = x + gate_ca.unsqueeze(1) * self.cross_attn(x_norm2, context, context)[0]

        x_norm3 = modulate(self.norm3(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm3, patch_grid_size[0], patch_grid_size[1])

        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, cls_token_dim):
        super().__init__()
        self.norm_final = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.linear_cls = nn.Linear(hidden_size, cls_token_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size)
        )

    def forward(self, x, c, cls=None):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)

        if cls is None:
            return self.linear(x), None

        return self.linear(x[:, 1:]), self.linear_cls(x[:, 0])


class SRDiT(nn.Module):
    def __init__(
            self,
            input_size=32,
            patch_size=2,
            in_channels=4,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            num_classes=1000,
            context_dim=768,
            cls_token_dim=768,
            z_dims=[1024],
            projector_dim=2048,
            **block_kwargs
    ):
        super().__init__()
        self.config = {
            "input_size": input_size,
            "patch_size": patch_size,
            "in_channels": in_channels,
            "hidden_size": hidden_size,
            "depth": depth,
            "num_heads": num_heads,
            "mlp_ratio": mlp_ratio,
            "class_dropout_prob": class_dropout_prob,
            "num_classes": num_classes,
            "context_dim": context_dim,
            "cls_token_dim": cls_token_dim,
            "z_dims": z_dims,
            "projector_dim": projector_dim,
            **block_kwargs
        }
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.depth = depth
        self.z_dims = z_dims
        self.input_size = input_size

        self.x_embedder = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True
        )

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        num_patches = (input_size // patch_size) ** 2
        self.num_patches = num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, hidden_size),
            requires_grad=False
        )

        self.blocks = nn.ModuleList([
            SRDiTBlock(hidden_size, num_heads, mlp_ratio, i > 0, context_dim, **block_kwargs)
            for i in range(depth)
        ])

        self.projectors = nn.ModuleList([
            build_mlp(hidden_size, projector_dim, z_dim)
            for z_dim in z_dims
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels, cls_token_dim)
        self.cls_projectors2 = nn.Linear(cls_token_dim, hidden_size)
        self.wg_norm = nn.RMSNorm(hidden_size, eps=1e-6)

        self.initialize_weights()

    def initialize_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_init_weights)

        grid_size = int(self.num_patches ** 0.5)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size, True, 1)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        nn.init.xavier_uniform_(self.x_embedder.weight.data.view([self.x_embedder.weight.data.shape[0], -1]))
        if self.x_embedder.bias is not None:
            nn.init.constant_(self.x_embedder.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        for b in self.blocks:
            nn.init.constant_(b.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(b.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)

    def unpatchify(self, x, h, w):
        c, p = self.in_channels, self.patch_size
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], c, h * p, w * p))

    def forward(self, x, t, context, y=None, cls_token=None):
        patch_grid_size = (x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size)

        x = self.x_embedder(x)
        x = x.flatten(2).transpose(1, 2)

        if cls_token is not None:
            cls_feat = self.wg_norm(self.cls_projectors2(cls_token)).unsqueeze(1)
            x = torch.cat((cls_feat, x), dim=1)

            if x.shape[1] == self.pos_embed.shape[1]:
                x = x + self.pos_embed
        else:
            raise ValueError("cls_token is required")

        cond = self.t_embedder(t)
        if y is not None:
            cond = cond + self.y_embedder(y, self.training)

        v1_full = None

        for i in range(self.depth):
            # RoPE and rope_ids removed from arguments
            x = self.blocks[i](x, cond, context, v1=v1_full, patch_grid_size=patch_grid_size)

            if v1_full is None:
                v1_full = self.blocks[i].attn.v_last

        zs = []
        for proj, z_dim in zip(self.projectors, self.z_dims):
            z = proj(x.reshape(-1, x.shape[-1]))
            z = z.reshape(x.shape[0], -1, z_dim)
            zs.append(z)

        x_out, cls_token_out = self.final_layer(x, cond, cls_token)

        return self.unpatchify(x_out, patch_grid_size[0], patch_grid_size[1]), zs, cls_token_out

    def save_pretrained(self, save_directory):
        import os
        import json

        os.makedirs(save_directory, exist_ok=True)

        # Save config
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=4)

        # Save weights
        try:
            from safetensors.torch import save_file
            save_path = os.path.join(save_directory, "diffusion_pytorch_model.safetensors")
            save_file(self.state_dict(), save_path)
        except ImportError:
            save_path = os.path.join(save_directory, "diffusion_pytorch_model.bin")
            torch.save(self.state_dict(), save_path)
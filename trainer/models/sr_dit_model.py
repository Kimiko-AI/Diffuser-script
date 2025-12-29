import math
import numpy as np
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.vision_transformer import PatchEmbed

def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')

class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(self, dim, pt_seq_len=16, custom_freqs=None, freqs_for='lang', theta=10000, max_freq=10, num_freqs=1):
        super().__init__()
        self.dim = dim
        self.pt_seq_len = pt_seq_len
        if custom_freqs is not None:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * math.pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        self.register_buffer("freqs", freqs)
        self.register_buffer("freqs_cos", None, persistent=False)
        self.register_buffer("freqs_sin", None, persistent=False)
        self.cached_grid_size = None
        self.rot_dim = dim * 2 if dim % 2 == 0 else dim

    def _update_grid(self, h, w, device, dtype):
        if self.cached_grid_size == (h, w) and self.freqs_cos is not None: return
        t_h = torch.arange(h, device=device, dtype=self.freqs.dtype)
        t_w = torch.arange(w, device=device, dtype=self.freqs.dtype)
        freqs_h = repeat(torch.einsum('i, j -> i j', t_h, self.freqs), '... n -> ... (n r)', r=2)
        freqs_w = repeat(torch.einsum('i, j -> i j', t_w, self.freqs), '... n -> ... (n r)', r=2)
        freqs_2d = torch.cat([freqs_h[:, None, :].expand(h, w, -1), freqs_w[None, :, :].expand(h, w, -1)], dim=-1)
        self.freqs_cos = freqs_2d.cos().reshape(-1, freqs_2d.shape[-1]).to(dtype=dtype)
        self.freqs_sin = freqs_2d.sin().reshape(-1, freqs_2d.shape[-1]).to(dtype=dtype)
        self.cached_grid_size = (h, w)

    def _gather_cos_sin(self, rope_ids, N, device, dtype):
        cos_table, sin_table = self.freqs_cos.to(dtype=dtype, device=device), self.freqs_sin.to(dtype=dtype, device=device)
        if rope_ids is None:
            return cos_table.view(1, 1, N, -1), sin_table.view(1, 1, N, -1)
        rope_ids = rope_ids.to(device=device, dtype=torch.long)
        cos = cos_table[rope_ids].unsqueeze(1) if rope_ids.dim() == 2 else cos_table.index_select(0, rope_ids).view(1, 1, N, -1)
        sin = sin_table[rope_ids].unsqueeze(1) if rope_ids.dim() == 2 else sin_table.index_select(0, rope_ids).view(1, 1, N, -1)
        return cos, sin

    def forward(self, t, rope_ids=None, patch_grid_size=None):
        B, Hh, N, D = t.shape
        if patch_grid_size: self._update_grid(patch_grid_size[0], patch_grid_size[1], t.device, t.dtype)
        elif rope_ids is None and self.cached_grid_size is None:
            side = int(math.sqrt(N))
            if side * side != N: side = int(math.sqrt(N - 1))
            self._update_grid(side, side, t.device, t.dtype)

        if rope_ids is None:
            extra = N - (self.cached_grid_size[0] * self.cached_grid_size[1]) if self.cached_grid_size else 0
            t_lead, t_tail = (t[:, :, :extra, :], t[:, :, extra:, :]) if extra > 0 else (None, t)
            if t_tail.shape[-2] == 0: return t
            cos, sin = self._gather_cos_sin(None, t_tail.shape[-2], t.device, t.dtype)
            rot = t_tail * cos + rotate_half(t_tail) * sin
            return torch.cat([t_lead, rot], dim=-2) if t_lead is not None else rot
        else:
            extra = N - rope_ids.shape[-1]
            t_lead, t_tail = (t[:, :, :extra, :], t[:, :, extra:, :]) if extra > 0 else (None, t)
            cos, sin = self._gather_cos_sin(rope_ids, t_tail.shape[-2], t.device, t.dtype)
            rot = t_tail * cos + rotate_half(t_tail) * sin
            return torch.cat([t_lead, rot], dim=-2) if t_lead is not None else rot

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid = np.stack(np.meshgrid(np.arange(grid_size, dtype=np.float32), np.arange(grid_size, dtype=np.float32)), axis=0).reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0: pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    omega = 1. / 10000 ** (np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.))
    out = np.einsum('m,d->md', pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(t.device)
        args = t[:, None].float() * freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(t.dtype)
        return self.mlp(t_freq)

class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + (dropout_prob > 0), hidden_size)
        self.num_classes, self.dropout_prob = num_classes, dropout_prob

    def forward(self, labels, train, force_drop_ids=None):
        if (train and self.dropout_prob > 0) or force_drop_ids is not None:
            drop_ids = force_drop_ids == 1 if force_drop_ids is not None else torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            labels = torch.where(drop_ids, self.num_classes, labels)
        return self.embedding_table(labels)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False, attn_drop=0., proj_drop=0., norm_layer=nn.RMSNorm, use_v1_residual=True):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.v1_lambda = nn.Parameter(torch.tensor(0.5)) if use_v1_residual else None
        self.v_last = None

    def forward(self, x, rope=None, rope_ids=None, v1=None, patch_grid_size=None):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        self.v_last = v
        q, k = self.q_norm(q), self.k_norm(k)
        if v1 is not None and self.v1_lambda is not None: v = self.v1_lambda * v1 + (1.0 - self.v1_lambda) * v
        if rope: q, k = rope(q, rope_ids, patch_grid_size), rope(k, rope_ids, patch_grid_size)
        x = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))

class ConvMLP(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        groups = hidden_features // 32 if hidden_features % 32 == 0 else 1
        self.conv = nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=groups)
        self.fc2 = nn.Conv2d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x, h, w):
        B, N, C = x.shape
        x = self.fc1(x.transpose(1, 2).reshape(B, C, h, w))
        x = self.fc2(self.conv(self.act(self.act(x))))
        return self.drop(x.flatten(2).transpose(1, 2))

class SRDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_v1_residual=True, context_dim=None, **block_kwargs):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads, qkv_bias=True, qk_norm=block_kwargs.get("qk_norm", False), use_v1_residual=use_v1_residual)
        self.norm2 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, kdim=context_dim, vdim=context_dim)
        self.norm3 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = ConvMLP(hidden_size, int(hidden_size * mlp_ratio), nn.GELU, 0)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size))

    def forward(self, x, cond, context, feat_rope=None, rope_ids=None, v1=None, patch_grid_size=None):
        shift_msa, scale_msa, gate_msa, shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(9, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), feat_rope, rope_ids, v1, patch_grid_size)
        x = x + gate_ca.unsqueeze(1) * self.cross_attn(modulate(self.norm2(x), shift_ca, scale_ca), context, context)[0]
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp), patch_grid_size[0], patch_grid_size[1])
        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, cls_token_dim):
        super().__init__()
        self.norm_final = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.linear_cls = nn.Linear(hidden_size, cls_token_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x, c, cls=None):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        if cls is None: return self.linear(x), None
        return self.linear(x[:, 1:]), self.linear_cls(x[:, 0])

class SRDiT(nn.Module):
    def __init__(self, input_size=32, patch_size=2, in_channels=4, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, class_dropout_prob=0.1, num_classes=1000, context_dim=768, cls_token_dim=768, z_dims=[1024], projector_dim=2048, **block_kwargs):
        super().__init__()
        self.in_channels, self.patch_size, self.depth, self.z_dims = in_channels, patch_size, depth, z_dims
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder, self.y_embedder = TimestepEmbedder(hidden_size), LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.x_embedder.num_patches + 1, hidden_size), requires_grad=False)
        self.blocks = nn.ModuleList([SRDiTBlock(hidden_size, num_heads, mlp_ratio, i > 0, context_dim, **block_kwargs) for i in range(depth)])
        self.projectors = nn.ModuleList([build_mlp(hidden_size, projector_dim, z_dim) for z_dim in z_dims])
        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels, cls_token_dim)
        self.cls_projectors2, self.wg_norm = nn.Linear(cls_token_dim, hidden_size), nn.RMSNorm(hidden_size, eps=1e-6)
        self.feat_rope = VisionRotaryEmbeddingFast(dim=(hidden_size // num_heads) // 2, pt_seq_len=input_size // patch_size)
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(lambda m: (torch.nn.init.xavier_uniform_(m.weight), nn.init.constant_(m.bias, 0)) if isinstance(m, nn.Linear) else None)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5), True, 1)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        nn.init.xavier_uniform_(self.x_embedder.proj.weight.data.view([self.x_embedder.proj.weight.data.shape[0], -1]))
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        
        for b in self.blocks:
            nn.init.constant_(b.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(b.adaLN_modulation[-1].bias, 0)
            
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)

    def unpatchify(self, x):
        c, p = self.in_channels, self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        return torch.einsum('nhwpqc->nchpwq', x.reshape(shape=(x.shape[0], h, w, p, p, c))).reshape(shape=(x.shape[0], c, h * p, w * p))

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def forward(self, x, t, context, y=None, cls_token=None):
        patch_grid_size = (x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size)
        x = self.x_embedder(x)
        if cls_token is not None:
            x = torch.cat((self.wg_norm(self.cls_projectors2(cls_token)).unsqueeze(1), x), dim=1)
            if x.shape[1] == self.pos_embed.shape[1]: x = x + self.pos_embed
        else: exit("cls_token is required")
        cond = self.t_embedder(t) + (self.y_embedder(y, self.training) if y is not None else 0)
        v1_full = None
        
        for i in range(self.depth):
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                # Checkpointing requires inputs to be tensors requiring grad. 
                # x has grad. cond has grad. context has grad. 
                # feat_rope, rope_ids, patch_grid_size are not requiring grad.
                # checkpoint handles args.
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.blocks[i]),
                    x, cond, context, self.feat_rope, None, v1_full, patch_grid_size,
                    use_reentrant=False
                )
            else:
                x = self.blocks[i](x, cond, context, self.feat_rope, None, v1_full, patch_grid_size)
            
            if v1_full is None: v1_full = self.blocks[i].attn.v_last
            
        zs = [proj(x.reshape(-1, x.shape[-1])).reshape(x.shape[0], -1, z_dim) for proj, z_dim in zip(self.projectors, self.z_dims)]
        x_out, cls_token_out = self.final_layer(x, cond, cls_token)
        return self.unpatchify(x_out), zs, cls_token_out
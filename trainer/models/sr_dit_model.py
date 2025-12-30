import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):
    """ Standard 2D ResNet Block with GroupNorm and SiLU """

    def __init__(self, in_channels, out_channels=None, dropout=0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        self.norm1 = nn.GroupNorm(32, in_channels)  # GroupNorm is standard for GenAI
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.act2(self.norm2(h)))
        return self.shortcut(x) + h


class RefinerFinalLayer(nn.Module):
    """
    Replaces the standard FinalLayer.
    1. AdaLN Modulate Transformer Output
    2. Concat with Input Embeddings (Skip Connection)
    3. 2x ResNet Blocks
    4. Final Projection
    """

    def __init__(self, hidden_size, patch_size, out_channels, cls_token_dim):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels

        # 1. Conditioning for Transformer Output
        self.norm_final = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size)
        )

        # 2. ResNet Refiner Components
        # Input dim is hidden_size * 2 (Transformer Out + Input Skip)
        self.fusion_conv = nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=1)

        self.res_block1 = ResNetBlock(hidden_size)
        self.res_block2 = ResNetBlock(hidden_size)

        # 3. Final Projections
        # Spatial Output
        self.final_conv = nn.Conv2d(hidden_size, patch_size * patch_size * out_channels, kernel_size=1)
        # Class Token Output
        self.linear_cls = nn.Linear(hidden_size, cls_token_dim)

    def forward(self, x, c, x_skip, cls=None):
        """
        x: Transformer output (B, T, C)
        c: Conditioning embedding
        x_skip: Original patch embeddings (B, C, H, W)
        """
        # --- A. Process Transformer Features ---
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)

        # Split CLS and Spatial
        if cls is not None:
            cls_out = self.linear_cls(x[:, 0])
            x_spatial = x[:, 1:]
        else:
            cls_out = None
            x_spatial = x

        # Reshape tokens back to 2D grid: (B, H*W, C) -> (B, C, H, W)
        B, L, C = x_spatial.shape
        H = W = int(L ** 0.5)
        x_spatial = x_spatial.transpose(1, 2).reshape(B, C, H, W)

        # --- B. Concatenate with Skip Connection ---
        # x_skip shape is (B, C, H, W).
        # Note: If x_skip has gradient, this path backpropagates to the very first layer directly.
        x_fused = torch.cat([x_spatial, x_skip], dim=1)  # (B, 2C, H, W)

        # --- C. ResNet Refinement ---
        x_fused = self.fusion_conv(x_fused)  # Reduce 2C -> C
        x_fused = self.res_block1(x_fused)
        x_fused = self.res_block2(x_fused)

        # --- D. Final Projection ---
        # Output: (B, p*p*C_out, H, W)
        # We flatten it back to sequence format to match unpatchify expectation
        # or just return 2D if unpatchify handles it.
        # Your unpatchify expects (N, T, C). Let's stick to that contract.

        x_out = self.final_conv(x_fused)  # (B, P*P*OutC, H, W)
        x_out = x_out.flatten(2).transpose(1, 2)  # (B, L, P*P*OutC)

        # Reattach CLS placeholder if needed for shape consistency,
        # though unpatchify usually ignores CLS.
        # We return the spatial part mostly.

        return x_out, cls_out

# --- Helpers ---

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def build_mlp(hidden_size: int, projector_dim: int, z_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )


# --- Components ---

class Mlp(nn.Module):
    """ Standard MLP (Linear -> GELU -> Linear) """

    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SRDiTBlock(nn.Module):
    def __init__(
            self,
            hidden_size,
            num_heads,
            mlp_ratio=4.0,
            context_dim=None,
            **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Self Attention
        self.attn = Attention(
            hidden_size,
            num_heads,
            qkv_bias=True,
            qk_norm=block_kwargs.get("qk_norm", False),
        )

        self.norm2 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Cross Attention (kept from your original SRDiT design)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            batch_first=True,
            kdim=context_dim,
            vdim=context_dim
        )

        self.norm3 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Standard MLP (No Conv)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=nn.GELU,
            drop=0
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size)
        )

    def forward(self, x, cond, context):
        # 1. Modulate Params
        shift_msa, scale_msa, gate_msa, shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(cond).chunk(9, dim=-1)

        # 2. Self Attention
        x_norm1 = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_norm1)

        # 3. Cross Attention
        x_norm2 = modulate(self.norm2(x), shift_ca, scale_ca)
        x = x + gate_ca.unsqueeze(1) * self.cross_attn(x_norm2, context, context)[0]

        # 4. Standard MLP
        x_norm3 = modulate(self.norm3(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm3)

        return x


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
            context_dim=768,
            cls_token_dim=768,
            z_dims=[1024],
            projector_dim=2048,
            # --- SPRINT Params ---
            encoder_depth=8,
            decoder_depth=2,
            sprint_drop_ratio=0.75,
            path_drop_prob=0.05,
            **block_kwargs
    ):
        super().__init__()
        # ... (Same initialization as before) ...
        self.config = locals()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.depth = depth
        self.z_dims = z_dims
        self.input_size = input_size

        self.num_f = encoder_depth
        self.num_h = decoder_depth
        self.num_g = depth - self.num_f - self.num_h

        self.sprint_drop_ratio = sprint_drop_ratio
        self.path_drop_prob = path_drop_prob

        # Embeddings
        self.x_embedder = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = CoordsEmbedder(hidden_size)

        num_patches = (input_size // patch_size) ** 2
        self.num_patches = num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_size), requires_grad=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.fusion_proj = nn.Linear(2 * hidden_size, hidden_size)

        self.blocks = nn.ModuleList([
            SRDiTBlock(hidden_size, num_heads, mlp_ratio, context_dim=context_dim, **block_kwargs)
            for _ in range(depth)
        ])

        self.projectors = nn.ModuleList([build_mlp(hidden_size, projector_dim, z_dim) for z_dim in z_dims])

        # --- CHANGED: Use RefinerFinalLayer ---
        self.final_layer = RefinerFinalLayer(hidden_size, patch_size, in_channels, cls_token_dim)

        self.cls_projectors2 = nn.Linear(cls_token_dim, hidden_size)
        self.wg_norm = nn.RMSNorm(hidden_size, eps=1e-6)

        self.initialize_weights()

    def initialize_weights(self):
        # ... (Same init logic) ...
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):  # Init Conv layers in ResNet/Embedder
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

        self.apply(_init_weights)

        grid_size = int(self.num_patches ** 0.5)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size, True, 1)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Init final modulation to 0 for stability
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        # Init final conv to 0 (common practice in DiT/SiT to start from noise)
        nn.init.constant_(self.final_layer.final_conv.weight, 0)
        nn.init.constant_(self.final_layer.final_conv.bias, 0)

    # ... (Helpers _drop_tokens, _pad_with_mask, _sprint_fuse remain same) ...
    def _drop_tokens(self, x, drop_ratio):
        # (Same as previous response)
        B, N, C = x.shape
        cls_token = x[:, :1, :]
        spatial_x = x[:, 1:, :]
        T = spatial_x.shape[1]
        if T <= 1 or drop_ratio <= 0: return x, None
        num_keep = max(1, int(T * (1.0 - drop_ratio)))
        noise = torch.rand(B, T, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :num_keep]
        spatial_keep = spatial_x.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, C))
        return torch.cat([cls_token, spatial_keep], dim=1), ids_keep

    def _pad_with_mask(self, x_sparse, ids_keep, T_full):
        # (Same as previous response)
        if ids_keep is None: return x_sparse
        cls_token = x_sparse[:, :1, :]
        spatial_sparse = x_sparse[:, 1:, :]
        B, _, C = spatial_sparse.shape
        spatial_full = self.mask_token.expand(B, T_full, C).clone()
        spatial_full.scatter_(1, ids_keep.unsqueeze(-1).expand(-1, -1, C), spatial_sparse)
        return torch.cat([cls_token, spatial_full], dim=1)

    def _sprint_fuse(self, f_dense, g_full):
        h = torch.cat([f_dense, g_full], dim=-1)
        return self.fusion_proj(h)

    def unpatchify(self, x, h, w):
        c, p = self.in_channels, self.patch_size
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], c, h * p, w * p))

    def forward(self, x, t, context, y=None, cls_token=None, uncond=False):
        patch_grid_size = (x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size)

        # 1. Embeddings
        # Capture SKIP connection here before flattening
        x_skip = self.x_embedder(x)  # (B, Hidden, H/p, W/p)
        x = x_skip.flatten(2).transpose(1, 2)  # (B, L, Hidden)

        if cls_token is not None:
            cls_feat = self.wg_norm(self.cls_projectors2(cls_token)).unsqueeze(1)
            x = torch.cat((cls_feat, x), dim=1)
            if x.shape[1] == self.pos_embed.shape[1]:
                x = x + self.pos_embed
        else:
            raise ValueError("cls_token is required")

        cond = self.t_embedder(t)
        y_emb = self.y_embedder(y, self.training)
        if y_emb.shape[0] == 1 and x.shape[0] > 1:
            y_emb = y_emb.repeat(x.shape[0], 1)
        cond = cond + y_emb

        # --- SPRINT Flow (Same as before) ---

        # Stage 1: Encoder
        for i in range(self.num_f):
            x = self.blocks[i](x, cond, context)
        x_enc = x

        # Projectors
        zs = []
        for proj, z_dim in zip(self.projectors, self.z_dims):
            z = proj(x.reshape(-1, x.shape[-1])).reshape(x.shape[0], -1, z_dim)
            zs.append(z)

        # Stage 2: Middle (Sparse)
        spatial_len = x.shape[1] - 1
        if self.training:
            x_mid, ids_keep = self._drop_tokens(x, self.sprint_drop_ratio)
        else:
            x_mid, ids_keep = x, None

        for i in range(self.num_f, self.num_f + self.num_g):
            x_mid = self.blocks[i](x_mid, cond, context)

        g_pad = self._pad_with_mask(x_mid, ids_keep, T_full=spatial_len)

        if self.training and self.path_drop_prob > 0.0:
            if torch.rand(1).item() < self.path_drop_prob:
                g_pad = g_pad * 0.0 + self.mask_token.expand_as(g_pad)
        elif uncond:
            g_pad = g_pad * 0.0 + self.mask_token.expand_as(g_pad)

        # Stage 3: Fusion
        x_fused = self._sprint_fuse(x_enc, g_pad)

        # Stage 4: Decoder
        x_dec = x_fused
        for i in range(self.num_f + self.num_g, self.depth):
            x_dec = self.blocks[i](x_dec, cond, context)

        # --- FINAL LAYER (Modified) ---
        # We pass x_skip (the 2D conv feature map) directly to the final layer
        x_out, cls_token_out = self.final_layer(x_dec, cond, x_skip, cls=cls_token)

        return self.unpatchify(x_out, patch_grid_size[0], patch_grid_size[1]), zs, cls_token_out
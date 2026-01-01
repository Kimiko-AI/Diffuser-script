import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==========================================
# 1. Helpers & Basic Layers
# ==========================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight


class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.w3 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


# ==========================================
# 2. Embedders
# ==========================================

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class CoordEmbedder(nn.Module):
    """
    Standard Fourier Features to embed pixel coordinates for the decoder.
    """

    def __init__(self, in_channels, hidden_size, max_freqs=8):
        super().__init__()
        self.max_freqs = max_freqs
        freq_dim = 2 * max_freqs * 2
        self.embedder = nn.Linear(in_channels + freq_dim, hidden_size, bias=True)

    def forward(self, inputs):
        B, N, C = inputs.shape
        patch_size = int(N ** 0.5)
        device = inputs.device

        # Grid [-1, 1]
        h = torch.linspace(-1, 1, patch_size, device=device)
        w = torch.linspace(-1, 1, patch_size, device=device)
        y, x = torch.meshgrid(h, w, indexing='ij')
        coords = torch.stack([x, y], dim=-1).reshape(1, N, 2).repeat(B, 1, 1)

        # Fourier features
        freqs = 2.0 ** torch.arange(self.max_freqs, device=device) * math.pi
        coords_feat = coords.unsqueeze(-1) * freqs.view(1, 1, 1, -1)
        coords_feat = torch.cat([torch.sin(coords_feat), torch.cos(coords_feat)], dim=-1)
        coords_feat = coords_feat.reshape(B, N, -1)

        x = torch.cat([inputs, coords_feat], dim=-1)
        return self.embedder(x)


# ==========================================
# 3. Blocks
# ==========================================

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)
        self.kv_y = nn.Linear(dim, dim * 2, bias=qkv_bias)  # Cross attn projection

    def forward(self, x, context):
        B, N, C = x.shape
        # Self-Attention
        qkv_x = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, kx, vx = qkv_x[0], qkv_x[1], qkv_x[2]
        q, kx = self.q_norm(q), self.k_norm(kx)

        # Cross-Attention
        kv_y = self.kv_y(context).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        ky, vy = kv_y[0], kv_y[1]
        ky = self.k_norm(ky)

        # Concat Keys/Values
        k = torch.cat([kx, ky], dim=2)
        v = torch.cat([vx, vy], dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class FlattenDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=False)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLU(hidden_size, int(hidden_size * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, context, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), context)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class TextRefineBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLU(hidden_size, int(hidden_size * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        norm_x = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + gate_msa * attn_out
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, cond):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class SimpleMLPAdaLN(nn.Module):
    def __init__(self, in_channels, model_channels, out_channels, z_channels, num_res_blocks, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.cond_embed = nn.Linear(z_channels, patch_size ** 2 * model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)
        self.res_blocks = nn.ModuleList([ResBlock(model_channels) for _ in range(num_res_blocks)])
        self.final_norm = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.final_linear = nn.Linear(model_channels, out_channels)

    def forward(self, x, c):
        x = self.input_proj(x)
        c = self.cond_embed(c)
        cond = c.reshape(c.shape[0], self.patch_size ** 2, -1)
        for block in self.res_blocks:
            x = block(x, cond)
        return self.final_linear(self.final_norm(x))


# ==========================================
# 4. Main Model
# ==========================================

class DeCoDiT(nn.Module):
    def __init__(
            self,
            in_channels=3,
            hidden_size=1152,
            decoder_hidden_size=64,
            num_encoder_blocks=18,
            num_decoder_blocks=4,
            num_text_blocks=4,
            patch_size=2,
            txt_embed_dim=1024,
            txt_max_length=100,
            num_heads=16
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size

        # 1. Embeddings
        self.s_embedder = nn.Linear(in_channels * patch_size ** 2, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)

        # New: Projector for the B,4 conditioning vector 'y'
        self.label_proj = nn.Linear(4, hidden_size)

        self.y_embedder = nn.Linear(txt_embed_dim, hidden_size)
        self.y_pos_embedding = nn.Parameter(torch.randn(1, txt_max_length, hidden_size) * 0.02)

        # Absolute Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, hidden_size))

        # 2. Blocks
        self.text_refine_blocks = nn.ModuleList([
            TextRefineBlock(hidden_size, num_heads) for _ in range(num_text_blocks)
        ])
        self.blocks = nn.ModuleList([
            FlattenDiTBlock(hidden_size, num_heads) for _ in range(num_encoder_blocks)
        ])

        # 3. Decoder
        self.x_embedder = CoordEmbedder(in_channels, decoder_hidden_size, max_freqs=8)
        self.dec_net = SimpleMLPAdaLN(
            in_channels=decoder_hidden_size,
            model_channels=decoder_hidden_size,
            out_channels=in_channels,
            z_channels=hidden_size,
            num_res_blocks=num_decoder_blocks,
            patch_size=patch_size
        )

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.s_embedder.weight)
        nn.init.constant_(self.s_embedder.bias, 0)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.zeros_(self.label_proj.weight)
        nn.init.zeros_(self.label_proj.bias)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.text_refine_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def get_pos_embed(self, num_patches):
        if num_patches <= self.pos_embed.shape[1]:
            return self.pos_embed[:, :num_patches, :]
        return F.interpolate(self.pos_embed.transpose(1, 2), size=num_patches, mode='linear').transpose(1, 2)

    def forward(self, x, t, context, y=None):
        """
        x: [B, C, H, W] Input Image
        t: [B] Timesteps
        context: [B, L, D] Text Embeddings
        y: [B, 4] Vector Conditioning
        """
        B, _, H, W = x.shape

        # --- 1. Prepare Inputs ---
        x_patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        N = x_patches.shape[1]

        # --- 2. Global Conditioning (Time + Vector y) ---
        t_emb = self.t_embedder(t.view(-1)).view(B, -1, self.hidden_size)
        c = t_emb

        # Add vector conditioning 'y' to AdaLN inputs
        if y is not None:
            # y is [B, 4], project to hidden and add to time embedding
            y_emb = self.label_proj(y).unsqueeze(1)  # [B, 1, hidden]
            c = c + y_emb

        c = F.silu(c)  # Global condition for AdaLN

        # --- 3. Text Refinement ---
        ctx = self.y_embedder(context) + self.y_pos_embedding[:, :context.shape[1], :]
        for block in self.text_refine_blocks:
            ctx = block(ctx, c)

        # --- 4. Encoder ---
        s = self.s_embedder(x_patches) + self.get_pos_embed(N)
        for block in self.blocks:
            s = block(s, ctx, c)

        # --- 5. Decoder Conditioning ---
        # Fuse global condition 'c' into the latent 's'
        # 's' becomes the conditioning context for the decoder
        s = F.silu(c + s)
        s = s.view(B * N, self.hidden_size)

        # --- 6. Pixel Decoding ---
        x_input = x_patches.reshape(B * N, self.in_channels, self.patch_size ** 2).transpose(1, 2)
        x_embedded = self.x_embedder(x_input)
        out = self.dec_net(x_embedded, s)

        # --- 7. Reshape ---
        out = out.transpose(1, 2).reshape(B, N, -1).transpose(1, 2)
        out = F.fold(out, output_size=(H, W), kernel_size=self.patch_size, stride=self.patch_size)

        return out
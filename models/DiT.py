import torch
import torch.nn as nn
import math
from timm.layers import DropPath


def modulate(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        # d_model: 词向量维度
        assert d_model % 2 == 0
        i_seq = torch.linspace(0, max_seq_len - 1, max_seq_len)
        j_seq = torch.linspace(0, d_model - 2, d_model // 2)
        pos, two_i = torch.meshgrid(i_seq, j_seq)
        pe_2i = torch.sin(pos / (10000 ** (two_i / d_model)))
        pe_2i_1 = torch.cos(pos / (10000 ** (two_i / d_model)))
        pe = torch.stack((pe_2i, pe_2i_1), 2).reshape(1, max_seq_len, d_model)
        self.register_buffer('pe', pe, False)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        pe = self.pe
        # x_prime = x * d_model ** 0.5
        return x + pe[:, 0:seq_len, :]


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, dim: int):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))  # shape: (dim/2,)

        # 预先构建 (seq_len, dim/2) 的角度矩阵
        t = torch.arange(max_seq_len).float()[:, None]  # shape: (seq_len, 1)
        freqs = t * inv_freq[None, :]  # shape: (seq_len, dim/2)

        self.register_buffer("cos", freqs.cos(), persistent = False)  # (seq_len, dim/2)
        self.register_buffer("sin", freqs.sin(), persistent = False)

    def forward(self, x):
        # x: (batch, seq_len, dim)
        x1 = x[..., ::2]  # 取偶数维
        x2 = x[..., 1::2]  # 取奇数维

        cos = self.cos[:x.size(1)].to(x.device)  # (seq_len, dim/2)
        sin = self.sin[:x.size(1)].to(x.device)

        # 形状匹配: (batch, seq_len, dim/2)
        cos = cos[None, :, :]
        sin = sin[None, :, :]

        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.stack([out1, out2], dim = -1).flatten(-2)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_ch, n_ch: int = 768):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, n_ch, kernel_size = patch_size, stride = patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, n_ch, H//P, W//P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, n_ch, frequency_embedding_size = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, n_ch, bias = True),
            nn.SiLU(),
            nn.Linear(n_ch, n_ch, bias = True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period = 10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start = 0, end = half, dtype = torch.float32) / half
        ).to(device = t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim = -1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim = -1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiTBlock(nn.Module):
    def __init__(self, n_ch: int, heads: int, mlp_ratio: int = 4,
                 attn_dropout: float = 0.0, drop_path_rate: float = 0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(n_ch, elementwise_affine = False, eps = 1e-6)
        self.att = nn.MultiheadAttention(n_ch, heads, batch_first = True, dropout = attn_dropout)
        self.norm2 = nn.LayerNorm(n_ch, elementwise_affine = False, eps = 1e-6)

        mlp_hidden_dim = int(n_ch * mlp_ratio)
        self.mlp_fc1 = nn.Linear(n_ch, mlp_hidden_dim)
        self.mlp_fc2 = nn.Linear(mlp_hidden_dim, n_ch)
        self.act = nn.GELU(approximate = "tanh")

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(n_ch, 6 * n_ch)
        )

        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim = -1)

        x1 = modulate(self.norm1(x), scale_msa, shift_msa)
        att_out, _ = self.att(x1, x1, x1)
        x = x + self.drop_path(gate_msa.unsqueeze(1) * att_out)

        x2 = modulate(self.norm2(x), scale_mlp, shift_mlp)
        mlp_out = self.mlp_fc2(self.act(self.mlp_fc1(x2)))
        x = x + self.drop_path(gate_mlp.unsqueeze(1) * mlp_out)

        return x


class FinalLayer(nn.Module):
    def __init__(self, n_ch, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(n_ch, elementwise_affine = False, eps = 1e-6)
        self.linear = nn.Linear(n_ch, patch_size * patch_size * out_channels, bias = True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(n_ch, 2 * n_ch, bias = True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim = 1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    def __init__(self, input_size: int, patch_size: int, input_ch: int, n_ch: int, n_blocks: int, num_heads: int = 16,
                 mlp_ratio: int = 4, learn_sigma: bool = False, pe: str = "abs", attn_dropout = 0.0, mlp_dropout = 0.0):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.input_ch = input_ch
        self.n_ch = n_ch
        self.output_ch = input_ch * 2 if learn_sigma else input_ch
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_patches = (input_size // patch_size) ** 2

        self.x_embedder = PatchEmbed(patch_size, input_ch, n_ch)
        self.t_embedder = TimestepEmbedder(n_ch)
        if pe == "abs":
            self.pos_embed = PositionalEncoding(self.num_patches, n_ch)
        elif pe == "rope":
            self.pos_embed = RotaryPositionEmbedding(self.num_patches, n_ch)

        self.blocks = nn.ModuleList(
            DiTBlock(n_ch, num_heads, mlp_ratio = mlp_ratio, attn_dropout = attn_dropout, drop_path_rate = mlp_dropout)
            for _ in range(n_blocks))
        self.final_layer = FinalLayer(n_ch, patch_size, self.output_ch)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize pos_embed by sin-cos embedding:
        # nn.init.trunc_normal_(self.pos_embed, std = 0.02)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std = 0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std = 0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor):
        """
        x: [batch_size, index, patch_size * patch_size * out_channels]
        imgs: [batch_size, C, H, W]
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        c = self.output_ch
        x = x.reshape([x.shape[0], h, w, p, p, c])  # (N, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)
        imgs = x.reshape([x.shape[0], c, h * p, w * p])  # (N, C, H, W)
        return imgs

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.x_embedder(x)
        x = self.pos_embed(x)
        t = self.t_embedder(t)
        for block in self.blocks:
            x = block(x, t)
        x = self.final_layer(x, t)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

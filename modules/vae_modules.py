import torch.nn as nn
from utils import *
from typing import Union, List, Tuple
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

device, local_rank = init_distributed()


class VAEResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_groups: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.norm1 = nn.GroupNorm(n_groups, in_ch)
        self.norm2 = nn.GroupNorm(n_groups, out_ch)
        self.act1 = nn.SiLU()
        self.act2 = nn.SiLU()
        self.act3 = nn.SiLU()
        if out_ch == in_ch:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 3, 1, 1)

    def forward(self, input: torch.Tensor):
        output = self.conv1(self.act1(self.norm1(input)))
        output = self.conv2(self.act3(self.norm2(output)))
        output += self.shortcut(input)
        return output


class VAEAttentionBlock(nn.Module):
    def __init__(self, n_ch: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        super().__init__()
        if d_k is None:
            d_k = n_ch
        self.norm = nn.GroupNorm(n_groups, n_ch)
        self.w_qkv = nn.Linear(n_ch, n_heads * d_k * 3)
        self.dense = nn.Linear(n_heads * d_k, n_ch)
        self.scale = d_k ** (-0.5)
        self.d_k = d_k
        self.n_heads = n_heads

    def forward(self, input: torch.Tensor):
        B, C, H, W = input.shape
        x = input.reshape(B, C, H * W).permute(0, 2, 1)  # [B, N, C]
        x_norm = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, N, C]

        qkv = self.w_qkv(x_norm)  # [B, N, 3 * H * D]
        qkv = qkv.view(B, -1, self.n_heads, 3 * self.d_k).permute(0, 2, 1, 3)  # [B, H, N, 3*D]
        q, k, v = torch.chunk(qkv, 3, dim = -1)  # [B, H, N, D]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        attn = attn.softmax(dim = -1)
        out = attn @ v  # [B, H, N, D]

        out = out.permute(0, 2, 1, 3).reshape(B, -1, self.n_heads * self.d_k)  # [B, N, H*D]
        out = self.dense(out) + x  # 残差连接

        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        return out

    # def build_2d_sincos_position_embedding(self, height, width, dim, device):
    #     grid_y, grid_x = torch.meshgrid(
    #         torch.arange(height, device=device),
    #         torch.arange(width, device=device),
    #         indexing="ij"
    #     )
    #     grid = torch.stack([grid_y, grid_x], dim=0).float()  # [2, H, W]
    #
    #     assert dim % 4 == 0, "n_channels must be divisible by 4 for 2D sin-cos position embedding"
    #     dim_each = dim // 2
    #     omega = torch.arange(dim_each // 2, device=device) / (dim_each // 2)
    #     omega = 1.0 / (10000 ** omega)
    #
    #     out_y = grid[0].flatten().unsqueeze(1) @ omega.unsqueeze(0)  # [H*W, dim//4]
    #     out_x = grid[1].flatten().unsqueeze(1) @ omega.unsqueeze(0)  # [H*W, dim//4]
    #
    #     pos_y = torch.cat([out_y.sin(), out_y.cos()], dim=1)
    #     pos_x = torch.cat([out_x.sin(), out_x.cos()], dim=1)
    #
    #     pos_emb = torch.cat([pos_y, pos_x], dim=1).unsqueeze(0)  # [1, H*W, dim]
    #     return pos_emb


class VAEDownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, has_attn: bool):
        super().__init__()
        self.res = VAEResidualBlock(in_ch, out_ch)
        if has_attn:
            self.attn = VAEAttentionBlock(out_ch)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.res(x)
        x = self.attn(x)
        return x


class VAEUpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, has_attn: bool):
        super().__init__()
        self.res = VAEResidualBlock(in_ch, out_ch)
        if has_attn:
            self.attn = VAEAttentionBlock(out_ch)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.res(x)
        x = self.attn(x)
        return x


class VAEUpSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor = 2.0)
        return self.conv(x)


class VAEDownSample(nn.Module):
    def __init__(self, n_ch: int):
        super().__init__()
        self.convT = nn.Conv2d(n_ch, n_ch, 3, 2, 1)

    def forward(self, x: torch.Tensor):
        x = self.convT(x)
        return x


class VAEMiddleBlock(nn.Module):
    def __init__(self, n_ch: int):
        super().__init__()
        self.res1 = VAEResidualBlock(n_ch, n_ch)
        self.attn = VAEAttentionBlock(n_ch)
        self.res2 = VAEResidualBlock(n_ch, n_ch)

    def forward(self, x: torch.Tensor):
        x = self.res1(x)
        x = self.attn(x)
        x = self.res2(x)
        return x


class VAE_Encoder(nn.Module):
    def __init__(self, input_ch: int, base_ch: int,
                 ch_mult: Union[List[int], Tuple[int]],
                 has_attn: Union[List[bool], Tuple[bool]],
                 latent_ch: int, n_blocks: int):
        super().__init__()

        if len(ch_mult) != len(has_attn):
            raise ValueError("ch_mult 和 has_attn 的长度必须一致")

        self.init_conv = nn.Conv2d(input_ch, base_ch, 3, 1, 1)
        self.blocks = nn.ModuleList()

        in_ch = base_ch
        for i, mult in enumerate(ch_mult):
            out_ch = base_ch * mult
            for _ in range(n_blocks):
                self.blocks.append(VAEDownBlock(in_ch, out_ch, has_attn[i]))
                in_ch = out_ch
            if i != len(ch_mult) - 1:
                self.blocks.append(VAEDownSample(out_ch))

        # 中间处理块
        self.blocks.append(VAEResidualBlock(in_ch, in_ch))
        self.blocks.append(VAEAttentionBlock(in_ch))
        self.blocks.append(VAEResidualBlock(in_ch, in_ch))
        self.blocks.append(nn.GroupNorm(8, in_ch))
        self.blocks.append(nn.SiLU())
        self.blocks.append(nn.Conv2d(in_ch, latent_ch, 3, 1, 1))

    def forward(self, x):
        x = self.init_conv(x)
        for block in self.blocks:
            x = block(x)
        return x


class VAE_Decoder(nn.Module):
    def __init__(self, output_ch: int, base_ch: int,
                 ch_mult: Union[List[int], Tuple[int]],
                 has_attn: Union[List[bool], Tuple[bool]],
                 latent_ch: int, n_blocks: int):
        super().__init__()

        if len(ch_mult) != len(has_attn):
            raise ValueError("ch_mult 和 has_attn 的长度必须一致")

        channels = [base_ch * mult for mult in ch_mult]
        in_ch = channels[-1]
        self.init_conv = nn.Conv2d(latent_ch, in_ch, 3, 1, 1)

        self.blocks = nn.ModuleList()
        self.blocks.append(VAEResidualBlock(in_ch, in_ch))
        if has_attn[-1]:
            self.blocks.append(VAEAttentionBlock(in_ch))
        self.blocks.append(VAEResidualBlock(in_ch, in_ch))

        for i in reversed(range(len(ch_mult))):
            out_ch = channels[i]
            for _ in range(n_blocks):
                self.blocks.append(VAEUpBlock(in_ch, out_ch, has_attn[i]))
                in_ch = out_ch
            if i != 0:
                self.blocks.append(VAEUpSample(in_ch))

        self.blocks.append(nn.GroupNorm(32, in_ch))
        self.blocks.append(nn.SiLU())
        self.blocks.append(nn.Conv2d(in_ch, output_ch, 3, 1, 1))
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.init_conv(x)
        for block in self.blocks:
            x = block(x)
        return self.act(x)



class IBQQuantizer(nn.Module):
    def __init__(self, emb_size: int, embedding_dim: int, temperature: float = 1.0, beta: float = 0.25,
                 lam_ent: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.emb_size = emb_size
        self.temperature = temperature
        self.beta = beta  # Commitment loss coefficient
        self.lam_ent = lam_ent  # Entropy regularization coefficient

        self.embedding = nn.Embedding(emb_size, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / emb_size, 1.0 / emb_size)

        self.enable_print = True

    def forward(self, x):
        # [B, C, H, W] -> [BHW, C]
        flat_x = x.permute(0, 2, 3, 1).contiguous()
        flat_x = flat_x.view(-1, self.embedding_dim)

        # IBQ: compute inner product logits
        logits = torch.matmul(flat_x, self.embedding.weight.T)  # [BHW, K]

        # temperature scaling
        soft_onehot = F.softmax(logits / self.temperature, dim = 1)  # [BHW, K]
        _, indices = soft_onehot.max(dim = 1)
        hard_onehot = F.one_hot(indices, num_classes = self.emb_size).float()  # [BHW, K]
        combined_onehot = hard_onehot - soft_onehot.detach() + soft_onehot  # 计算Ind 用于更新encoder

        quantized = torch.matmul(combined_onehot, self.embedding.weight)  # 计算z_q [BHW, D]
        quantized = quantized.view(x.shape)  # [B, C, H, W]

        codebook_loss = F.mse_loss(quantized.detach(), x)  # 用于更新codebook
        q_latent_loss = F.mse_loss(quantized, x.detach())  # 用于更新encoder

        entropy_loss = -torch.sum(soft_onehot * torch.log(soft_onehot + 1e-8), dim = 1).mean()  # 鼓励使用更多code
        loss = q_latent_loss + self.beta * codebook_loss + self.lam_ent * entropy_loss
        used_code_mask = F.one_hot(indices, num_classes = self.emb_size).bool().any(dim = 0)

        return quantized, loss, used_code_mask, indices


class FeatureMatchingDiscriminator(nn.Module):
    def __init__(self, input_ch = 3, base_ch = 64):
        super().__init__()
        self.features = nn.ModuleList([
            nn.Sequential(
                spectral_norm(nn.Conv2d(input_ch, base_ch, 4, 2, 1)),  # 64x64 -> 32x32
                nn.LeakyReLU(0.2, inplace = True)
            ),
            nn.Sequential(
                spectral_norm(nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1)),  # 32x32 -> 16x16
                nn.InstanceNorm2d(base_ch * 2),
                nn.LeakyReLU(0.2, inplace = True)
            ),
            nn.Sequential(
                spectral_norm(nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1)),  # 16x16 -> 8x8
                nn.InstanceNorm2d(base_ch * 4),
                nn.LeakyReLU(0.2, inplace = True)
            ),
            nn.Sequential(
                spectral_norm(nn.Conv2d(base_ch * 4, base_ch * 8, 4, 2, 1)),  # 8x8 -> 4x4
                nn.InstanceNorm2d(base_ch * 8),
                nn.LeakyReLU(0.2, inplace = True)
            )
        ])
        self.final = spectral_norm(nn.Conv2d(base_ch * 8, 1, 4, padding = 1))  # 4x4 -> 1x1

    def forward(self, x):
        fm_feats = []
        for layer in self.features:
            x = layer(x)
            fm_feats.append(x)
        score = self.final(x)
        return score, fm_feats


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        self.downsample = downsample
        self.learned_shortcut = in_ch != out_ch or downsample
        self.conv1 = spectral_norm(nn.Conv2d(in_ch, out_ch, 3, 1, 1))
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.norm1 = nn.InstanceNorm2d(in_ch)
        self.norm2 = nn.InstanceNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)

        if self.learned_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, 1, 0),
                nn.AvgPool2d(2) if downsample else nn.Identity()
            )
        else:
            self.shortcut = nn.Identity()

        self.pool = nn.AvgPool2d(2) if downsample else nn.Identity()

    def forward(self, x):
        x_short = self.shortcut(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.pool(x)
        return x + x_short


class Discriminator(nn.Module):
    def __init__(self, input_ch: int, basic_ch: int = 64, n_blocks: int = 4):
        super().__init__()
        layers = [nn.Conv2d(input_ch, basic_ch, 3, 1, 1)]  # initial stem conv
        in_ch = basic_ch
        for i in range(n_blocks):
            out_ch = min(in_ch * 2, 512)
            layers.append(ResBlock(in_ch, out_ch, downsample=True))
            in_ch = out_ch

        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(spectral_norm(nn.Conv2d(in_ch, 1, kernel_size=3, stride=1, padding=1)))  # PatchGAN output

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

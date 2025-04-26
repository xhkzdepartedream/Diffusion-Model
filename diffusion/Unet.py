from typing import Union, List, Tuple, Optional
import torch
import torch.nn as nn
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class TimeEmbedding(nn.Module):
    def __init__(self, ch_num: int):
        super().__init__()
        self.ch_num = ch_num
        self.dense1 = nn.Linear(ch_num // 4, ch_num)
        self.dense2 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(ch_num, ch_num)
        )

    def forward(self, t: torch.Tensor):  # t.shape = (batch_size,)
        emb_dim = self.ch_num // 8
        emb = math.log(10000) / (emb_dim - 1)
        emb = torch.exp(torch.arange(emb_dim, device = t.device) * (-emb))
        emb = t.reshape(-1, 1) @ emb.reshape(1, -1)  # shape = (batch_size, half_dim)
        emb = torch.cat((emb.sin(), emb.cos()), dim = 1)  # shape = (batch_size, 2*half_dim)
        emb = self.dense1(emb)  # shape = (batch_size, ch_num)
        emb = self.dense2(emb)
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_ch: int, n_groups: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.norm1 = nn.GroupNorm(n_groups, in_ch)
        self.norm2 = nn.GroupNorm(n_groups, out_ch)
        self.time_emb = nn.Linear(time_ch, out_ch)
        self.act1 = nn.SiLU()
        self.act2 = nn.SiLU()
        self.act3 = nn.SiLU()
        if out_ch == in_ch:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 3, 1, 1)

    def forward(self, input: torch.Tensor, t: torch.Tensor):
        output = self.conv1(self.act1(self.norm1(input)))
        output += self.time_emb(self.act2(t))[:, :, None, None]
        output = self.conv2(self.act3(self.norm2(output)))
        output += self.shortcut(input)
        return output


class AttentionBlock(nn.Module):
    def __init__(self, n_ch: int, n_heads: int = 1, d_k: int = None, n_groups: int = 16):
        super().__init__()
        if d_k is None:
            d_k = n_ch
        self.norm = nn.GroupNorm(n_groups, n_ch)
        self.w_qkv = nn.Linear(n_ch, n_heads * d_k * 3)
        self.dense = nn.Linear(n_heads * d_k, n_ch)
        self.scale = d_k ** (-0.5)
        self.d_k = d_k
        self.n_heads = n_heads

    def forward(self, input: torch.Tensor, t: Optional[torch.Tensor] = None):
        _ = t
        batch_size, n_ch, height, width = input.shape
        input = input.reshape(batch_size, n_ch, -1).permute(0, 2, 1)  # shape: [batch_size, index(h*w), n_channels]
        qkv = self.w_qkv(input).reshape(batch_size, -1, self.n_heads, self.d_k * 3)  # n_channels = d_k * n_heads
        q, k, v = torch.chunk(qkv, 3, dim = -1)  # shape: [batch_size, index, n_heads, d_k]
        attn = torch.einsum('bihd,bjhd->bijh', q, k) / self.scale
        attn = attn.softmax(dim = 2)
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        res = res.reshape(batch_size, -1, self.n_heads * self.d_k)
        res = self.dense(res) + input
        res = res.permute(0, 2, 1).reshape(batch_size, n_ch, height, width)
        return res


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_ch: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_ch, out_ch, time_ch)
        if has_attn:
            self.attn = AttentionBlock(out_ch)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_ch: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_ch + out_ch, out_ch, time_ch)
        if has_attn:
            self.attn = AttentionBlock(out_ch)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    def __init__(self, n_ch: int, time_ch: int):
        super().__init__()
        self.res1 = ResidualBlock(n_ch, n_ch, time_ch)
        self.attn = AttentionBlock(n_ch)
        self.res2 = ResidualBlock(n_ch, n_ch, time_ch)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class UpSample(nn.Module):
    def __init__(self, n_ch: int):
        super().__init__()
        self.convT = nn.ConvTranspose2d(n_ch, n_ch, 4, 2, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        _ = t
        x = self.convT(x)
        return x


class DownSample(nn.Module):
    def __init__(self, n_ch: int):
        super().__init__()
        self.convT = nn.Conv2d(n_ch, n_ch, 3, 2, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        _ = t
        x = self.convT(x)
        return x


class Unet(nn.Module):
    def __init__(self,
                 input_ch: int,
                 n_ch: int,
                 ch_mults: Union[Tuple[int, ...], List[int]],
                 is_attn: Union[Tuple[bool, ...], List[bool]],
                 n_blocks: int = 2):  # encoder/decoder下采样，每一层额外需要多少个block
        super().__init__()
        n = len(ch_mults)
        self.time_emb = TimeEmbedding(n_ch * 4)
        self.preprocessing = nn.Conv2d(input_ch, n_ch, 3, 1, 1)
        down = []
        in_ch = out_ch = n_ch
        for i in range(n):
            out_ch = in_ch * ch_mults[i]
            for j in range(n_blocks):
                down.append(DownBlock(in_ch, out_ch, n_ch * 4, is_attn[i]))
                in_ch = out_ch

            if (i < n - 1):
                down.append(DownSample(out_ch))
        self.down = nn.ModuleList(down)

        self.middle = MiddleBlock(out_ch, n_ch * 4)

        up = []
        for i in reversed(range(n)):
            out_ch = in_ch
            for j in range(n_blocks):
                up.append(UpBlock(in_ch, out_ch, n_ch * 4, is_attn[i]))
            out_ch = out_ch // ch_mults[i]
            up.append(UpBlock(in_ch, out_ch, n_ch * 4, is_attn[i]))
            in_ch = out_ch
            if i > 0:
                up.append(UpSample(out_ch))

        self.up = nn.ModuleList(up)
        self.norm = nn.Sequential(
            nn.GroupNorm(8, n_ch),
            nn.SiLU()
        )
        self.final = nn.Conv2d(n_ch, input_ch, 3, 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t = t.to(torch.float)
        t = self.time_emb(t)
        x = self.preprocessing(x)
        h = [x]
        for m in self.down:
            x = m(x, t)
            h.append(x)
        x = self.middle(x, t)
        for m in self.up:
            if isinstance(m, UpSample):
                x = m(x, t)
            else:
                s = h.pop()
                x = m(torch.cat((x, s), dim = 1), t)
        x = self.final(self.norm(x))
        return x


class testnet(nn.Module):
    def __init__(self,
                 input_ch: int,
                 n_ch: int,
                 ch_mults: Union[Tuple[int, ...], List[int]],
                 is_attn: Union[Tuple[bool, ...], List[bool]],
                 n_blocks: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(input_ch, 3, 3, 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        return self.conv1(x)

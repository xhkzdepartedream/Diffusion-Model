from modules.vae_modules import *
from utils import init_distributed


device, local_rank = init_distributed()


class VAE(nn.Module):
    def __init__(self, input_size: int, input_ch: int, base_ch: int, ch_mults: Union[List[int], Tuple[int]],
                 has_attn: Union[List[bool], Tuple[bool]], latent_ch: int, n_blocks: int):
        super().__init__()
        self.encoder = VAE_Encoder(input_ch, base_ch, ch_mults, has_attn, latent_ch, n_blocks)
        self.conv_stats = nn.Conv2d(latent_ch, 2 * latent_ch, kernel_size = 1)
        self.decoder = VAE_Decoder(input_ch, base_ch, ch_mults, has_attn, latent_ch, n_blocks)
        self.post_quant_conv = nn.Conv2d(latent_ch, latent_ch, 1)
        self.latent_size = input_size / 2 ** (len(ch_mults) - 1)

    def encode(self, x: torch.Tensor, deterministic = False):
        x = self.encoder(x)
        stats = self.conv_stats(x)
        mean, logvar = torch.chunk(stats, 2, dim = 1)

        logvar = torch.clamp(logvar, -30.0, 20.0)
        var = torch.exp(logvar)

        if deterministic:
            z = mean
        else:
            rand_noise = torch.randn_like(mean)
            z = mean + var.sqrt() * rand_noise
        z_scaled = z * 0.18215
        return z_scaled, mean, var

    def decode(self, z: torch.Tensor):
        x_ = self.post_quant_conv(z)
        x_ = self.decoder(x_)
        return x_

    def forward(self, x: torch.Tensor, deterministic = False):
        z, mean, var = self.encode(x, deterministic)
        x_ = self.decode(z)
        return x_, z, mean, var

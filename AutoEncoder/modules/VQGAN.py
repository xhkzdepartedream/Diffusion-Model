from diffusion_and_unet import *
from AutoEncoder.modules import *

device, local_rank = init_distributed()


class VQVAE(nn.Module):
    def __init__(self, input_size: int, input_ch: int, base_ch: int, ch_mults: Union[List[int], Tuple[int]],
                 has_attn: Union[List[bool], Tuple[bool]], latent_ch: int, n_blocks: int,
                 emb_size: int, temperature: float):
        super().__init__()
        self.encoder = VAE_Encoder(input_ch, base_ch, ch_mults, has_attn, latent_ch, n_blocks)
        self.quant_conv = nn.Sequential(nn.Conv2d(latent_ch, latent_ch, 1), nn.Tanh())
        self.decoder = VAE_Decoder(input_ch, base_ch, ch_mults, has_attn, latent_ch, n_blocks)
        self.codebook = IBQQuantizer(emb_size, latent_ch, temperature, lam_ent = 0.1)
        self.post_quant_conv = nn.Conv2d(latent_ch, latent_ch, 1)
        self.perc_module = PerceptualModule(mode = 'lpips', lpips_net = 'vgg')
        self.input_size = input_size
        self.latent_size = input_size / 2 ** (len(ch_mults) - 1)

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        x = self.quant_conv(x)
        z, vq_loss, mask, indices = self.codebook(x)
        x_ = self.post_quant_conv(z)
        x_ = self.decoder(x_)
        return x_, vq_loss, mask, indices

    def loss(self, x: torch.Tensor, x_: torch.Tensor, vq_loss: torch.Tensor,
             prec_loss_factor: float, vq_loss_factor: float, return_detail: bool = False):
        recon_loss = F.mse_loss(x, x_)
        perc_loss = self.perc_module(x, x_)
        total_loss = recon_loss + prec_loss_factor * perc_loss + vq_loss * vq_loss_factor
        if return_detail:
            return total_loss, recon_loss, perc_loss, vq_loss
        return total_loss

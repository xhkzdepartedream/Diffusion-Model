import torch

from AutoEncoder import *
from utils import init_distributed
from init_dataset import *
from torch.amp import autocast
from torch.nn.functional import relu, cosine_similarity, normalize
from torch.autograd import grad
import torch.nn.functional as F

device, local_rank = init_distributed()


def get_recon_factor(x: int, ch: int):
    if x <= 10:
        return 5
    elif x <= 40:
        return (5 + 10 / x * 80) * (ch / 16)
    elif x <= 60:
        # return 14.0 + 6 / 100 * (x - 100)
        return 12 * (ch / 16)
    else:
        return max(60 + x, 100)


def get_gan_factor(x: int, ch: int):
    if x <= 10:
        return 2 / 10 * x
    elif x <= 50:
        return 3
    elif x <= 100:
        return 2 + 2 / 50 * (x - 50)
    else:
        return 5


def get_perc_factor(x: int, ch: int):
    return max(6 + 4 / 100 * x, 70)


def get_kl_factor(x: int, ch: int):
    ret = 1e-4 + 3e-4 / 60 * (x - 40)
    if x <= 10:
        ret = 1e-4
    elif x <= 40:
        ret = 1e-4 + 2.2e-4 / 60 * (x - 40)
    elif x <= 100:
        ret = max(2e-4 + 5e-4 / 120 * (x - 40), 8e-4)
    elif x <= 140:
        ret = 3e-4 + 4e-4 / 120 * (x - 40)
    else:
        ret = 9e-4 + 12e-4 / 60 * (x - 140)
    return 3 * ret


# def get_kl_factor(x: int, ch: int):
#     ret = 0
#     if x <= 10:
#         ret = 5e-4
#     elif x <= 40:
#         ret = 6e-4 + 9e-4 / 30 * (x - 10)
#     elif x <= 100:
#         ret = 1.8e-3 + 3.2e-3 / 60 * (x - 60)
#     else:
#         ret = 1.2e-3
#     return ret


def get_vf_factor(x: int, ch: int):
    if x <= 10:
        return 5
    elif x <= 20:
        return 8
    elif x <= 40:
        return 10 + 7 / 20 * (x - 20)
    elif x <= 100:
        return 15 + 10 / 60 * (x - 40)
    else:
        return 20 + 10 / 50 * (x - 100)


def saturation(img):
    # 输入 (B, 3, H, W)
    max_val, _ = img.max(dim = 1)
    min_val, _ = img.min(dim = 1)
    return (max_val - min_val).mean()


class VAVAE_trainer:
    def __init__(self, dataset: Dataset, input_ch: int, channels: Union[List[int], Tuple[int]],
                 has_attn: Union[List[bool], Tuple[bool]], latent_ch: int, n_blocks: int, dis_basic_ch: int,
                 use_vf: bool, title: str):
        super().__init__()
        self.vae = VAE(input_ch, channels, has_attn, latent_ch, n_blocks)
        self.channels = 1024 / 2 ** len(channels)
        self.latent_ch = latent_ch
        self.title = title
        self.perc_module = PerceptualModule(mode = 'lpips', lpips_net = 'vgg').to(device)
        self.vf_module = VFloss_module(latent_ch).to(device) if use_vf else None
        self.vf_linear = nn.Linear(1024, latent_ch).to(device)
        self.discriminator = Discriminator(input_ch, dis_basic_ch).to(device)
        self.vaeopt = torch.optim.Adam(self.vae.parameters(), lr = 1e-4)
        self.disopt = torch.optim.Adam(self.discriminator.parameters(), lr = 3 * 1e-5)
        self.scaler = torch.amp.GradScaler('cuda')
        self.scheduler = CosineAnnealingWarmRestarts(
            self.vaeopt,
            T_0 = 20,
            T_mult = 1,
            eta_min = 1e-5
        )
        self.train_dataset = dataset
        self.datasampler = DistributedSampler(dataset)

    def _save_checkpoint(self, epoch: int):
        if dist.get_rank() == 0:
            checkpoint = {
                'epoch': epoch,
                'vqvae_state_dict': self.vae.state_dict(),
                'discri_state_dict': self.discriminator.state_dict(),
                'vaeopt_state_dict': self.vaeopt.state_dict(),
                'disopt_state_dict': self.disopt.state_dict(),
                'scaler_state_dict': self.scaler.state_dict(),
            }
            torch.save(checkpoint, f"vavae{int(self.channels)}c{self.latent_ch}d_{self.title}_{epoch}.pth")
            print(f"[INFO] VAVAE Checkpoint saved at epoch {epoch}.")

    def _load_checkpoint(self, path, rank: int):
        self.checkpoint_path = path
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(path, map_location = map_location)
        self.vae.load_state_dict(checkpoint['vqvae_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discri_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        print(f"[INFO]VAVAE Loaded checkpoint from {path}, starting at epoch {checkpoint['epoch'] + 1}")

    def cal_adaweight_gan(self, nll_loss, g_loss, last_layer):

        nll_grads = grad(nll_loss, last_layer, retain_graph = True)[0]
        g_grads = grad(g_loss, last_layer, retain_graph = True)[0]
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.1, 1e4).detach()
        return d_weight

    def cal_adaweight_vf(self, nll_loss, vf_loss, last_layer = None):
        nll_grads = grad(nll_loss, last_layer, retain_graph = True)[0]
        vf_grads = grad(vf_loss, last_layer, retain_graph = True)[0]
        vf_weight = torch.norm(nll_grads) / (torch.norm(vf_grads) + 1e-4)
        vf_weight = torch.clamp(vf_weight, 0.5, 1e8).detach()
        return vf_weight

    def train(self, epochs: int, batch_size: int, lr: float, get_recon_factor, get_kl_factor, get_perc_factor,
              get_gan_factor, get_vf_factor, color_factor: float, distmat_margin: float, cos_margin: float,
              distmat_weight: float, cos_weight: float, warmup: int, checkpoint_path: Optional[str] = None):
        for param_group in self.vaeopt.param_groups:
            param_group['lr'] = lr
        self.start_epoch = 1
        current_device = f'cuda:{torch.cuda.current_device()}'

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path, local_rank)

        self.dataloader = DataLoader(self.train_dataset, batch_size = batch_size, shuffle = False,
                                     num_workers = 32, pin_memory = True, sampler = self.datasampler)
        self.vae.train()
        self.discriminator.train()

        for epoch in range(self.start_epoch, epochs + 1):
            kl_factor = get_kl_factor(epoch, self.channels)
            perc_factor = get_perc_factor(epoch, self.channels)
            recon_factor = get_recon_factor(epoch, self.channels)
            gan_factor = get_gan_factor(epoch, self.channels)
            vf_factor = get_vf_factor(epoch, self.channels)
            if epoch <= warmup:
                gan_factor = 0
            disable_tqdm = (dist.get_rank() != 0)

            # 初始化loss统计变量
            total_perc_loss = 0.0
            total_recon_loss = 0.0
            total_kl_loss = 0.0
            total_gan_loss = 0.0
            total_vf_loss = 0.0
            total_d_loss = 0.0
            total_loss = 0.0
            total_color_loss = 0.0
            batch_count = 0

            for batch, _ in tqdm(self.dataloader, disable = disable_tqdm):
                x = batch.to(device)
                with autocast(device_type = current_device, dtype = torch.bfloat16):
                    x_, z, mean, var = self.vae(x)
                    z = z.reshape(z.shape[0], z.shape[1], -1).transpose(1, 2)
                    kl_loss = 0.5 * torch.sum(mean ** 2 + var - torch.log(var + 1e-8) - 1) + 0.01 * torch.norm(mean)
                    prec_loss = self.perc_module(x, x_)
                    recon_loss = F.mse_loss(x, x_)
                    joint_recon_loss = recon_loss + perc_factor * prec_loss
                    gan_loss = torch.mean(-self.discriminator(x_))

                    # VFloss
                    if self.vf_module is not None:
                        x_fea = self.vf_linear(self.vf_module(x))
                        x_fea_norm = normalize(x_fea, dim = 1)
                        z_norm = normalize(z, dim = 1)
                        a = torch.bmm(x_fea_norm, x_fea_norm.transpose(1, 2))
                        b = torch.bmm(z_norm, z_norm.transpose(1, 2))
                        diff = torch.abs(a - b)

                        vf_loss_mdms = relu(diff - distmat_margin).mean()
                        vf_loss_mcos = relu(1 - cos_margin - cosine_similarity(x_fea, z)).mean()
                        vf_loss = vf_loss_mdms * distmat_weight + vf_loss_mcos * cos_weight
                        # auto_vf_weight = self.cal_adaweight_vf(joint_recon_loss, vf_loss, self.vae.module.conv_stats.weight)
                        auto_vf_weight = 1.0
                    else:
                        auto_vf_weight = 0
                        vf_loss = torch.zeros([1]).to(device)

                    # auto_gan_weight = self.cal_adaweight_gan(joint_recon_loss, gan_loss, self.vae.module.conv_stats.weight)
                    auto_gan_weight = 1.0

                    loss = (recon_loss * recon_factor + kl_loss * kl_factor + prec_loss * perc_factor +
                            gan_loss * auto_gan_weight * gan_factor + vf_loss * auto_vf_weight * vf_factor)

                    real_logits = self.discriminator(x)
                    fake_logits = self.discriminator(x_.detach())
                    d_loss_real = torch.mean(torch.relu(1. - real_logits))
                    d_loss_fake = torch.mean(torch.relu(1. + fake_logits))
                    d_loss = d_loss_real + d_loss_fake

                self.vaeopt.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.vaeopt)

                self.disopt.zero_grad()
                self.scaler.scale(d_loss).backward()
                self.scaler.step(self.disopt)
                self.scaler.update()

                # 累加loss
                total_perc_loss += prec_loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                total_gan_loss += gan_loss.item()
                total_vf_loss += vf_loss.item()
                total_d_loss += d_loss.item()
                total_loss += loss.item()
                batch_count += 1

            # 只在主进程打印
            if dist.get_rank() == 0:
                print(f"\n[Epoch {epoch}] recon_loss: {total_recon_loss / batch_count:.4f} | "
                      f"perc_loss: {total_perc_loss / batch_count:.4f} | "
                      f"kl_loss: {total_kl_loss / batch_count:.4f} | "
                      f"gan_loss: {total_gan_loss / batch_count:.4f} | "
                      f"vf_loss: {total_vf_loss / batch_count:.4f} | "
                      f"d_loss: {total_d_loss / batch_count:.4f} | "
                      f"total_loss: {total_loss / batch_count:.4f}\n")

                # def ratio(x): return (x / avg_total_loss) * 100
                #
                # print(f"[Loss Ratio (%)] recon: {ratio(avg_recon_loss):.1f}% | "
                #       f"perc: {ratio(avg_perc_loss):.1f}% | "
                #       f"kl: {ratio(avg_kl_loss):.1f}% | "
                #       f"gan: {ratio(avg_gan_loss):.1f}% | "
                #       f"vf: {ratio(avg_vf_loss):.1f}%\n")

            if epoch % 10 == 0:
                self._save_checkpoint(epoch)

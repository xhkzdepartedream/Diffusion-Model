from AutoEncoder.modules import *
from diffusion_and_unet import *
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

device, local_rank = init_distributed()


def get_vq_loss_factor(x: int):
    if x <= 10:
        return 0.2
    elif x <= 40:
        return 0.25
    elif x <= 80:
        return 0.3
    else:
        return min(4 + 6 / 80 * (x - 100), 4)


def get_perc_loss_factor(x: int):
    if x <= 10:
        return 1
    elif x <= 40:
        return 1
    elif x <= 80:
        return 1.2
    else:
        return max(1 + (x - 60) / 40 * 0.3, 1.5)


def get_gan_loss_factor(x: int):
    if x <= 40:
        return 0.1 / 20 * max((x - 5), 0)
    elif x <= 50:
        return 0.2
    elif x <= 75:
        return 0.4
    elif x <= 100:
        return 0.7
    else:
        return 0.8


def get_recon_loss_factor(x: int):
    return 2.0


class VQGAN_trainer:
    def __init__(self, dataset: Dataset, input_size: int, input_ch: int, base_ch: int,
                 ch_mults: Union[List[int], Tuple[int]], has_attn: Union[List[bool], Tuple[bool]], latent_ch: int,
                 n_blocks: int,
                 emb_size: int, temperatue: float, basic_ch: int):
        super().__init__()
        # loss=recon_loss + prec_loss_factor*perc_loss + q_latent_loss+beta*e_latent_loss + lambda_*adv_loss
        self.vqvae = VQVAE(input_size, input_ch, base_ch, ch_mults, has_attn, latent_ch, n_blocks, emb_size, temperatue)
        self.latent_ch = latent_ch
        self.latent_size = self.vqvae.latent_size
        self.discriminator = Discriminator(input_ch, basic_ch)
        self.vaeopt = torch.optim.Adam(self.vqvae.parameters(), lr = 1e-4)
        self.disopt = torch.optim.Adam(self.discriminator.parameters(), lr = 1e-3)
        self.scaler = torch.amp.GradScaler('cuda')
        self.scheduler = CosineAnnealingWarmRestarts(
            self.vaeopt,
            T_0 = 20,
            T_mult = 1,
            eta_min = 1e-5
        )

        self.train_dataset = dataset
        self.datasampler = DistributedSampler(dataset)

        self.emb_size = emb_size

    def _save_checkpoint(self, epoch: int):
        if dist.get_rank() == 0:
            checkpoint = {
                'epoch': epoch,
                'vqvae_state_dict': self.vqvae.state_dict(),
                'discri_state_dict': self.discriminator.state_dict(),
                'vaeopt_state_dict': self.vaeopt.state_dict(),
                'disopt_state_dict': self.disopt.state_dict(),
                'scaler_state_dict': self.scaler.state_dict(),
            }
            torch.save(checkpoint, f"vqgan{int(self.latent_size)}c{self.latent_ch}d_epoch_{epoch}.pth")
            print(f"[INFO] VQGAN Checkpoint saved at epoch {epoch}.")

    def _load_checkpoint(self, path, rank: int):
        self.checkpoint_path = path
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}  # 把保存时的 0号GPU 映射到当前rank
        checkpoint = torch.load(path, map_location = map_location)
        self.vqvae.load_state_dict(checkpoint['vqvae_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discri_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        print(f"[INFO]VQGAN Loaded checkpoint from {path}, starting at epoch {checkpoint['epoch'] + 1}")

    def calculate_lambda(self, perceptual_loss, gan_loss):
        last_layer = self.vqvae.module.decoder.model[-1]
        last_layer_weight = last_layer.weight
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph = True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph = True)[0]
        lambda_ = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        lambda_ = torch.clamp(lambda_, 0, 1e4).detach()
        return 0.8 * lambda_

    def train(self, epochs: int, batch_size: int, lr: float, get_perc_loss, get_vq_loss,
              bata: float, get_gan_loss, enp_loss_factor: float, warmup: int,
              checkpoint_path: Optional[str] = None):

        self.vqvae.module.codebook.beta = bata
        self.vqvae.module.codebook.lam_ent = enp_loss_factor
        current_device = f'cuda:{torch.cuda.current_device()}'
        self.dataloader = DataLoader(self.train_dataset, batch_size = batch_size, shuffle = False,
                                     num_workers = 32, pin_memory = True, sampler = self.datasampler)
        vq_params = list(self.vqvae.module.codebook.embedding.parameters())
        main_params = [p for name, p in self.vqvae.module.named_parameters() if 'codebook.embedding' not in name]

        self.vaeopt = torch.optim.Adam([
            {'params': main_params, 'lr': lr},
            {'params': vq_params, 'lr': 1e-3}
        ])

        self.start_epoch = 1
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path, local_rank)

        for epoch in range(self.start_epoch, epochs + 1):

            self.vqvae.train()
            self.discriminator.train()

            total_d_loss_real = 0
            total_d_loss_fake = 0
            total_d_loss = 0
            total_adv_loss = 0
            total_recon_loss = 0
            total_perc_loss = 0
            total_vq_loss = 0
            total_total_loss = 0
            step_count = 0

            total_used_mask = torch.zeros(self.emb_size, dtype = torch.bool, device = device)

            if epoch <= warmup:
                temp_vq_factor = get_vq_loss(epoch)
                temp_gan_loss_factor = 0
            else:
                temp_vq_factor = get_vq_loss(epoch)
                temp_gan_loss_factor = get_gan_loss(epoch)

            perc_loss_factor = get_perc_loss(epoch)

            disable_tqdm = (dist.get_rank() != 0)
            for batch, _ in tqdm(self.dataloader, disable = disable_tqdm):
                x = batch.to(device)
                with torch.amp.autocast(device_type = current_device, dtype = torch.bfloat16):
                    x_, vq_loss, used_mask, indeces = self.vqvae(x)

                    total_loss, recon_loss, perc_loss, vq_loss = self.vqvae.module.loss(
                        x, x_, vq_loss, perc_loss_factor, temp_vq_factor, True)

                    real_logits = self.discriminator(x)
                    fake_logits = self.discriminator(x_.detach())

                    d_loss_real = torch.mean(torch.relu(1. - real_logits))
                    d_loss_fake = torch.mean(torch.relu(1. + fake_logits))
                    d_loss = d_loss_real + d_loss_fake

                self.disopt.zero_grad()
                self.scaler.scale(d_loss).backward()
                self.scaler.step(self.disopt)
                total_used_mask |= used_mask

                with torch.amp.autocast(device_type = current_device, dtype = torch.bfloat16):
                    adv_loss = self.discriminator(x_)
                    adv_loss = torch.mean(-adv_loss)
                    total_loss = total_loss + temp_gan_loss_factor * adv_loss

                self.vaeopt.zero_grad()
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.vaeopt)
                self.scaler.update()

                total_d_loss_real += d_loss_real.item()
                total_d_loss_fake += d_loss_fake.item()
                total_d_loss += d_loss.item()
                total_adv_loss += adv_loss.item()
                total_recon_loss += recon_loss.item()
                total_perc_loss += perc_loss.item()
                total_vq_loss += vq_loss.item()
                total_total_loss += total_loss.item()
                step_count += 1

            used_codes = total_used_mask.sum().item()
            usage_ratio = used_codes / self.emb_size

            if dist.get_rank() == 0:
                print(f"[Epoch {epoch}]")
                print(f"   [D] Real: {total_d_loss_real / step_count:.4f} | "
                      f"Fake: {total_d_loss_fake / step_count:.4f} | "
                      f"Total: {total_d_loss / step_count:.4f}")
                print(f"   [G] Adv: {total_adv_loss / step_count:.4f} | "
                      f"Recon: {total_recon_loss / step_count:.4f} | "
                      f"Perc: {total_perc_loss / step_count:.4f} | "
                      f"VQ: {total_vq_loss / step_count:.4f} | "
                      f"Total: {total_total_loss / step_count:.4f}")
                print(f"Usage ratio: {usage_ratio:.2%}")

            if epoch % 10 == 0 or epoch == warmup:
                self._save_checkpoint(epoch)

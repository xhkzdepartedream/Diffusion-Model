from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, cosine_similarity, normalize
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import autocast
from tqdm import tqdm
import os
from diffusers import AutoencoderKL
from itertools import chain

from diffusion_and_unet import init_distributed
from init_dataset import *  # Assuming init_dataset contains necessary dataset classes and transforms
from AutoEncoder.modules.perceptual_module import PerceptualModule
from AutoEncoder.modules.vfloss_module import VFloss_module

device, local_rank = init_distributed()


class AutoencoderKL_trainer:
    def __init__(self, dataset: Dataset, title: str, pretrained_model_name_or_path: Optional[str] = None):
        super().__init__()
        if pretrained_model_name_or_path:
            self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path)
        else:
            raise ValueError("pretrained_model_name_or_path must be provided for AutoencoderKL fine-tuning.")

        self.vae = self.vae.to(device)
        self.title = title
        self.perc_module = PerceptualModule(mode = 'lpips').to(device)

        latent_ch = self.vae.config.latent_channels
        self.vfloss_module = VFloss_module(latent_ch = latent_ch).to(device)
        self.vf_linear = nn.Linear(latent_ch, 1024).to(device)
        self.z_pool = nn.AdaptiveAvgPool1d(256).to(device)  # 256 is the number of ViT patch tokens

        self.optimizer = torch.optim.AdamW(
            chain(self.vae.parameters(), self.vf_linear.parameters()),
            lr = 1e-4
        )
        self.scaler = torch.amp.GradScaler('cuda')

        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0 = 20,
            T_mult = 1,
            eta_min = 1e-5
        )
        self.train_dataset = dataset
        self.datasampler = DistributedSampler(dataset)
        self.start_epoch = 1

        for param in self.perc_module.parameters():
            param.requires_grad = False
        for param in self.vfloss_module.parameters():
            param.requires_grad = False

    def _save_checkpoint(self, epoch: int):
        if dist.get_rank() == 0:
            checkpoint = {
                'epoch': epoch,
                'vae_state_dict': self.vae.state_dict(),
                'vf_linear_state_dict': self.vf_linear.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scaler_state_dict': self.scaler.state_dict(),
            }
            if isinstance(self.vae, torch.nn.parallel.DistributedDataParallel):
                self.vae.module.save_pretrained(f"./autoencoderkl_finetuned_{self.title}_{epoch}")
            else:
                self.vae.save_pretrained(f"./autoencoderkl_finetuned_{self.title}_{epoch}")
            torch.save(checkpoint, f"autoencoderkl_finetuned_{self.title}_{epoch}.pth")
            print(f"[INFO] AutoencoderKL Checkpoint saved at epoch {epoch}.")

    def _load_checkpoint(self, path: str, rank: int):
        if os.path.isdir(path):
            self.vae = AutoencoderKL.from_pretrained(path)
            self.vae = self.vae.to(device)
            print(f"[INFO] AutoencoderKL model loaded from directory {path}.")
            print(f"[WARNNING] Params for 'self.vf_linear' not found.")
        elif os.path.isfile(path) and path.endswith('.pth'):
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            checkpoint = torch.load(path, map_location = map_location)
            self.vae.load_state_dict(checkpoint['vae_state_dict'])
            self.vf_linear.load_state_dict(checkpoint['vf_linear_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            print(f"[INFO] AutoencoderKL Loaded checkpoint from {path}, starting at epoch {checkpoint['epoch'] + 1}")
        else:
            print(f"[WARNING] Invalid checkpoint path: {path}")

    def train(self, epochs: int, batch_size: int, lr: float,
              recon_factor: float = 1.0, kl_factor: float = 1e-6,
              perc_factor: float = 0.1, vf_factor: float = 0.1,
              distmat_margin: float = 0.1, cos_margin: float = 0.0,
              distmat_weight: float = 0.1, cos_weight: float = 0.9,
              warm_up:int = 5,checkpoint_path: Optional[str] = None):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        current_device = f'cuda:{torch.cuda.current_device()}'

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path, local_rank)

        self.dataloader = DataLoader(self.train_dataset, batch_size = batch_size, shuffle = False,
                                     num_workers = 32, pin_memory = True, sampler = self.datasampler)

        for epoch in range(self.start_epoch, epochs + 1):
            if epoch <= warm_up:
                for param in self.vae.parameters():
                    param.requires_grad = False
            else:
                for param in self.vae.parameters():
                    param.requires_grad = True

            self.vae.train()
            self.vf_linear.train()


            disable_tqdm = (dist.get_rank() != 0)

            total_recon_loss, total_kl_loss, total_perc_loss, total_vf_loss, total_loss = 0.0, 0.0, 0.0, 0.0, 0.0
            batch_count = 0

            for batch, _ in tqdm(self.dataloader, disable = disable_tqdm):
                x = batch.to(device)

                with autocast(device_type = current_device, dtype = torch.bfloat16):
                    posterior = self.vae.module.encode(x).latent_dist
                    z = posterior.sample()
                    reconstruction = self.vae.module.decode(z).sample

                    recon_loss = F.mse_loss(reconstruction, x)
                    kl_loss = posterior.kl().mean()
                    perc_loss = self.perc_module(x, reconstruction)

                    # VFLoss Calculation
                    x_fea = self.vfloss_module(x)  # [B,C,32,32]
                    z_reshaped = z.view(z.shape[0], z.shape[1], -1).permute(0, 2, 1)  # [B,1024,4]
                    z_pooled = self.z_pool(z_reshaped.permute(0, 2, 1)).permute(0, 2, 1)
                    z_projected = self.vf_linear(z_pooled)

                    x_fea_norm = normalize(x_fea, dim = 2)
                    z_projected_norm = normalize(z_projected, dim = 2)
                    a = torch.bmm(x_fea_norm, x_fea_norm.transpose(1, 2))
                    b = torch.bmm(z_projected_norm, z_projected_norm.transpose(1, 2))
                    diff = torch.abs(a - b)
                    vf_loss_mdms = relu(diff - distmat_margin).mean()
                    vf_loss_mcos = relu(1 - cos_margin - cosine_similarity(x_fea, z_projected, dim = 2)).mean()
                    vf_loss = vf_loss_mdms * distmat_weight + vf_loss_mcos * cos_weight

                    loss = (recon_loss * recon_factor +
                            kl_loss * kl_factor +
                            perc_factor * perc_loss +
                            vf_loss * vf_factor)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step(epoch + batch_count / len(self.dataloader))

                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                total_perc_loss += perc_loss.item()
                total_vf_loss += vf_loss.item()
                total_loss += loss.item()
                batch_count += 1

            if dist.get_rank() == 0:
                print(f"\n[Epoch {epoch}] recon_loss: {total_recon_loss / batch_count:.4f} | "
                      f"kl_loss: {total_kl_loss / batch_count:.4f} | "
                      f"perc_loss: {total_perc_loss / batch_count:.4f} | "
                      f"vf_loss: {total_vf_loss / batch_count:.4f} | "
                      f"total_loss: {total_loss / batch_count:.4f}\n")

            if epoch % 5 == 0:
                self._save_checkpoint(epoch)

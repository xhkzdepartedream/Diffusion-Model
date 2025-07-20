from typing import Optional
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import autocast
from tqdm import tqdm
from utils import load_model_from_checkpoint
from diffusers import AutoencoderKL

from utils import init_distributed
from data.init_dataset import *
from modules.perceptual_module import PerceptualModule

device, local_rank = init_distributed()


class AutoencoderKL_trainer_no_vfloss:
    def __init__(self, dataset: Dataset, title: str, pretrained_model_name_or_path: Optional[str] = None):
        super().__init__()
        if pretrained_model_name_or_path:
            self.vae = load_model_from_checkpoint(pretrained_model_name_or_path, "autoencoderkl", device)
        else:
            raise ValueError("pretrained_model_name_or_path must be provided for AutoencoderKL fine-tuning.")

        self.vae = self.vae.to(device)
        self.title = title
        self.perc_module = PerceptualModule(mode = 'lpips').to(device)

        self.optimizer = torch.optim.AdamW(
            self.vae.parameters(),
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

    def _save_checkpoint(self, epoch: int):
        if dist.get_rank() == 0:
            checkpoint = {
                'epoch': epoch,
                'vae_state_dict': self.vae.state_dict(),
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
        elif os.path.isfile(path) and path.endswith('.pth'):
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            checkpoint = torch.load(path, map_location = map_location)
            self.vae.load_state_dict(checkpoint['vae_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            print(f"[INFO] AutoencoderKL Loaded checkpoint from {path}, starting at epoch {checkpoint['epoch'] + 1}")
        else:
            print(f"[WARNING] Invalid checkpoint path: {path}")

    def train(self, epochs: int, batch_size: int, lr: float,
              recon_factor: float = 1.0, kl_factor: float = 1e-6,
              perc_factor: float = 0.1, checkpoint_path: Optional[str] = None):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        current_device = f'cuda:{torch.cuda.current_device()}'

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path, local_rank)

        self.dataloader = DataLoader(self.train_dataset, batch_size = batch_size, shuffle = False,
                                     num_workers = 32, pin_memory = True, sampler = self.datasampler)

        for epoch in range(self.start_epoch, epochs + 1):
            self.vae.train()
            disable_tqdm = (dist.get_rank() != 0)
            total_recon_loss, total_kl_loss, total_perc_loss, total_loss = 0.0, 0.0, 0.0, 0.0
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

                    loss = (recon_loss * recon_factor +
                            kl_loss * kl_factor +
                            perc_factor * perc_loss)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step(epoch + batch_count / len(self.dataloader))

                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                total_perc_loss += perc_loss.item()
                total_loss += loss.item()
                batch_count += 1

            if dist.get_rank() == 0:
                print(f"\n[Epoch {epoch}] recon_loss: {total_recon_loss / batch_count:.4f} | "
                      f"kl_loss: {total_kl_loss / batch_count:.4f} | "
                      f"perc_loss: {total_perc_loss / batch_count:.4f} | "
                      f"total_loss: {total_loss / batch_count:.4f}\n")

            if epoch % 5 == 0:
                self._save_checkpoint(epoch)

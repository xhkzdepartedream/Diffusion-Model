import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import Dataset
from torch_ema import ExponentialMovingAverage
from diffusion.DenoiseDiffusion import DenoiseDiffusion, CosineDenoiseDiffusion
from utils import *
from models.DiT import DiT
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import logging
from typing import Optional

device, local_rank = init_distributed()


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level = logging.INFO,
            format = '[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt = '%Y-%m-%d %H:%M:%S',
            handlers = [logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


class DiT_trainer:
    def __init__(self, dataset: Dataset, n_steps: int, batch_size: int, lr: float, n_epoch: int, use_cos: bool,
                 input_size: int, patch_size: int, input_ch: int, n_ch: int, n_blocks: int, title: str,
                 pe: str, num_heads: int = 16, mlp_ratio: int = 4, learn_sigma: bool = True, attn_dropout = 0.0,
                 mlp_dropout = 0.0):
        self.n_steps = n_steps
        self.input_ch = input_ch
        self.n_ch = n_ch
        self.batch_size = batch_size
        self.lr = lr
        self.n_epoch = n_epoch
        self.losses = []
        self.device = device
        self.learn_sigma = learn_sigma
        self.title = title
        self.train_dataset = dataset

        self.datasampler = DistributedSampler(dataset)
        self.dataloader = DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = False,
                                     num_workers = 32,
                                     pin_memory = True, sampler = self.datasampler)

        self.eps_model = DiT(input_size, patch_size, input_ch, n_ch, n_blocks, num_heads, mlp_ratio,
                             pe = pe, learn_sigma = learn_sigma, attn_dropout = attn_dropout,
                             mlp_dropout = mlp_dropout).to(device)

        if use_cos:
            self.diffusion = CosineDenoiseDiffusion(self.eps_model, self.n_steps, device)
        else:
            self.diffusion = DenoiseDiffusion(self.eps_model, self.n_steps, device, 'v_prediction')

        self.scaler = torch.amp.GradScaler()
        self.optimizer = torch.optim.AdamW(self.eps_model.parameters(), lr = self.lr)
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0 = 20,
            T_mult = 1,
            eta_min = 1e-5
        )

        self.ema = ExponentialMovingAverage(self.eps_model.parameters(), decay = 0.999)
        self.start_epoch = 1

    def _save_checkpoint(self, epoch: int):
        if dist.get_rank() == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.eps_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'scaler_state_dict': self.scaler.state_dict(),
            }
            torch.save(checkpoint, f"{self.title}_epoch_{epoch}.pth")
            print(f"[INFO] DIFFUSION MODEL Checkpoint saved at epoch {epoch}.")

    def _load_checkpoint(self, checkpoint_path: str, rank: int):
        self.checkpoint_path = checkpoint_path
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}  # 把保存时的 0号GPU 映射到当前rank

        checkpoint = torch.load(checkpoint_path, map_location = map_location)

        self.eps_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1

        print(f"[INFO][RANK {rank}] Loaded checkpoint from epoch {checkpoint['epoch']}.")

    def test(self):
        print(self.train_dataset[0])
        img, label = self.train_dataset[0]
        print(img.shape)

    def train(self, vlb_factor: float, checkpoint_path: Optional[str] = None):
        self.losses.clear()
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path, rank = local_rank)

        for epoch in range(self.start_epoch, self.n_epoch + 1):
            total_loss = 0
            total_mse = 0
            total_vlb = 0
            disable_tqdm = dist.get_rank() != 0

            for images in tqdm(self.dataloader, disable = disable_tqdm):
                images = images.to(self.device)
                self.optimizer.zero_grad()

                with torch.amp.autocast('cuda'):
                    if vlb_factor != 0.0:
                        loss, simple, vlb = self.diffusion.loss(images, self.learn_sigma, vlb_factor)
                    else:
                        loss = self.diffusion.loss(images, self.learn_sigma, vlb_factor)

                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.eps_model.parameters(), max_norm = 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.ema.update()
                self.scheduler.step(epoch + len(self.losses) / len(self.dataloader))
                total_loss += loss.item()
                if vlb_factor != 0.0:
                    total_mse += simple.item()
                    total_vlb += vlb.item()

            self.losses.append(total_loss)
            if dist.get_rank() == 0:
                if vlb_factor != 0.0:
                    print(f"Epoch {epoch} | Loss: {total_loss:.4f} | MSE: {total_mse:.4f} | VLB: {total_vlb:.4f}")
                else:
                    print(f"Epoch {epoch} | Loss: {total_loss:.4f}")

            if epoch % 100 == 0:
                self._save_checkpoint(epoch)

        plt.plot(range(self.start_epoch, self.start_epoch + len(self.losses)), self.losses)
        plt.title('Loss下降曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()

        return self.eps_model, self.losses

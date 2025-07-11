from torch.optim import Adam
from typing import Union, List, Tuple
from Transformer import TransformerModel
import torch.distributed as dist
from utils import init_distributed
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn.functional as F
import torch

device, local_rank = init_distributed()

class Transformer_trainer():
    def __init__(self, index_dataset: Dataset, vocab_params: Union[List[int], Tuple[int]]):

        self.codebook_size, self.sos_token_id, self.eos_token_id, self.vocab_size = vocab_params
        self.model = TransformerModel(self.vocab_size).to(device)
        self.index_dataset = index_dataset
        self.index_datasampler = DistributedSampler(index_dataset)
        self.scaler = GradScaler()
        self.start_epoch = 1


    def _save_checkpoint(self, epoch: int):
        if dist.get_rank() == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.module.state_dict(),
                'scaler_state_dict': self.scaler.state_dict(),
            }
            torch.save(checkpoint, f"transformer_epoch_{epoch}.pth")
            print(f"[INFO] TRANSFORMER Checkpoint saved at epoch {epoch}.")


    def _load_checkpoint(self, path, rank: int):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(path, map_location = map_location)
        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        print(f"[INFO] Loaded checkpoint from {path}, starting at epoch {self.start_epoch}")


    def train(self, epochs: int, batch_size: int, lr: float):
        index_dataloader = DataLoader(self.index_dataset, batch_size = batch_size,
                                      shuffle = False, num_workers = 8,
                                      pin_memory = True, sampler = self.index_datasampler)
        self.optimizer = Adam(self.model.parameters(), lr)
        for epoch in range(self.start_epoch, epochs + 1):
            self.index_datasampler.set_epoch(epoch)
            self.model.train()
            total_loss = 0.0

            for batch in tqdm(index_dataloader):
                # [B, H, W] -> [B, L]
                batch = batch.view(batch.size(0), -1).to(device)  # token 序列
                sos = torch.full((batch.size(0), 1), self.sos_token_id, dtype = torch.long, device = device)
                eos = torch.full((batch.size(0), 1), self.eos_token_id, dtype = torch.long, device = device)

                input_seq = torch.cat([sos, batch], dim = 1)
                target_seq = batch

                self.optimizer.zero_grad()

                with autocast():
                    logits = self.model(input_seq)
                    logits = logits[:, 1:, :]
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_seq.view(-1))

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()

            if dist.get_rank() == 0:
                print(f"[Epoch {epoch}] Average Loss: {total_loss / len(index_dataloader):.4f}")
                if epoch % 25 == 0:
                    self._save_checkpoint(epoch)

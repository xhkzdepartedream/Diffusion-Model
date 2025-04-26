from .Unet import *
from .ddpm import DenoiseDiffusion
from .utils import *
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_ema import ExponentialMovingAverage

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class trainer:
    def __init__(self, dataset: Dataset, checkpoint_path: Optional[str] = None):
        self.n_steps = 400
        self.input_ch = 3
        self.n_ch = 64
        self.ch_mults = [1, 2, 2, 4]
        self.has_attn = [False, False, True, True]
        self.batch_size = 128
        self.lr = 3 * 1e-5
        self.n_epoch = 300
        self.losses = []
        self.device = device

        self.train_dataset = dataset
        self.dataloader = DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True, num_workers = 8, pin_memory = True)
        self.eps_model = Unet(self.input_ch, self.n_ch, self.ch_mults, self.has_attn, n_blocks = 1).to(device)
        # if device != 'cpu':
        #     self.eps_model = nn.DataParallel(self.eps_model, device_ids = [0, 1])
        self.test_model = testnet(self.input_ch, self.n_ch, self.ch_mults, self.has_attn).to(device)
        self.diffusion = DenoiseDiffusion(self.eps_model, self.n_steps, device)
        self.scaler = torch.cuda.amp.GradScaler()
        self.ema = None  # 训练时初始化
        self.optimizer = None
        self.start_epoch = 1
        if checkpoint_path is not None:
            self.checkpoint_path = checkpoint_path

    def save_checkpoint(self, epoch: int):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.eps_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
        }
        torch.save(checkpoint, f"model_epoch_{epoch}.pth")
        print(f"[INFO] Checkpoint saved at epoch {epoch}.")

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path)
        self.eps_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        print(f"[INFO] Loaded checkpoint from epoch {checkpoint['epoch']}.")

    def test(self):
        print(self.train_dataset[0])
        img, label = self.train_dataset[0]
        print(img.shape)

    def train(self, is_train: bool, loss_type: str = 'eps', resume: bool = False):
        self.losses.clear()
        self.optimizer = torch.optim.Adam(self.eps_model.parameters(), lr = self.lr)
        if is_train:
            self.ema = ExponentialMovingAverage(self.eps_model.parameters(), decay = 0.999)

        if resume:
            self.load_checkpoint()

        assert len(self.ch_mults) == len(self.has_attn)

        for epoch in range(self.start_epoch, self.n_epoch + 1):
            total_loss = 0
            for images in tqdm(self.dataloader):
                images = images.to(self.device)
                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    loss = self.diffusion.loss(images, loss_type = loss_type)

                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.eps_model.parameters(), max_norm = 1.1)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if is_train:
                    self.ema.update()

                total_loss += loss.item()

            self.losses.append(total_loss)
            print(f"Epoch {epoch} | Loss: {total_loss:.4f}")

            if is_train and epoch % 25 == 0:
                self.save_checkpoint(epoch)

        plt.plot(range(self.start_epoch, self.start_epoch + len(self.losses)), self.losses)
        plt.title('Loss下降曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()

        return self.eps_model, self.losses

import os

import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

from .Unet import *
from .denoisediffusion import DenoiseDiffusion
from .utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class sampler:
    def __init__(self, eps_model):
        self.n_steps = 1000
        self.shape = (1, 3, 64, 64)
        self.device = device
        self.eps_model = eps_model
        # self.eps_model = nn.DataParallel(self.eps_model, device_ids = [0, 1])
        self.diffusion = DenoiseDiffusion(self.eps_model, self.n_steps, device)

    @torch.no_grad()
    def show_tensor_image(image: torch.Tensor):
        # 若是 batch（NCHW），取第一个图像
        if image.dim() == 4:
            image = image[0]
        # 反归一化：[-1, 1] → [0, 1] → [0, 255]
        reverse_transforms = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / 2),  # [-1,1] → [0,1]
            transforms.Lambda(lambda t: t.clamp(0, 1)),  # 防止超出 [0,1]
            transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW → HWC
            transforms.Lambda(lambda t: (t * 255).numpy().astype(np.uint8)),  # → uint8
        ])
        image = reverse_transforms(image)
        plt.imshow(image)
        plt.axis("off")
        plt.show()

    @torch.no_grad()
    def visualize_denoising_process(self, steps: int, shape: Tuple[int, int, int, int],
                                    title: str = 'denoising_process2'):
        path = './visuals/'
        os.makedirs(path, exist_ok = True)

        # ===== EMA 权重切换 =====
        if hasattr(self, 'ema'):
            self.ema.store()
            self.ema.copy_to(self.eps_model)

        with torch.no_grad():
            xt = torch.randn(shape, device = self.device)
            images = [xt.clone()]
            # assert xt.device == self.eps_model.device

            for t in reversed(range(steps)):
                t_batch = torch.full((shape[0],), t, device = self.device, dtype = torch.long)
                xt = self.diffusion.p_sample(xt, t_batch)
                images.append(xt.clone())

            images = [(img + 1) / 2 for img in images]
            images = [img.clamp(0.0, 1.0).cpu() for img in images]

            interval = max(steps // 5, 1)
            selected_indices = list(range(0, steps + 1, interval))
            selected_images = [images[i] for i in selected_indices]

            fig, axes = plt.subplots(1, len(selected_images), figsize = (3 * len(selected_images), 3))
            if len(selected_images) == 1:
                axes = [axes]

            for i, ax in enumerate(axes):
                img = selected_images[i][0]
                img = img.permute(1, 2, 0).numpy()
                img = (img * 255).astype(np.uint8)

                if img.shape[-1] == 1:
                    img = img.squeeze(-1)

                ax.imshow(img, cmap = 'gray' if img.ndim == 2 else None)
                ax.axis('off')
                ax.set_title(f"t={steps - selected_indices[i]}")

            plt.tight_layout()
            plt.savefig(os.path.join(path, f"{title}.png"), bbox_inches = 'tight')

        # ===== EMA 权重恢复 =====
        if hasattr(self, 'ema'):
            self.ema.restore()

    @torch.no_grad()
    def visualize_ddim_process(self, steps: int, leap_steps_num: int, shape: Tuple[int, int, int, int],
                               title: str = 'ddim_process', eta: float = 0.0, xt: Optional[torch.Tensor] = None):
        import os
        import matplotlib.pyplot as plt
        import numpy as np

        path = './visuals/'
        os.makedirs(path, exist_ok = True)

        if xt is None:
            xt = torch.randn(shape, device = device)
        images = [xt.clone()]

        time_seq = torch.linspace(steps - 1, 0, leap_steps_num, device = device).long()

        for i in range(leap_steps_num - 1):
            t = time_seq[i]
            t_next = time_seq[i + 1]
            xt = self.diffusion.ddim_sample(xt, t.unsqueeze(0), t_next.unsqueeze(0), eta)
            images.append(xt.clone())

        images = [(img + 1) / 2 for img in images]
        images = [img.clamp(0.0, 1.0).cpu() for img in images]

        # 选取部分节点画图
        interval = max(leap_steps_num // 5, 1)
        selected_indices = list(range(0, leap_steps_num, interval)) + [leap_steps_num - 1]
        selected_images = [images[i] for i in selected_indices]

        fig, axes = plt.subplots(1, len(selected_images), figsize = (3 * len(selected_images), 3))
        if len(selected_images) == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            img = selected_images[i][0]
            img = img.permute(1, 2, 0).numpy()
            img = (img * 255).astype(np.uint8)

            if img.shape[-1] == 1:
                img = img.squeeze(-1)

            ax.imshow(img, cmap = 'gray' if img.ndim == 2 else None)
            ax.axis('off')
            ax.set_title(f"t={time_seq[selected_indices[i]].item()}")

        plt.tight_layout()
        plt.savefig(os.path.join(path, f"{title}.png"), bbox_inches = 'tight')
        print(f'[INFO] Task:[title={title},eta={eta}] has done.')

import numpy as np
from torchvision import transforms

from .Unet import *
from .DenoiseDiffusion import DenoiseDiffusion, CosineDenoiseDiffusion
from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class sampler:
    def __init__(self, eps_model, device, use_cos: bool):
        self.n_steps = 1000
        self.shape = (1, 3, 64, 64)
        self.device = device
        self.eps_model = eps_model
        if hasattr(eps_model, 'learn_sigma'):
            self.learn_sigma = eps_model.learn_sigma
        else:
            self.learn_sigma = False
        # self.eps_model = nn.DataParallel(self.eps_model, device_ids = [0, 1])
        if use_cos:
            self.diffusion = CosineDenoiseDiffusion(self.eps_model, self.n_steps, device = device)
        else:
            self.diffusion = DenoiseDiffusion(self.eps_model, self.n_steps, device = device, prediction_type = 'v_prediction')

    @torch.no_grad()
    def show_tensor_image(self, image: torch.Tensor):
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
    def sample_x0(self, shape: Union[Tuple[int], List[int]],
                  steps: int = 500,
                  use_ddim: bool = False,
                  leap_steps_num: int = 50,
                  eta: float = 0.0,
                  xt: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        返回最终采样得到的 x0 图像 tensor（范围在 [-1,1]）

        参数:
            shape: 生成图像的形状 (B, C, H, W)
            steps: DDPM 采样步数（仅用于 DDPM）
            use_ddim: 是否使用 DDIM 采样
            leap_steps_num: DDIM 的采样步数（如使用 DDIM）
            eta: DDIM 的控制参数
            xt: 初始噪声（可选，若为 None 则自动采样）

        返回:
            x0: 采样完成后的图像 tensor，形状为 (B, C, H, W)，范围 [-1, 1]
        """
        import torch

        # ===== EMA 权重切换 =====
        if hasattr(self, 'ema'):
            self.ema.store()
            self.ema.copy_to(self.eps_model)

        xt = xt if xt is not None else torch.randn(shape, device = self.device)

        if not use_ddim:
            for t in reversed(range(steps)):
                t_batch = torch.full((shape[0],), t, device = self.device, dtype = torch.long)
                xt = self.diffusion.p_sample(xt, t_batch, self.learn_sigma)
            x0 = xt
        else:
            time_seq = torch.linspace(steps - 1, 0, leap_steps_num, device = self.device).long()
            for i in range(leap_steps_num - 1):
                t = time_seq[i]
                t_next = time_seq[i + 1]
                xt = self.diffusion.ddim_sample(xt, t.unsqueeze(0), t_next.unsqueeze(0), eta)
            x0 = xt

        # ===== EMA 权重恢复 =====
        if hasattr(self, 'ema'):
            self.ema.restore()

        return x0

    @torch.no_grad()
    def visualize_denoising_process(self, steps: int, shape: Tuple[int, int, int, int],
                                    title: str = 'denoising_process2'):
        path = './visuals_dm/'
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
                xt = self.diffusion.p_sample(xt, t_batch, learn_sigma = self.learn_sigma)
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

        path = './visuals_dm/'
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

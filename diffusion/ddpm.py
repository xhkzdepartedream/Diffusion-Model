from typing import Tuple, Optional
import torch
import torch.nn.functional as F
import torch.nn as nn
from .utils import gather

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DenoiseDiffusion:
    def __init__(self, model: nn.Module, n_steps: int, device):
        super().__init__()
        self.beta = torch.linspace(1e-4, 0.02, n_steps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim = 0)
        self.n_steps = n_steps
        self.sigma2 = self.beta
        self.model = model

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x0 -> xt.mu xt.sigma
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)
        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        # x0 -> xt
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        xt = mean + var ** 0.5 * eps
        return xt

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        # xt -> xt-1
        eps_theta = self.model(xt, t)
        alpha_bar = gather(self.alpha_bar, t)
        alpha = gather(self.alpha, t)
        mean = 1 / (alpha ** 0.5) * (xt - (1.0 - alpha) / ((1.0 - alpha_bar) ** 0.5) * eps_theta)
        var = gather(self.sigma2, t)
        eps = torch.randn_like(xt, device = xt.device)
        return mean + (var ** 0.5) * eps

    def ddim_sample(self, xt: torch.Tensor, t: torch.Tensor, tgt_step: torch.Tensor, eta: float = 0.0):
        # xt -> x0
        eps_theta = self.model(xt, t)
        eps = torch.randn_like(eps_theta)
        alpha_bar_t = gather(self.alpha_bar, t)
        alpha_bar_tgt = gather(self.alpha_bar, tgt_step)
        x0_hat = (xt - (1 - alpha_bar_t) ** 0.5 * eps_theta) / (alpha_bar_t ** 0.5)
        x_tgt = alpha_bar_tgt ** 0.5 * x0_hat + (1 - alpha_bar_tgt) ** 0.5 * (
                    eta * eps + (1 - eta ** 2) ** 0.5 * eps_theta)
        return x_tgt

    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None, loss_type: str = 'eps'):
        """
        :param x0: 原始图像
        :param noise: 加噪的噪声，若为None则自动生成
        :param loss_type: 可选 'eps', 'eps_weighted', 'v'
                          - 'eps': 标准DDPM的MSE损失 (默认)
                          - 'eps_weighted': 加权版 eps 预测损失，乘 sqrt(1 - alpha_bar)
                          - 'v': 使用v预测的损失（乘 alpha_bar）
        :return: loss
        """
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device = x0.device, dtype = torch.long)

        if noise is None:
            noise = torch.randn_like(x0).to(x0.device)

        xt = self.q_sample(x0, t, noise)
        eps_theta = self.model(xt, t)  # 预测噪声

        if loss_type == 'eps':
            return F.mse_loss(eps_theta, noise)

        alpha_bar = gather(self.alpha_bar, t)

        if loss_type == 'eps_weighted':
            weight = torch.sqrt(1 - alpha_bar)
            return F.mse_loss(weight * eps_theta, weight * noise)

        if loss_type == 'v':
            # v = sqrt(alpha_bar) * noise - sqrt(1 - alpha_bar) * x0
            return F.mse_loss(alpha_bar ** 0.5 * eps_theta, alpha_bar ** 0.5 * noise)

        raise ValueError(f"Unsupported loss_type '{loss_type}', choose from ['eps', 'eps_weighted', 'v']")

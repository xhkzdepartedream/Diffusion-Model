from typing import Tuple, Optional

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from utils import gather

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DenoiseDiffusion:
    def __init__(self, model: nn.Module, n_steps: int, device, prediction_type: str = 'epsilon'):
        super().__init__()
        self.beta = torch.linspace(1e-4, 0.02, n_steps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim = 0)
        self.n_steps = n_steps
        # padding alpha_bar_prev
        alpha_bar_prev = F.pad(self.alpha_bar[:-1], (1, 0), value = 1.0)
        self.posterior_variance = self.beta * (1. - alpha_bar_prev) / (1. - self.alpha_bar)
        self.sigma2 = self.posterior_variance
        self.model = model
        self.prediction_type = prediction_type

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

    def _predict_eps_from_v(self, v_pred: torch.Tensor, xt: torch.Tensor, t: torch.Tensor):
        """
        从 v-prediction 恢复 epsilon-prediction
        """
        alpha_bar_t = gather(self.alpha_bar, t)
        sqrt_alpha_bar = alpha_bar_t.sqrt()
        sqrt_one_minus_alpha_bar = (1 - alpha_bar_t).sqrt()
        eps = sqrt_one_minus_alpha_bar * xt + sqrt_alpha_bar * v_pred
        return eps

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, learn_sigma: bool):
        # xt -> xt-1
        if self.prediction_type == 'v_prediction' and learn_sigma:
            raise NotImplementedError("Learning sigma is not implemented for v-prediction")

        if learn_sigma:
            # Epsilon-prediction with learned sigma
            out = self.model(xt, t)
            eps_theta, log_sigma2 = out.chunk(2, dim = 1)

            alpha_bar = gather(self.alpha_bar, t)
            alpha = gather(self.alpha, t)

            coef = (1 - alpha) / (1 - alpha_bar).sqrt()
            mean = (1.0 / alpha.sqrt()) * (xt - coef * eps_theta)

            sigma = torch.exp(0.5 * log_sigma2)
            noise = torch.randn_like(xt)
            return mean + sigma * noise
        else:
            # Epsilon-prediction or v-prediction with fixed sigma
            if self.prediction_type == 'epsilon':
                eps_theta = self.model(xt, t)
            elif self.prediction_type == 'v_prediction':
                v_pred = self.model(xt, t)
                eps_theta = self._predict_eps_from_v(v_pred, xt, t)
            else:
                raise ValueError(f"Unknown prediction type: {self.prediction_type}")

            alpha_bar = gather(self.alpha_bar, t)
            alpha = gather(self.alpha, t)
            mean = 1 / (alpha ** 0.5) * (xt - (1.0 - alpha) / ((1.0 - alpha_bar) ** 0.5) * eps_theta)
            var = gather(self.sigma2, t)
            eps = torch.randn_like(xt, device = xt.device)
            return mean + (var ** 0.5) * eps

    def ddim_sample(self, xt: torch.Tensor, t: torch.Tensor, tgt_step: torch.Tensor, eta: float = 0.0):
        if self.prediction_type == 'epsilon':
            eps_theta = self.model(xt, t)
        elif self.prediction_type == 'v_prediction':
            v_pred = self.model(xt, t)
            eps_theta = self._predict_eps_from_v(v_pred, xt, t)
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        eps = torch.randn_like(eps_theta)
        alpha_bar_t = gather(self.alpha_bar, t)
        alpha_bar_tgt = gather(self.alpha_bar, tgt_step)
        x0_hat = (xt - (1 - alpha_bar_t) ** 0.5 * eps_theta) / (alpha_bar_t ** 0.5)
        x_tgt = alpha_bar_tgt ** 0.5 * x0_hat + (1 - alpha_bar_tgt) ** 0.5 * (
                eta * eps + (1 - eta ** 2) ** 0.5 * eps_theta)
        return x_tgt

    def loss(self, x0: torch.Tensor, learn_sigma: bool, vlb_factor: float = 0.0, noise: Optional[torch.Tensor] = None):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device = x0.device, dtype = torch.long)

        if noise is None:
            noise = torch.randn_like(x0)

        xt = self.q_sample(x0, t, noise)

        if self.prediction_type == 'epsilon':
            if learn_sigma:
                out = self.model(xt, t)
                eps_theta, log_var = out.chunk(2, dim = 1)
                log_var = log_var.clamp(min = -10.0, max = 10.0)
                simple = (eps_theta - noise) ** 2
                vlb = 0.5 * torch.exp(-log_var) * simple + 0.5 * log_var
                loss = simple + vlb_factor * vlb
                return loss.mean(), simple.mean(), vlb.mean()
            else:
                eps_theta = self.model(xt, t)
                return F.mse_loss(eps_theta, noise)
        elif self.prediction_type == 'v_prediction':
            if learn_sigma:
                raise NotImplementedError("Learning sigma is not implemented for v-prediction")

            alpha_bar_t = gather(self.alpha_bar, t)
            sqrt_alpha_bar = alpha_bar_t.sqrt()
            sqrt_one_minus_alpha_bar = (1 - alpha_bar_t).sqrt()
            v_target = sqrt_alpha_bar * noise - sqrt_one_minus_alpha_bar * x0

            v_pred = self.model(xt, t)
            return F.mse_loss(v_pred, v_target)
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")


class CosineDenoiseDiffusion:
    def __init__(self, model: nn.Module, n_steps: int, device, s: float = 0.008):
        super().__init__()
        self.n_steps = n_steps
        self.model = model
        self.device = device

        # --- cosine schedule ---
        self.alpha_bar = self._cosine_alpha_bar(n_steps, s)
        self.alpha = self.alpha_bar[1:] / self.alpha_bar[:-1]
        self.alpha = torch.cat([self.alpha_bar[:1], self.alpha])  # 补齐第0项
        self.beta = 1 - self.alpha
        # 根据 "Improved Denoising Diffusion Probabilistic Models" 论文的建议，对 beta 进行裁剪，防止其值过大导致不稳定
        self.beta = torch.clamp(self.beta, min = 0.0, max = 0.999)
        self.sigma2 = self.beta  # 可用于variance取值

    def _cosine_alpha_bar(self, n_steps: int, s: float) -> torch.Tensor:
        """
        余弦调度：直接定义累计 ᾱ_t，然后反推 α_t 和 β_t
        """
        steps = n_steps + 1
        x = torch.linspace(0, n_steps, steps, device = self.device)
        alpha_bar = torch.cos(((x / n_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]  # 归一化使 alpha_bar[0]=1
        return alpha_bar

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)
        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + var.sqrt() * eps

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, learn_sigma: bool):
        if learn_sigma:
            out = self.model(xt, t)
            eps_theta, log_sigma2 = out.chunk(2, dim = 1)
            alpha_bar = gather(self.alpha_bar, t)
            alpha = gather(self.alpha, t)
            coef = (1 - alpha) / (1 - alpha_bar).sqrt()
            mean = (1.0 / alpha.sqrt()) * (xt - coef * eps_theta)
            sigma = torch.exp(0.5 * log_sigma2)
            noise = torch.randn_like(xt)
            return mean + sigma * noise
        else:
            eps_theta = self.model(xt, t)
            alpha_bar = gather(self.alpha_bar, t)
            alpha = gather(self.alpha, t)
            mean = (1 / alpha.sqrt()) * (xt - ((1 - alpha) / (1 - alpha_bar).sqrt()) * eps_theta)
            var = gather(self.sigma2, t)
            eps = torch.randn_like(xt)
            return mean + var.sqrt() * eps

    def ddim_sample(self, xt: torch.Tensor, t: torch.Tensor, tgt_step: torch.Tensor, eta: float = 0.0):
        eps_theta = self.model(xt, t)
        eps = torch.randn_like(eps_theta)
        alpha_bar_t = gather(self.alpha_bar, t)
        alpha_bar_tgt = gather(self.alpha_bar, tgt_step)
        x0_hat = (xt - (1 - alpha_bar_t).sqrt() * eps_theta) / alpha_bar_t.sqrt()
        # 修复了 `torch.Tensor()` 导致的设备不匹配问题
        # The original code `torch.Tensor(1 - eta ** 2).sqrt()` creates a tensor on CPU, causing device mismatch.
        x_tgt = alpha_bar_tgt.sqrt() * x0_hat + (1 - alpha_bar_tgt).sqrt() * (
                eta * eps + ((1 - eta ** 2) ** 0.5) * eps_theta)
        return x_tgt

    def loss(self, x0: torch.Tensor, learn_sigma: bool, vlb_factor: float = 0.0, noise: Optional[torch.Tensor] = None):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device = x0.device, dtype = torch.long)

        if noise is None:
            noise = torch.randn_like(x0)

        xt = self.q_sample(x0, t, noise)

        if learn_sigma:
            out = self.model(xt, t)
            eps_theta, log_var = out.chunk(2, dim = 1)
            log_var = log_var.clamp(min = -10.0, max = 10.0)
            simple = (eps_theta - noise) ** 2
            vlb = 0.5 * torch.exp(-log_var) * simple + 0.5 * log_var
            loss = simple + vlb_factor * vlb
            return loss.mean(), simple.mean(), vlb.mean()
        else:
            eps_theta = self.model(xt, t)
            return F.mse_loss(eps_theta, noise)

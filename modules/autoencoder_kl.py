import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import init_distributed
import os
import warnings

device, local_rank = init_distributed()

# 尝试导入diffusers库
try:
    from diffusers import AutoencoderKL

    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    warnings.warn(
        "无法导入 diffusers.models.autoencoder_kl.AutoencoderKL。"
        "请先安装 diffusers 库: pip install diffusers"
    )


def get_pretrained_autoencoder_kl(
        pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4",
        subfolder = None,
        revision = None,
        force_upcast = False
):
    """
    返回一个预训练的AutoencoderKL模型。
    
    参数:
        pretrained_model_name_or_path (str): 预训练模型的名称或路径，默认为"stabilityai/sd-vae-ft-mse"
        subfolder (str, optional): 如果模型存储在子文件夹中，指定子文件夹名称
        revision (str, optional): 特定模型修订版本
        force_upcast (bool): 是否强制将权重上采样到float32，默认为False
        
    返回:
        AutoencoderKL: 预训练的VAE模型
    """
    if not DIFFUSERS_AVAILABLE:
        raise ImportError(
            "无法导入 diffusers.models.autoencoder_kl.AutoencoderKL。"
            "请先安装 diffusers 库: pip install diffusers"
        )

    try:
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path,
            subfolder = subfolder,
            revision = revision,
            torch_dtype = torch.float16 if not force_upcast else torch.float32
        )

        # 将模型移动到当前设备
        vae = vae.to(device)

        # 设置为评估模式
        vae.eval()

        return vae

    except Exception as e:
        print(f"加载预训练AutoencoderKL模型时出错: {e}")
        raise


def encode_images(vae, images, return_dict = False):
    """
    使用VAE编码器将图像编码为潜在表示。
    
    参数:
        vae: AutoencoderKL实例
        images (torch.Tensor): 形状为[B, C, H, W]的图像张量，值域为[-1, 1]
        return_dict (bool): 是否返回字典形式的结果
        
    返回:
        torch.Tensor 或 dict: 潜在表示或包含潜在表示的字典
    """
    if not DIFFUSERS_AVAILABLE:
        raise ImportError("diffusers库不可用")

    if not isinstance(vae, AutoencoderKL):
        raise TypeError("vae必须是AutoencoderKL的实例")

    with torch.no_grad():
        # 确保图像在正确的设备上
        if images.device != device:
            images = images.to(device)

        # 编码图像
        latents = vae.encode(images).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

    if return_dict:
        return {"latents": latents}
    return latents


def decode_latents(vae, latents, return_dict = False):
    """
    使用VAE解码器将潜在表示解码为图像。
    
    参数:
        vae: AutoencoderKL实例
        latents (torch.Tensor): 形状为[B, C, H, W]的潜在表示张量
        return_dict (bool): 是否返回字典形式的结果
        
    返回:
        torch.Tensor 或 dict: 解码后的图像或包含图像的字典，值域为[-1, 1]
    """
    if not DIFFUSERS_AVAILABLE:
        raise ImportError("diffusers库不可用")

    if not isinstance(vae, AutoencoderKL):
        raise TypeError("vae必须是AutoencoderKL的实例")

    with torch.no_grad():
        # 确保潜在表示在正确的设备上
        if latents.device != device:
            latents = latents.to(device)

        # 应用缩放因子
        latents = latents / vae.config.scaling_factor

        # 解码潜在表示
        images = vae.decode(latents).sample

    if return_dict:
        return {"images": images}
    return images


def save_vae_reconstruction(original_image, vae, save_path, filename = "reconstruction.png"):
    """
    保存原始图像及其VAE重建结果的对比图。
    
    参数:
        original_image (torch.Tensor): 形状为[1, C, H, W]的原始图像张量，值域为[-1, 1]
        vae: VAE模型实例
        save_path (str): 保存图像的目录路径
        filename (str): 保存的文件名
    """
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 确保输入是tensor
    if not isinstance(original_image, torch.Tensor):
        raise TypeError("original_image必须是torch.Tensor类型")

    # 使用VAE进行重建
    if hasattr(vae, "forward"):
        with torch.no_grad():
            # 判断VAE类型并进行适当的处理
            if DIFFUSERS_AVAILABLE and isinstance(vae, AutoencoderKL):
                # 直接使用diffusers的AutoencoderKL
                latents = vae.encode(original_image).latent_dist.sample()
                x_recon = vae.decode(latents).sample
            else:
                # 假设是项目中的VAE模型或AutoencoderKLWrapper
                x_recon, _, _, _ = vae(original_image)

            # 将图像转换为[0, 1]范围用于可视化
            original_vis = ((original_image.float() + 1.0) / 2.0).clamp(0.0, 1.0)
            recon_vis = ((x_recon.float() + 1.0) / 2.0).clamp(0.0, 1.0)

            # 创建网格
            grid = make_grid([original_vis[0], recon_vis[0]], nrow = 2)
            grid_np = grid.cpu().permute(1, 2, 0).numpy()

            # 保存图像
            plt.figure(figsize = (10, 5))
            plt.imshow(grid_np)
            plt.title("原始图像 vs 重建图像")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, filename))
            plt.close()
    else:
        raise TypeError("vae必须具有forward方法")


class AutoencoderKLWrapper(nn.Module):
    """
    对AutoencoderKL的封装，使其接口与项目中的VAE模型兼容。
    """

    def __init__(self, pretrained_model_name_or_path = "stabilityai/sd-vae-ft-mse", subfolder = None, revision = None):
        super().__init__()
        self.vae = get_pretrained_autoencoder_kl(
            pretrained_model_name_or_path = pretrained_model_name_or_path,
            subfolder = subfolder,
            revision = revision
        )
        self.scaling_factor = self.vae.config.scaling_factor

    def encode(self, x, deterministic = False):
        """
        编码图像，与项目中的VAE.encode接口兼容。
        
        参数:
            x (torch.Tensor): 输入图像
            deterministic (bool): 是否使用确定性编码（不使用随机采样）
            
        返回:
            z: 潜在表示
            mean: 均值（在AutoencoderKL中不可用，返回z）
            var: 方差（在AutoencoderKL中不可用，返回None）
        """
        with torch.no_grad():
            if deterministic:
                z = self.vae.encode(x).latent_dist.mode()
            else:
                z = self.vae.encode(x).latent_dist.sample()
            z = z * self.scaling_factor
            # AutoencoderKL不直接提供mean和var，为了兼容性返回z和None
            return z, z, None

    def decode(self, z):
        """
        解码潜在表示，与项目中的VAE.decode接口兼容。
        
        参数:
            z (torch.Tensor): 潜在表示
            
        返回:
            torch.Tensor: 解码后的图像
        """
        with torch.no_grad():
            z = z / self.scaling_factor
            return self.vae.decode(z).sample

    def forward(self, x, deterministic = False):
        """
        前向传播，与项目中的VAE.forward接口兼容。
        
        参数:
            x (torch.Tensor): 输入图像
            deterministic (bool): 是否使用确定性编码
            
        返回:
            x_: 重建图像
            z: 潜在表示
            mean: 均值（在AutoencoderKL中不可用，返回z）
            var: 方差（在AutoencoderKL中不可用，返回None）
        """
        z, mean, var = self.encode(x, deterministic)
        x_ = self.decode(z)
        return x_, z, mean, var


def get_compatible_vae_model(pretrained_model_name_or_path = "stabilityai/sd-vae-ft-mse", subfolder = None,
                             revision = None):
    """
    获取一个与项目中VAE模型接口兼容的预训练AutoencoderKL模型。
    
    参数:
        pretrained_model_name_or_path (str): 预训练模型的名称或路径
        subfolder (str, optional): 如果模型存储在子文件夹中，指定子文件夹名称
        revision (str, optional): 特定模型修订版本
        
    返回:
        AutoencoderKLWrapper: 封装的VAE模型
    """
    return AutoencoderKLWrapper(
        pretrained_model_name_or_path = pretrained_model_name_or_path,
        subfolder = subfolder,
        revision = revision
    )

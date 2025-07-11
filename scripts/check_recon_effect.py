
import os
import torch
from diffusers import AutoencoderKL
from AutoEncoder.modules.VA_VAE import VAE  # 假设您的自定义VAE在这个路径
from AutoEncoder.modules.VQGAN import VQVAE  # 假设您的VQGAN在这个路径
from utils import show_reconstructions, load_model_from_checkpoint

# --- 全局配置 ---
# 根据您的环境修改这些默认路径
DEFAULT_IMAGE_DIR = '/path/to/your/celeba_hq_256/'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(model_type, checkpoint_path):
    """根据参数加载指定的模型。"""
    model_type = model_type.lower()
    model = None

    print(f"[INFO] 正在加载模型: {model_type} 从 {checkpoint_path}")

    if model_type == 'vae':
        # --- 加载您的自定义VAE ---
        # 注意：您需要根据您的VAE模型定义来实例化它
        model_instance = VAE(input_size=256, input_ch=3, base_ch=128, ch_mults=[1, 1, 2, 2, 1],
                             has_attn=[False, False, True, False, False], latent_ch=32, n_blocks=2)
        model = load_model_from_checkpoint(checkpoint_path, 'vae', DEVICE, model_instance)

    elif model_type == 'autoencoderkl':
        # --- 加载预训练的AutoencoderKL ---
        model = AutoencoderKL.from_pretrained("/path/to/your/autoencoderkl/")
        model = load_model_from_checkpoint(checkpoint_path, 'autoencoderkl', DEVICE, model)

    elif model_type == 'vqgan':
        # --- 加载您的VQGAN ---
        # 同样，您需要根据您的VQGAN模型定义来实例化它
        model_instance = VQVAE(in_channels=3, ch_mults=[128, 160, 256, 512, 512],
                               attn_resolutions=[False, False, False, True, False],
                               z_channels=48, n_res_blocks=2, codebook_size=1024, beta=0.2)
        # 假设VQGAN也用'vae'键, 如果不是, 请在 utils.py 的 load_model_from_checkpoint 中修改
        model = load_model_from_checkpoint(checkpoint_path, 'vae', DEVICE, model_instance)

    else:
        raise ValueError(f"未知的模型类型: {model_type}")

    model.to(DEVICE)
    model.eval()
    print("[INFO] 模型加载完成并已设置为评估模式。")
    return model


def main(model_type,checkpoint_path):
    """主执行函数"""
    # --- 在这里修改你要测试的模型 --- #
    # 1. 选择模型类型: 'vae', 'autoencoderkl', 或 'vqgan'
    # 2. 设置模型权重的路径
    # checkpoint_path = "/path/to/your/autoencoderkl_directory/"  # for autoencoderkl
    # checkpoint_path = "/path/to/your/vavae.pth" # for vae or vqgan
    # --------------------------------- #

    # 其他配置
    image_dir = DEFAULT_IMAGE_DIR
    num_images = 8
    cuda_device_id = '0'

    # 设置CUDA设备
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device_id

    # 加载模型
    model = load_model(model_type, checkpoint_path)

    # 显示重构效果
    show_reconstructions(
        model,
        image_dir=image_dir,
        num_images=num_images,
        device=DEVICE,
        title_prefix=f"{model_type.upper()} "
    )


if __name__ == '__main__':
    main(model_type = 'autoencoderkl',checkpoint_path = "/path/to/your/autoencoderkl.pth")

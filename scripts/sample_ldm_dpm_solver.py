import torch
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, DDPMScheduler
from diffusion import Unet
from utils import load_model_from_checkpoint
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def show_tensor_image(image, num_images = 1, size = (3, 256, 256)):
    image_tensor = image.detach().cpu()
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow = 1)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def main(times: int, prediction_type: str):
    # 1. 加载模型
    # 注意：请将下面的路径替换为你自己的模型权重的本地路径
    unet_checkpoint_path = "/data1/yangyanliang/Diffusion-Model/unet2_epoch_100.pth"
    vae_checkpoint_path = "./autoencoderkl_finetuned_celeba_hq2_10/"

    unet_model = Unet(4, 128, [1, 1, 2, 2, 1], [False, False, True, True, False], 3).to(device)
    unet_model = load_model_from_checkpoint(unet_checkpoint_path, 'unet', device, unet_model)

    vae = load_model_from_checkpoint(vae_checkpoint_path, 'autoencoderkl', device)
    vae.eval()

    print(f"[INFO] Using prediction type: {prediction_type}")
    print("[INFO] Model loading completed.")

    # 2. 设置 DPM-Solver++ scheduler
    # DDPMScheduler 用于提供 beta schedule 配置
    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps = 800,
        beta_start = 1e-4,
        beta_end = 0.02,
        beta_schedule = "linear"
    )
    
    scheduler = DPMSolverMultistepScheduler.from_config(ddpm_scheduler.config)
    # 关键：根据要使用的模型类型，设置 scheduler 的预测类型
    scheduler.config.prediction_type = prediction_type

    # 3. 统一的采样过程
    for i in range(times):
        shape = [1, 4, 32, 32]
        latents = torch.randn(shape, device = device)
        scheduler.set_timesteps(num_inference_steps = 20, device = device)
        
        for t in scheduler.timesteps:
            # 将 timestep 扩展到 batch size
            timestep_batch = torch.tensor([t], device = device).repeat(latents.shape[0])
            
            with torch.no_grad():
                # 模型输出 model_output 可能是 noise_pred 或 v_pred
                model_output = unet_model(latents, timestep_batch)

            # scheduler.step 会根据 prediction_type 自动处理 model_output
            latents = scheduler.step(model_output, t, latents).prev_sample

        # 4. 解码和可视化
        # z = z / 0.18215 # 如果你的 VAE 需要这个缩放
        with torch.no_grad():
            image = vae.decode(latents).sample

        show_tensor_image(image)


if __name__ == '__main__':
    # 你可以根据你的模型来选择使用哪种预测方式
    # print("--- Sampling with epsilon prediction ---")
    # main(times=4, prediction_type='epsilon')
    
    print("--- Sampling with v_prediction ---")
    main(times=8, prediction_type='v_prediction')
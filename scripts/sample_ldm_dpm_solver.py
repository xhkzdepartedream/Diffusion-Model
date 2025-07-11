import torch
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, DDPMScheduler
from diffusion_and_unet import Unet
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


def main(times:int):
    # 1. 加载模型
    eps_model = Unet(4, 128, [1, 1, 2, 2, 1], [False, False, True, True, False], 3).to(device)
    eps_model = load_model_from_checkpoint(
        "/path/to/your/unet.pth", 'unet',
        device, eps_model)

    vae = load_model_from_checkpoint(
        "/path/to/your/autoencoderkl/", 'autoencoderkl',
        device)
    vae.eval()

    print("[INFO] Model loading completed.")

    # 2. 设置 DPM-Solver++ scheduler
    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps = 800,
        beta_start = 1e-4,
        beta_end = 0.02,
        beta_schedule = "linear"
    )
    scheduler = DPMSolverMultistepScheduler.from_config(ddpm_scheduler.config)

    # 3. 采样过程
    for i in range(times):
        shape = [1, 4, 32, 32]
        latents = torch.randn(shape, device = device)
        scheduler.set_timesteps(num_inference_steps = 20, device = device)
        for t in scheduler.timesteps:
            # expand the timestep to the batch size
            timestep_batch = torch.tensor([t], device=device).repeat(latents.shape[0])
            with torch.no_grad():
                noise_pred = eps_model(latents, timestep_batch)

            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # 4. 解码和可视化
        # z = z / 0.18215 # 如果你的 VAE 需要这个缩放
        with torch.no_grad():
            image = vae.decode(latents).sample

        show_tensor_image(image)


if __name__ == '__main__':
    main(8)

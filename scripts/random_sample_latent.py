import sys

import matplotlib.pyplot as plt

sys.path.append("..")
import torch
from utils import load_model_from_checkpoint, init_distributed
from AutoEncoder.modules import *


def show_tensor_image(image, title=None):
    image = image.squeeze()
    image = image.detach().cpu()
    image = image.permute(1, 2, 0)
    image = image.numpy()
    image = (image * 255).astype(np.uint8)
    plt.imshow(image)
    plt.axis('off')
    plt.show()



device, local_rank = init_distributed()
vae = VAE(input_size=256, input_ch=3, base_ch=128, ch_mults=[1, 2, 2, 1, 1], has_attn=[False, False, True, True, False], latent_ch=64, n_blocks=2)
vae = load_model_from_checkpoint("/path/to/your/vavae.pth", vae, device, 'vae')
vae.eval()

for _ in range(10):
    z = torch.randn(1, 64, 16, 16).to(device)
    img = vae.decode(z)
    show_tensor_image(img)

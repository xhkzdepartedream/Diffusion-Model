import Transformer.DiT
from Transformer.DiT import DiT
from AutoEncoder import *
from diffusion_and_unet import *
import torch
from tqdm import tqdm

device = 'cpu'
print("[INFO] Initialization completed.")

import lmdb
from io import BytesIO
import torch


def get_latent_tensor(index, lmdb_path = "../data/latents.lmdb"):
    """
    根据提供的索引从LMDB数据库中获取对应的潜在表示，并转换为PyTorch张量。

    参数:
    - index: int, 要获取的潜在表示的索引。
    - lmdb_path: str, LMDB数据库的路径，默认是"../data/latents.lmdb"。

    返回:
    - latent_tensor: torch.Tensor, 对应索引的潜在表示张量。
    """
    # 打开LMDB环境
    env = lmdb.open(lmdb_path, readonly = True, lock = False)

    # 生成key
    key = f"{index:06d}".encode("ascii")

    with env.begin() as txn:
        buf = txn.get(key)
        if buf is None:
            raise ValueError(f"No latent found for index {index}.")

        # 将数据流转换为张量
        latent_tensor = torch.load(BytesIO(buf))

    env.close()

    return latent_tensor


def main(DiTorNot: bool):
    if DiTorNot:
        eps_model = Transformer.DiT.DiT(16, 2, 32, 512, 12, learn_sigma = False)
        eps_model = load_model_from_checkpoint("/path/to/your/DiT.pth",
                                               eps_model, device, 'dit')
    else:
        eps_model = Unet(32, 256, [1, 2, 2], [True, True, False], n_blocks = 4).to(device)
        eps_model = load_model_from_checkpoint("/path/to/your/unet.pth", eps_model,
                                               device, 'unet')
    vae = VAE(3, [128, 160, 256, 256, 512, 512],
              [False, False, False, True, False, False], 32, 2)
    vae = load_model_from_checkpoint("/path/to/your/vavae.pth",
                                     vae, device, 'vae')

    print("[INFO] Model loading completed.")
    evaluator = sampler(eps_model, device, False)
    print(evaluator.learn_sigma)
    shape = [1, 32, 16, 16]
    xt = torch.randn(shape)
    print(xt.shape)
    z = evaluator.sample_x0(shape, 1000, True, 5, 0.03, xt).to(device)

    stats = torch.load("/path/to/your/latent_stats.pth")
    z = z * stats['std'] + stats['mean']
    x0 = vae.decoder(z)
    evaluator.show_tensor_image(x0)


if __name__ == '__main__':
    main(DiTorNot = True)

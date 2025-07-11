# save_indices.py

import torch
from torch.utils.data import DataLoader
import lmdb
from tqdm import tqdm
from init_dataset import CustomCarsDataset
from torchvision import transforms
from AutoEncoder import VQVAE, load_model_from_checkpoint
from io import BytesIO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 VQ-VAE 模型
model = VQVAE(3, [128, 160, 256, 512, 512],
              [False, False, False, True, False], 64, 2, 1024, 0.1)
model = load_model_from_checkpoint("../vqgan32c64d.pth", model, device, 'vae')
model.eval()

# 图像路径和数据增强
image_dir1 = '../data/cars_train'
image_dir2 = '../data/cars_train'

transform_vae_aug = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2, hue = 0.05),
    transforms.ToTensor()
])

# 加载并合并数据集
dataset1 = CustomCarsDataset(image_dir1, transform = transform_vae_aug)
dataset2 = CustomCarsDataset(image_dir2, transform = transform_vae_aug)
full_dataset = torch.utils.data.ConcatDataset([dataset1, dataset2])
dataloader = DataLoader(full_dataset, batch_size = 64, shuffle = False, num_workers = 4, pin_memory = True)

# 创建 LMDB 数据库
lmdb_path = "../data/latents_indices_32c64d.lmdb"
map_size = 60 * 1024 * 1024 * 1024  # 60 GB
env = lmdb.open(lmdb_path, map_size = map_size)

n_aug = 2
idx = 0
printornot = True

with torch.no_grad():
    with env.begin(write = True) as txn:
        for batch in tqdm(dataloader):
            imgs = batch.to(device)
            if printornot:
                print("Image batch shape:", imgs.shape)
                printornot = False

            for _ in range(n_aug):
                encoded = model.encoder(imgs)
                quant = model.quant_conv(encoded)
                _, _, _, indices = model.codebook(quant)  # 获取索引

                # 将索引 reshape 为 [B, H, W]
                indices = indices.view(imgs.size(0), encoded.shape[2], encoded.shape[3])

                for i in range(indices.size(0)):
                    key = f"{idx:06d}".encode("ascii")
                    buf = BytesIO()
                    torch.save(indices[i].cpu(), buf)
                    txn.put(key, buf.getvalue())
                    idx += 1

        txn.put(b'length', str(idx).encode("ascii"))

print(f"成功写入 {idx} 个增强后的索引到 {lmdb_path}")

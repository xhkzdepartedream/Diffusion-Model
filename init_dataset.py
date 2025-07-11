import os
import random
import lmdb
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from io import BytesIO


class CustomCarsDataset(Dataset):
    def __init__(self, image_dir, transform = None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if
                            fname.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image


class CelebaHQDataset(Dataset):
    def __init__(self, image_dir, transform = None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, fname)
                            for fname in os.listdir(image_dir)
                            if fname.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        filename = os.path.basename(img_path)
        return image, filename


class LatentDataset(Dataset):
    def __init__(self, lmdb_path):
        self.lmdb_path = lmdb_path
        self.env = None
        self.txn = None

        # 预先读取长度（只需一次，主进程中读）
        with lmdb.open(self.lmdb_path, readonly = True, lock = False) as env:
            with env.begin(write = False) as txn:
                self.length = int(txn.get(b'length'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly = True, lock = False, readahead = False, meminit = False)
            self.txn = self.env.begin(write = False)

        key = f"{index:06d}".encode("ascii")
        buf = self.txn.get(key)
        if buf is None:
            raise KeyError(f"Key {key} not found in LMDB")

        latent = torch.load(BytesIO(buf))
        return latent


class LatentIndexDataset(Dataset):
    def __init__(self, lmdb_path):
        self.env = lmdb.open(lmdb_path, readonly = True, lock = False, readahead = False)
        with self.env.begin() as txn:
            length = txn.get(b'length')
            if length is None:
                raise ValueError("未找到 'length' 关键字。请确认 LMDB 是否正确写入。")
            self.length = int(length)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            key = f"{idx:06d}".encode("ascii")
            data = txn.get(key)
            if data is None:
                raise IndexError(f"无法在 LMDB 中找到索引 {idx}")
            indices = torch.load(BytesIO(data))  # LongTensor, shape: [H, W]
        return indices


def show_transformed_images(dataset: Dataset, n: int = 5):
    indices = random.sample(range(len(dataset)), n)
    images = [dataset[i] for i in indices]
    fig, axes = plt.subplots(1, n, figsize = (3 * n, 3))
    if n == 1:
        axes = [axes]
    for img, ax in zip(images, axes):
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np * 0.5) + 0.5
        img_np = img_np.clip(0, 1)

        ax.imshow(img_np)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


transform = transforms.Compose([
    transforms.Resize(256, interpolation = transforms.InterpolationMode.BICUBIC),
    transforms.RandomResizedCrop(
        224,  # 最终尺寸
        scale = (0.8, 1.0),
        ratio = (0.9, 1.1),
        interpolation = transforms.InterpolationMode.BICUBIC
    ),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.ColorJitter(brightness = 0.1, contrast = 0.1),
    transforms.Resize(64, interpolation = transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
])

transform_cars = transforms.Compose([
    transforms.Resize((256, 256), interpolation = transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
])

transform_celeba = transforms.Compose([
    transforms.ToTensor(),  # PIL → tensor, [0, 1]
    transforms.Normalize(mean = [0.5, 0.5, 0.5],
                         std = [0.5, 0.5, 0.5]),  # → [-1, 1]
    transforms.RandomHorizontalFlip(p = 0.5),  # 水平翻转（常规增强）
])

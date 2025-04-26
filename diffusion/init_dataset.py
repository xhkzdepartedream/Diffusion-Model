import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


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
    transforms.Resize(64, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
])

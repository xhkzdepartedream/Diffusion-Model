# face_crop_ddp.py

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image
from tqdm import tqdm
import numpy as np
from utils import init_distributed  # 你已有的 DDP 初始化函数
from init_dataset import CelebaHQDataset, transform_celeba
from modules.bisnet import BiSeNet  # 来自 face-parsing.PyTorch
import cv2

def feather_mask(mask: np.ndarray, kernel_size: int = 21) -> np.ndarray:
    """
    对二值mask进行高斯模糊，产生羽化效果。
    kernel_size 必须为奇数，越大羽化越宽。
    返回归一化到 [0,1] 的羽化mask
    """
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)
    feathered = blurred / 255.0
    feathered = np.clip(feathered, 0.0, 1.0)
    return feathered
def get_seg_model(device):
    n_classes = 19
    net = BiSeNet(n_classes = n_classes)
    net.load_state_dict(torch.load("/path/to/your/79999_iter.pth", map_location = device))  # 预训练模型路径
    net.to(device)
    net.eval()
    return net


def segment_and_save(img_tensor, filename, net, device, save_dir):
    with torch.no_grad():
        img_tensor = img_tensor.to(device).unsqueeze(0)
        out = net(img_tensor)[0]
        parsing = out.squeeze(0).argmax(0).cpu().numpy()

    face_classes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    mask = np.isin(parsing, face_classes).astype(np.uint8) * 255

    img_np = img_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255
    img_np = img_np.astype(np.uint8)
    white_background = np.ones_like(img_np, dtype=np.uint8) * 255

    # feather
    feathered_mask = feather_mask(mask, kernel_size=31)
    feathered_mask_3ch = feathered_mask[..., None]

    # 融合
    masked_img = (img_np * feathered_mask_3ch + white_background * (1 - feathered_mask_3ch)).astype(np.uint8)

    save_path = os.path.join(save_dir, str(filename))
    os.makedirs(save_dir, exist_ok=True)
    masked_img = cv2.resize(masked_img, (256, 256), interpolation = cv2.INTER_AREA)
    cv2.imwrite(save_path, cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))




def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    device, local_rank = init_distributed()

    input_dir = '/path/to/your/celeba_hq_256/'
    output_dir = '/path/to/your/cropped_figure/'

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    dataset = CelebaHQDataset(input_dir, transform)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, sampler = sampler, batch_size = 1, num_workers = 4)

    model = get_seg_model(device)

    for img, fname in tqdm(dataloader):
        segment_and_save(img[0], fname[0], model, device, output_dir)

    cleanup()


if __name__ == '__main__':
    main()

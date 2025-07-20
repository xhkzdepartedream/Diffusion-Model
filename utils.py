import matplotlib.pyplot as plt
import torch
import os
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
from data.init_dataset import CelebaHQDataset, transform_celeba
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from diffusers import AutoencoderKL

def gather(x: torch.Tensor, t: torch.Tensor):
    """
    以下面代码为例：
    index = torch.tensor([[2, 1, 0]])
    tensor_1 = tensor_0.gather(dim=1, index)
    print(tensor_1)
    (1) output.shape = index.shape # 确定最后输出的output的shape必须与index的相同，这里是1*3的tensor，那么output必须也是1*3的tensor，先把壳打起来torch.tensor([[?,?,?]])
    (2) 对output所有值的索引，按shape方式排出来，也就是[[(0,0),(0,1),(0,2)]]
    (3) 还是对output，拿index里的值替换上面dim指定位置，dim=0替换行，dim=1即替换列。变成[[(0,2),(0,1),(0,0)]]
    (4) 按这个索引获取tensor_0相应位置的值，填进去就好了，得到torch.tensor([[5,4,3]])
    """
    x = x.to(t.device)
    res = x.gather(-1, t)
    return res.reshape(-1, 1, 1, 1)



_device = None
_local_rank = None


def init_distributed():
    import os
    import torch
    import torch.distributed as dist
    global _device, _local_rank

    if _device is not None and _local_rank is not None:
        return _device, _local_rank
    if "LOCAL_RANK" in os.environ:
        _local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(_local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend = 'nccl')
        _device = torch.device("cuda", _local_rank) if torch.cuda.is_available() else torch.device("cpu")
    else:
        _local_rank = 0
        _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Local rank: {_local_rank}")
    return _device, _local_rank

def get_obj_from_str(string):
    module, cls = string.rsplit(".", 1)
    return getattr(__import__(module, fromlist=[cls]), cls)

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key target to instantiate.")

    return get_obj_from_str(config["target"])(**config.get("params", dict()))



@torch.no_grad()
def show_reconstructions(model, images=None, image_dir=None, num_images=8, device='cuda', title_prefix=""):
    """
    展示模型（VAE或AutoencoderKL）的原图与重构图效果对比。

    参数:
        model: 已加载的模型，要求已 .eval() 并移至 device。
        images (torch.Tensor, optional): 输入图像张量 (batch, C, H, W)。如果提供，image_dir将被忽略。
        image_dir (str, optional): CelebA-HQ 图像路径（文件夹）。如果images未提供，则从此处加载。
        num_images (int): 展示图像对数量。
        device (str): 使用的设备，通常为 'cuda' 或 'cpu'。
        title_prefix (str): 图表标题的前缀。
    """
    model.eval()
    model.to(device)

    if images is not None:
        # If images are provided directly, use them
        if images.ndim == 3: # If a single image (C, H, W), add batch dim
            images = images.unsqueeze(0)
        images = images.to(device)
        # Limit to num_images if more are provided
        if images.shape[0] > num_images:
            images = images[:num_images]
    elif image_dir is not None:
        # Load images from directory
        dataset = CelebaHQDataset(image_dir, transform=transform_celeba)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=num_images, shuffle=True)
        images, _ = next(iter(dataloader))
        images = images.to(device)
    else:
        raise ValueError("Either 'images' tensor or 'image_dir' must be provided.")

    # Perform reconstruction
    if isinstance(model, AutoencoderKL):
        posterior = model.encode(images).latent_dist
        recon = model.decode(posterior.sample()).sample
    else:
        # Assuming custom VAE returns reconstruction, mean, logvar, etc.
        recon, _, _, _ = model(images)

    # Display
    # 增加 figsize 使图像更大更清晰
    fig, axes = plt.subplots(nrows=2, ncols=num_images, figsize=(num_images * 4, 8))
    if num_images == 1: # Handle single image case for axes indexing
        axes = axes.reshape(2, 1)

    for i in range(num_images):
        # Original image
        img_orig = TF.to_pil_image((images[i].clamp(-1, 1) + 1) / 2)
        axes[0, i].imshow(img_orig)
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel(f"{title_prefix}Original", fontsize=16) # 增大字体

        # Reconstructed image
        img_recon = TF.to_pil_image((recon[i].clamp(-1, 1) + 1) / 2)
        axes[1, i].imshow(img_recon)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel(f"{title_prefix}Reconstruction", fontsize=16) # 增大字体

    plt.tight_layout()
    plt.show()


def save_dataset_reconstructions(dataset, vae: torch.nn.Module,
                                 save_path: str, device: str = 'cpu'):
    """
    保存数据集中每张图像的 VAE 重构结果，并以原图文件名 + '_recon' 命名。

    参数:
        dataset: 可迭代对象，元素为 (image_tensor, filename_str)
        vae (nn.Module): 训练好的 VAE 模型
        save_path (str): 重构图像的保存目录
        device (str): 'cpu' 或 'cuda'
    """
    os.makedirs(save_path, exist_ok = True)
    vae.eval()

    for x, filename in tqdm(dataset, desc = "Saving reconstructions"):
        if x.ndim == 3:
            x = x.unsqueeze(0)  # 增加 batch 维度

        x = x.to(device)

        with torch.no_grad():
            if isinstance(vae, AutoencoderKL):
                posterior = vae.encode(x).latent_dist
                x_recon = vae.decode(posterior.sample()).sample
            else:
                x_recon, _, _, _ = vae(x)

        x_recon_vis = ((x_recon + 1) / 2).clamp(0, 1).squeeze(0).cpu()
        image = to_pil_image(x_recon_vis)

        name, ext = os.path.splitext(filename)
        recon_filename = f"{name}_recon{ext}"
        image.save(os.path.join(save_path, recon_filename))

    print(f"所有重构图已保存至：{save_path}")


def save_dataset_reconstructions_distributed(dataset, vae: torch.nn.Module,
                                             save_path: str, device: str = 'cpu',
                                             local_rank: int = 0):
    os.makedirs(save_path, exist_ok = True)
    vae.eval()

    # 用 DistributedSampler 划分数据
    sampler = DistributedSampler(dataset, shuffle = False)
    dataloader = DataLoader(dataset, batch_size = 1, sampler = sampler)

    with torch.no_grad():
        for x, filename in tqdm(dataloader, desc = f"Rank {local_rank} Saving reconstructions"):
            if x.ndim == 3:
                x = x.unsqueeze(0)

            x = x.to(device)
            if isinstance(vae, AutoencoderKL):
                posterior = vae.encode(x).latent_dist
                x_recon = vae.decode(posterior.sample()).sample
            else:
                x_recon, _, _, _ = vae(x)

            x_recon_vis = ((x_recon + 1) / 2).clamp(0, 1).squeeze(0).cpu()
            image = to_pil_image(x_recon_vis)

            name, ext = os.path.splitext(filename[0])  # 注意 filename 是 list
            recon_filename = f"{name}_recon_rank{local_rank}{ext}"
            image.save(os.path.join(save_path, recon_filename))

    print(f"[Rank {local_rank}] 保存完成")


def load_model_from_checkpoint(checkpoint_path, model_type: str, device, model_instance=None, **kwargs):
    """
    从给定的checkpoint路径加载模型权重。
    支持加载自定义VAE/UNet/Transformer模型（从.pth文件）和AutoencoderKL模型。

    对于 AutoencoderKL:
    1. 如果 `checkpoint_path` 是一个目录或Hugging Face模型ID，会使用 `from_pretrained` 加载。
    2. 如果 `checkpoint_path` 是一个 .pth 文件，它会尝试以下操作：
       a) 如果文件是一个包含 'config' 和 'state_dict'/'vae_state_dict' 键的字典，
          它会根据config自动创建模型实例并加载权重（推荐方式）。
       b) 如果提供了 `model_instance`，它会加载权重到这个实例中（用于兼容旧的.pth文件）。
       c) 否则，会抛出错误。

    参数:
    - checkpoint_path (str): 模型权重文件或目录的路径。
    - model_type (str): 模型的类型，例如 'autoencoderkl', 'vae', 'unet', 'transformer'。
    - device (torch.device): 设备对象（如'cuda'或'cpu'）。
    - model_instance (torch.nn.Module, optional): 对于需要加载状态字典的模型，传入模型实例。
                                                  对于AutoencoderKL从.pth加载是可选的（见上文）。
    - **kwargs: 传递给AutoencoderKL.from_pretrained()的额外参数。

    返回:
    - torch.nn.Module: 加载完权重并移至正确设备后的模型实例。
    """

    def remove_module_prefix(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module._orig_mod."):
                new_key = k[len("module._orig_mod."):]
            elif k.startswith("module."):
                new_key = k[len("module."):]
            elif k.startswith("_orig_mod."):
                new_key = k[len("_orig_mod."):]
            else:
                new_key = k
            new_state_dict[new_key] = v
        return new_state_dict

    if model_type.lower() == 'autoencoderkl':
        # 如果路径是目录，则使用 from_pretrained 加载并完成
        if os.path.isdir(checkpoint_path):
            print(f"[INFO] Loading AutoencoderKL from directory: {checkpoint_path}")
            model = AutoencoderKL.from_pretrained(checkpoint_path, **kwargs)
            model.to(device)
            return model
        
        # 如果路径是 .pth 文件，则处理它
        elif os.path.isfile(checkpoint_path):
            print(f"Attempting to load AutoencoderKL from .pth file: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # 推荐方式：从 checkpoint 内的 config 和 state_dict 加载
            if isinstance(checkpoint, dict) and 'config' in checkpoint and (
                    'state_dict' in checkpoint or 'vae_state_dict' in checkpoint):
                print("Found 'config' and 'state_dict' in checkpoint. Building model from config.")
                model = AutoencoderKL.from_config(checkpoint['config'])
                state_dict_key = 'vae_state_dict' if 'vae_state_dict' in checkpoint else 'state_dict'
                loaded_state_dict = checkpoint[state_dict_key]
            
            # 兼容方式：如果提供了 model_instance，则使用它
            elif model_instance is not None:
                print("Checkpoint does not contain 'config'. Using provided 'model_instance'.")
                model = model_instance
                loaded_state_dict = checkpoint.get('vae_state_dict', checkpoint.get('state_dict', checkpoint))
            
            # 新增：没有 config 和 model_instance 时，从默认模型加载
            else:
                print("[INFO] No 'config' or 'model_instance' found. Creating a default AutoencoderKL.")
                # 使用 "stabilityai/sd-vae-ft-mse" 作为基础模型
                model = AutoencoderKL.from_pretrained("checkpoints/autoencoderkl/")
                # 尝试从 checkpoint 中获取 state_dict
                if isinstance(checkpoint, dict):
                    loaded_state_dict = checkpoint.get('vae_state_dict', checkpoint.get('state_dict', checkpoint))
                else:
                    loaded_state_dict = checkpoint # 假定 checkpoint 本身就是 state_dict
                
                if not isinstance(loaded_state_dict, dict):
                    raise ValueError("The provided .pth file does not seem to be a valid state_dict.")

            cleaned_state_dict = remove_module_prefix(loaded_state_dict)
            model.load_state_dict(cleaned_state_dict)
            model.to(device)
            return model
        
        # 如果路径既不是目录也不是文件，则报错
        else:
            raise FileNotFoundError(f"Checkpoint path not found or is not a valid file/directory: {checkpoint_path}")

    else:  # 处理 VAE, UNet, Transformer 等其他自定义模型
        if model_instance is None:
            raise ValueError(f"For model_type '{model_type}', 'model_instance' must be provided.")

        model = model_instance
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 根据模型类型确定 state_dict 的键
        if model_type.lower() == 'vae':
            loaded_state_dict = checkpoint.get('vae_state_dict', checkpoint)
        elif model_type.lower() == 'unet':
            loaded_state_dict = checkpoint.get('model_state_dict', checkpoint)
        elif model_type.lower() == 'transformer':
            loaded_state_dict = checkpoint.get('transformer_state_dict', checkpoint)
        else:  # 默认回退
            loaded_state_dict = checkpoint.get('model_state_dict', checkpoint)

        cleaned_state_dict = remove_module_prefix(loaded_state_dict)
        model.load_state_dict(cleaned_state_dict)
        model.to(device)
        return model
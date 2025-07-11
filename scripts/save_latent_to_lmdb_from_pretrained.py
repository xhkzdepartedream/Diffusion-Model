import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import lmdb
import time
from init_dataset import CelebaHQDataset, transform_celeba
from diffusers import AutoencoderKL
import shutil
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from io import BytesIO
from utils import init_distributed,load_model_from_checkpoint


def get_dataset_and_sampler(image_dir):
    dataset = CelebaHQDataset(image_dir, transform = transform_celeba)
    sampler = DistributedSampler(dataset, shuffle = False)
    return dataset, sampler


def merge_lmdbs(source_dir, target_path, num_ranks, local_rank):
    if local_rank == 0:
        if os.path.exists(target_path) and os.path.isdir(target_path):
            print(f"正在删除旧的 LMDB 目录：{target_path}")
            shutil.rmtree(target_path)
        # Give some time for the directory to be deleted across nodes if it's a shared filesystem
        time.sleep(1)
        if os.path.exists(target_path):
            raise RuntimeError(f"LMDB 目录未能成功删除：{target_path}")

    map_size = 60 * 1024 * 1024 * 1024
    env = lmdb.open(target_path, map_size = map_size)

    idx = 0
    with env.begin(write = True) as txn:
        for rank in range(num_ranks):
            path = os.path.join(source_dir, f"latents_rank{rank}.lmdb")
            print(f"正在合并 {path} ...")
            source_env = lmdb.open(path, readonly = True, lock = False)

            with source_env.begin() as src_txn:
                cursor = src_txn.cursor()
                for key, value in tqdm(cursor, desc = f"Merging rank {rank}"):
                    if key == b'length':
                        continue
                    new_key = f"{idx:06d}".encode("ascii")
                    txn.put(new_key, value)
                    idx += 1

            source_env.close()

        txn.put(b'length', str(idx).encode("ascii"))

    env.close()
    print(f"合并完成，共写入 {idx} 个 latent。输出库：{target_path}")

def save_latents_to_lmdb(model: AutoencoderKL, image_dir: str, output_lmdb_dir: str, compression_method: str = 'none',
                         batch_size: int = 64, num_workers: int = 4, augment_flip: bool = False):
    """
    将图像编码为 latent 并保存到 LMDB 数据库。

    参数:
        model (AutoencoderKL): 已加载的 AutoencoderKL 模型实例。
        image_dir (str): 原始图像的路径。
        output_lmdb_dir (str): 输出 LMDB 数据库的目录。
        compression_method (str): 压缩方式，可选 'none' (不缩放), 'scale' (使用模型自带的scaling_factor),
                                  'stdscale' (标准化到均值0，标准差1)。默认为 'none'。
        batch_size (int): 数据加载的批次大小。
        num_workers (int): 数据加载的工作进程数。
        augment_flip (bool): 是否对图像进行水平翻转增强，并将翻转后的 latent 一同保存。
    """
    device, local_rank = init_distributed()
    world_size = dist.get_world_size()

    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])
    model.eval()

    dataset, sampler = get_dataset_and_sampler(image_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)

    # Handle LMDB path for each rank
    rank_lmdb_path = os.path.join(output_lmdb_dir, f"latents_rank{local_rank}.lmdb")
    if os.path.exists(rank_lmdb_path):
        shutil.rmtree(rank_lmdb_path)
    env = lmdb.open(rank_lmdb_path, map_size=60 * 1024 * 1024 * 1024)

    # First pass to calculate mean and std
    sum_latent = torch.tensor(0.0, device=device)
    sum_squared = torch.tensor(0.0, device=device)
    n_pixels = torch.tensor(0, device=device)

    for batch, _ in tqdm(dataloader, desc=f"Rank {local_rank} - Stats"):  # Use dataloader for stats pass
        imgs = batch.to(device)
        if augment_flip:
            flipped_imgs = torch.flip(imgs, dims=[3])
            imgs = torch.cat([imgs, flipped_imgs])

        with torch.no_grad():
            latent = model.module.encode(imgs).latent_dist.sample()
        sum_latent += latent.sum()
        sum_squared += (latent ** 2).sum()
        n_pixels += latent.numel()

    dist.reduce(sum_latent, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(sum_squared, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(n_pixels, dst=0, op=dist.ReduceOp.SUM)

    mean = torch.tensor([0.0], device=device)
    std = torch.tensor([1.0], device=device)
    if local_rank == 0:
        if n_pixels > 0:
            mean = sum_latent / n_pixels
            var = sum_squared / n_pixels - mean ** 2
            std = var.sqrt() + 1e-6
        print(f"[Global latent stats] mean: {mean.item():.6f}, std: {std.item():.6f}")
        torch.save({'mean': mean.cpu(), 'std': std.cpu()}, os.path.join(output_lmdb_dir, 'latent_stats.pth'))

    dist.broadcast(mean, src=0)
    dist.broadcast(std, src=0)
    dist.barrier()  # Ensure all ranks have the stats before proceeding

    # Second pass to encode and save latents
    idx_base = 0
    if local_rank == 0:
        with env.begin() as txn:
            length_bytes = txn.get(b'length')
            idx_base = int(length_bytes.decode()) if length_bytes else 0

    start_idx = torch.tensor(idx_base).to(device)
    dist.broadcast(start_idx, src=0)
    local_idx = start_idx.item() + local_rank

    with env.begin(write=True) as txn:
        total_written = 0
        for batch, _ in tqdm(dataloader, desc=f"Rank {local_rank} - Encoding"):  # Use dataloader for encoding pass
            imgs = batch.to(device)
            if augment_flip:
                flipped_imgs = torch.flip(imgs, dims=[3])
                imgs = torch.cat([imgs, flipped_imgs])

            with torch.no_grad():
                z = model.module.encode(imgs).latent_dist.sample()
                if compression_method == 'scale':
                    z = z * model.module.config.scaling_factor
                elif compression_method == 'stdscale':
                    z = (z - mean) / std

            for latent in z:
                key = f"{local_idx:06d}".encode("ascii")
                buf = BytesIO()
                torch.save(latent.cpu(), buf)  # Save as CPU tensor to reduce memory on GPU
                txn.put(key, buf.getvalue())
                local_idx += world_size  # Increment by world_size for distributed saving
                total_written += 1

        if local_rank == 0:
            txn.put(b'length', str(local_idx).encode("ascii"))
            print(f"[Rank 0] Total written: {total_written}")

    env.close()
    dist.barrier()  # Ensure all ranks finish writing before merging

    # Merging and cleanup (only on rank 0)
    if local_rank == 0:
        merge_lmdbs(source_dir=output_lmdb_dir, target_path=os.path.join(output_lmdb_dir, 'latents.lmdb'),
                    num_ranks=world_size, local_rank=local_rank)
        for r in range(world_size):
            temp_lmdb_path = os.path.join(output_lmdb_dir, f"latents_rank{r}.lmdb")
            if os.path.exists(temp_lmdb_path):
                shutil.rmtree(temp_lmdb_path)
                print(f"[Rank 0] Deleted temporary LMDB: {temp_lmdb_path}")


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    device, local_rank = init_distributed()
    print(f"[Rank {local_rank}] Starting latent saving process...")

    model_path = "/path/to/your/autoencoderkl/"
    loaded_model = load_model_from_checkpoint(
        checkpoint_path = model_path,
        model_type = 'autoencoderkl',
        device = device
    )
    src_dict = '/path/to/your/celeba_hq_256/'
    dst_dict = '/path/to/your/data/'
    save_latents_to_lmdb(
        model=loaded_model,
        image_dir=src_dict,
        output_lmdb_dir=dst_dict,
        compression_method='none', # or 'scale' or 'none'
        batch_size=64,
        num_workers=8,
        augment_flip=True
    )
    cleanup()
    print(f"[Rank {local_rank}] Cleanup completed.")
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import lmdb
import time
from init_dataset import CelebaHQDataset, transform_celeba
from AutoEncoder import VAE, load_model_from_checkpoint
import torch
from diffusers import AutoencoderKL
import shutil
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from io import BytesIO
from utils import init_distributed
from modules import *


def get_dataset_and_sampler(image_dir):
    dataset = CelebaHQDataset(image_dir, transform = transform_celeba)
    sampler = DistributedSampler(dataset, shuffle = False)
    return dataset, sampler


def merge_lmdbs(source_dir, target_path, num_ranks):
    if local_rank == 0:
        if os.path.exists(target_path) and os.path.isdir(target_path):
            print(f"正在删除旧的 LMDB 目录：{target_path}")
            shutil.rmtree(target_path)

        # 再次确认删除成功
        if os.path.exists(target_path):
            raise RuntimeError(f"LMDB 目录未能成功删除：{target_path}")

    else:
        time.sleep(3)

    map_size = 60 * 1024 * 1024 * 1024  # 60GB，可按需要调整
    env = lmdb.open(target_path, map_size = map_size)

    idx = 0  # 全局 key 索引

    with env.begin(write = True) as txn:
        for rank in range(num_ranks):
            path = os.path.join(source_dir, f"latents_rank{rank}.lmdb")
            print(f"正在合并 {path} ...")
            source_env = lmdb.open(path, readonly = True, lock = False)

            with source_env.begin() as src_txn:
                cursor = src_txn.cursor()
                for key, value in tqdm(cursor, desc = f"Merging rank {rank}"):
                    if key == b'length':
                        continue  # 跳过 length 字段
                    new_key = f"{idx:06d}".encode("ascii")
                    txn.put(new_key, value)
                    idx += 1

            source_env.close()

        # 写入新的 length
        txn.put(b'length', str(idx).encode("ascii"))

    env.close()
    print(f"合并完成，共写入 {idx} 个 latent。输出库：{target_path}")


def process(rank):
    print(f"Running on rank {rank}.")
    device, local_rank = init_distributed()

    model = AutoencoderKL.from_pretrained('/path/to/your/autoencoderkl/')
    # model = load_model_from_checkpoint("/path/to/your/vavae.pth", model, device, 'vae')
    model = model.to(device)
    model = DDP(model, device_ids = [local_rank])
    model.eval()

    image_dir = '/path/to/your/cropped_figure/'
    dataset, sampler = get_dataset_and_sampler(image_dir)
    dataloader = DataLoader(dataset, batch_size = 64, sampler = sampler, num_workers = 4, pin_memory = True)

    map_size = 60 * 1024 * 1024 * 1024
    lmdb_path = f"../data/latents_rank{local_rank}.lmdb"

    # 每个 rank 清除自己的 LMDB
    if os.path.exists(lmdb_path):
        shutil.rmtree(lmdb_path)
    env = lmdb.open(lmdb_path, map_size = map_size)

    sum_latent = torch.tensor(0.0).to(device)
    sum_squared = torch.tensor(0.0).to(device)
    n_samples = torch.tensor(0).to(device)

    with torch.no_grad():
        for batch, _ in tqdm(dataloader, desc = f"Rank {local_rank} - Stats"):
            imgs = batch.to(device)
            z, _, _ = model.module.encode(imgs)
            B, C, H, W = z.shape
            n_pixels = B * C * H * W

            sum_latent += z.sum()
            sum_squared += (z ** 2).sum()
            n_samples += n_pixels

    dist.reduce(sum_latent, dst = 0, op = dist.ReduceOp.SUM)
    dist.reduce(sum_squared, dst = 0, op = dist.ReduceOp.SUM)
    dist.reduce(n_samples, dst = 0, op = dist.ReduceOp.SUM)

    if local_rank == 0:
        mean = sum_latent / n_samples
        var = (sum_squared / n_samples) - (mean ** 2)
        std = var.sqrt() + 1e-10
        print(f"[Global] Mean: {mean}, Std: {std}")
        torch.save({'mean': mean.cpu(), 'std': std.cpu()}, 'latent_stats.pth')
    else:
        mean = torch.tensor([0.0]).to(device)
        std = torch.tensor([0.0]).to(device)

    dist.broadcast(mean, src = 0)
    dist.broadcast(std, src = 0)
    dist.barrier()

    idx_base = 0
    if local_rank == 0:
        with env.begin() as txn:
            length_bytes = txn.get(b'length')
            idx_base = int(length_bytes.decode()) if length_bytes else 0

    start_idx = torch.tensor(idx_base).to(device)
    dist.broadcast(start_idx, src = 0)
    local_idx = start_idx.item() + local_rank

    with env.begin(write = True) as txn:
        total_written = 0
        for batch, _ in tqdm(dataloader, desc = f"Rank {local_rank} - Encoding"):
            imgs = batch.to(device)
            with torch.no_grad():
                z, _, _ = model.module.encode(imgs)
                z = (z - mean) / std

            for latent in z:
                key = f"{local_idx:06d}".encode("ascii")
                buf = BytesIO()
                torch.save(latent.cpu(), buf)
                txn.put(key, buf.getvalue())
                local_idx += world_size
                total_written += 1

        if local_rank == 0:
            new_length = local_idx
            txn.put(b'length', str(new_length).encode("ascii"))
            print(f"[Rank 0] Total written: {total_written}")

    env.close()


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    import sys

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    print(f"[Rank {local_rank}] Start processing...")

    try:
        process(local_rank)
    except Exception as e:
        print(f"[Rank {local_rank}] Error occurred: {e}")
        raise
    finally:
        cleanup()
        print(f"[Rank {local_rank}] Cleanup completed.")

    if local_rank == 0:
        merge_lmdbs(source_dir = "../data", target_path = "../data/latents.lmdb", num_ranks = world_size)
        for r in range(world_size):
            lmdb_path = f"../data/latents_rank{r}.lmdb"
            if os.path.exists(lmdb_path):
                shutil.rmtree(lmdb_path)
                print(f"[Rank 0] Deleted temporary LMDB: {lmdb_path}")

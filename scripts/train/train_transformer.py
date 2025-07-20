from diffusion import *  # 假设这是你的数据处理、模型定义等模块
from torch.nn.parallel import DistributedDataParallel as DDP
from Transformer import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
from data.init_dataset import *

device, local_rank = init_distributed()


def main():
    dir = "./data/latents_indices_32c64d.lmdb/"
    train_dataset = LatentIndexDataset(dir)
    print("[INFO] Initialization completed.")
    transformer_launcher = Transformer_trainer(train_dataset, [1024, 1024, 1025, 1026])
    transformer_launcher.model = torch.compile(transformer_launcher.model)
    transformer_launcher.model = DDP(transformer_launcher.model, device_ids = [local_rank], output_device = local_rank)
    torch.cuda.empty_cache()
    transformer_launcher.train(100, 64, 3 * 1e-4)

if __name__ == '__main__':
    main()
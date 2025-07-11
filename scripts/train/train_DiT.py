from Transformer.DiT_trainer import DiT_trainer
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
from init_dataset import *

device, local_rank = init_distributed()


def main():
    dir = "/path/to/your/latents.lmdb/"
    train_dataset = LatentDataset(dir)
    print("[INFO] Initialization completed.")
    DiT_launcher = DiT_trainer(train_dataset, 800, 128, 1e-4, 800, False, 32, 2, 4,
                               512, 12, title = 'DiT', num_heads = 8, learn_sigma = False, attn_dropout = 0.1,
                               mlp_dropout = 0.1)
    DiT_launcher.eps_model = DDP(DiT_launcher.eps_model, device_ids = [local_rank], output_device = local_rank)
    torch.cuda.empty_cache()
    DiT_launcher.train(vlb_factor = 0.0)


if __name__ == '__main__':
    main()

from trainer.DiT_trainer import DiT_trainer
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
from data.init_dataset import *

device, local_rank = init_distributed()


def main():
    dir = "data/latents.lmdb/"
    train_dataset = LatentDataset(dir)
    print("[INFO] Initialization completed.")
    DiT_launcher = DiT_trainer(train_dataset, 750, 128, 1e-4, 500, False, 32, 2, 4,
                               768, 12, pe = "rope", title = 'DiT_rope', num_heads = 8, learn_sigma = False,
                               attn_dropout = 0, mlp_dropout = 0)
    DiT_launcher.eps_model = DDP(DiT_launcher.eps_model, device_ids = [local_rank], output_device = local_rank)
    torch.cuda.empty_cache()
    DiT_launcher.train(vlb_factor = 0.0)


if __name__ == '__main__':
    main()

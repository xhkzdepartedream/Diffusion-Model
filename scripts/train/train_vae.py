import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
from AutoEncoder import *
from torch.nn.parallel import DistributedDataParallel as DDP

device, local_rank = init_distributed()


def main():
    dir = "./data/cars_train/"
    train_dataset = CustomCarsDataset(dir, transform = transform_vae)

    vae_launcher = vae_trainer(train_dataset, 'u', batch_size = 64, lr = 1e-4, n_epoch = 100, lambda_ = 1,
                               max_beta = 3.2 * 1e-4, warmup_epochs = 0, warmup_start = 0)
    vae_launcher.vae = vae_launcher.vae.to(device)
    vae_launcher.vae = DDP(vae_launcher.vae, device_ids = [local_rank], output_device = local_rank,
                           find_unused_parameters = True)
    vae_launcher.train()


if __name__ == '__main__':
    main()
    with open("/done.txt", "w") as f:
        f.write("DONE")

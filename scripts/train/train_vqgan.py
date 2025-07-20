import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
from data.init_dataset import *
from AutoEncoder import *
from torch.nn.parallel import DistributedDataParallel as DDP

device, local_rank = init_distributed()


def main():
    image_dir1 = '/data1/yangyanliang/.cache/kagglehub/datasets/badasstechie/celebahq-resized-256x256/versions/1/celeba_hq_256/'

    # 加载数据集
    dataset1 = CelebaHQDataset(image_dir1, transform = transform_celeba)

    dataset1 = torch.utils.data.Subset(dataset1, range(10000))

    vqgan_launcher = VQGAN_trainer(dataset1, 256, 3, 128, [1, 2, 2, 1, 1],
                                   [False, False, False, True, False],
                                   32, 2, 2048, 0.5, 64)
    vqgan_launcher.vqvae = vqgan_launcher.vqvae.to(device)
    vqgan_launcher.discriminator = vqgan_launcher.discriminator.to(device)
    vqgan_launcher.vqvae = DDP(vqgan_launcher.vqvae, device_ids = [local_rank], output_device = local_rank)
    vqgan_launcher.train(100, 8, 1e-4, get_perc_loss_factor, get_vq_loss_factor,
                         1, get_gan_loss_factor, 0, 5,)


if __name__ == '__main__':
    main()
    REMOTE_FLAG_PATH = "./scripts/done.txt"
    with open(REMOTE_FLAG_PATH, 'w') as f:
        f.write('done.')

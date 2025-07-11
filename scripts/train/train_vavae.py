import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
from AutoEncoder.VAVAE_trainer import *
from torch.nn.parallel import DistributedDataParallel as DDP

device, local_rank = init_distributed()


def main():
    image_dir1 = '/path/to/your/celeba_hq_256/'

    # 加载数据集
    dataset1 = CelebaHQDataset(image_dir1, transform = transform_celeba)

    dataset1 = torch.utils.data.Subset(dataset1, range(10000))
    vae_launcher = VAVAE_trainer(dataset = dataset1, input_size = 256, input_ch = 3, base_ch = 128,
                                 ch_mults = [1, 1, 2, 2, 1], has_attn = [False, False, True, False, False],
                                 latent_ch = 32,
                                 n_blocks = 2, dis_basic_ch = 64, use_vf = True, title = "test3")
    vae_launcher.vae = vae_launcher.vae.to(device)
    vae_launcher.discriminator = vae_launcher.discriminator.to(device)
    # vqvae_launcher.vae = torch.compile(vqvae_launcher.vae)
    vae_launcher.vae = DDP(vae_launcher.vae, device_ids = [local_rank], output_device = local_rank,
                           find_unused_parameters = False)

    vae_launcher.train(200, 8, 3 * 1e-5, get_recon_factor, get_kl_factor, get_perc_factor, get_gan_factor,
                       get_vf_factor, 0.25, 0.5, 1.0, 1.0, 2,
                       "/path/to/your/vavae.pth")


if __name__ == '__main__':
    main()
    REMOTE_FLAG_PATH = "/path/to/your/done.txt"
    with open(REMOTE_FLAG_PATH, 'w') as f:
        f.write('done.')

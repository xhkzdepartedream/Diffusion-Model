import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
from trainer.AutoencoderKL_trainer import AutoencoderKL_trainer
from utils import init_distributed
from data.init_dataset import CelebaHQDataset, transform_celeba
from torch.nn.parallel import DistributedDataParallel as DDP

device, local_rank = init_distributed()


def main():
    # Define your dataset directory and transformation
    # Make sure to replace this with your actual CELEBA-HQ dataset path
    dir = '/data1/yangyanliang/.cache/kagglehub/datasets/badasstechie/celebahq-resized-256x256/versions/1/celeba_hq_256/'
    train_dataset = CelebaHQDataset(dir, transform = transform_celeba)
    # Specify the pretrained model to fine-tune
    # You can use a local path or a Hugging Face model ID
    pretrained_model_name_or_path = "/data1/yangyanliang/checkpoints/autoencoderkl/"

    # Initialize the trainer
    autoencoderkl_launcher = AutoencoderKL_trainer(
        dataset = train_dataset,
        title = 'celeba_hq_vf',
        pretrained_model_name_or_path = pretrained_model_name_or_path
    )

    # Wrap the model with DDP for distributed training
    autoencoderkl_launcher.vae = DDP(autoencoderkl_launcher.vae, device_ids = [local_rank],
                                     output_device = local_rank, find_unused_parameters = False)
    autoencoderkl_launcher.vf_linear = DDP(autoencoderkl_launcher.vf_linear, device_ids = [local_rank],
                                           output_device = local_rank, find_unused_parameters = False)

    # Define training parameters
    epochs = 40
    batch_size = 8
    lr = 1e-4  # Fine-tuning typically uses a smaller learning rate
    recon_factor = 1.0
    kl_factor = 1.1e-6
    perc_factor = 0.3
    vf_factor = 0.1
    distmat_margin = 0.25
    cos_margin = 0.5
    distmat_weight = 0.5
    cos_weight = 0.5
    warm_up = 10
    ckpt = "./autoencoderkl_finetuned_celeba_hq_vf_30.pth"

    # Start training
    autoencoderkl_launcher.train(
        epochs = epochs,
        batch_size = batch_size,
        lr = lr,
        recon_factor = recon_factor,
        kl_factor = kl_factor,
        perc_factor = perc_factor,
        vf_factor = vf_factor,
        distmat_margin = distmat_margin,
        cos_margin = cos_margin,
        distmat_weight = distmat_weight,
        cos_weight = cos_weight,
        warm_up = warm_up,
        checkpoint_path = ckpt
    )


if __name__ == '__main__':
    main()

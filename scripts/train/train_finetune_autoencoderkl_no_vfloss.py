import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
from trainer.AutoencoderKL_trainer_no_vfloss import AutoencoderKL_trainer_no_vfloss
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
    pretrained_model_name_or_path = "./autoencoderkl_success2.pth"

    # Initialize the trainer
    autoencoderkl_launcher = AutoencoderKL_trainer_no_vfloss(
        dataset = train_dataset,
        title = 'celeba_hq2',  # Updated title
        pretrained_model_name_or_path = pretrained_model_name_or_path
    )

    # Wrap the model with DDP for distributed training
    autoencoderkl_launcher.vae = DDP(autoencoderkl_launcher.vae, device_ids = [local_rank],
                                     output_device = local_rank, find_unused_parameters = False)

    # Define training parameters
    epochs = 10
    batch_size = 8
    lr = 1e-4  # Fine-tuning typically uses a smaller learning rate
    recon_factor = 1.0
    kl_factor = 1e-6
    perc_factor = 0.3

    # Start training
    autoencoderkl_launcher.train(
        epochs = epochs,
        batch_size = batch_size,
        lr = lr,
        recon_factor = recon_factor,
        kl_factor = kl_factor,
        perc_factor = perc_factor,
    )


if __name__ == '__main__':
    main()

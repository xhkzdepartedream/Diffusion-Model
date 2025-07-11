from AutoEncoder import *
from init_dataset import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
from utils import init_distributed


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


device, local_rank = init_distributed()
dir = '/path/to/your/cropped_figure/'

dataset = CelebaHQDataset(dir, transform = transform_celeba)
model = VAE(3, [128, 160, 256, 256, 512, 512],
            [False, False, False, True, False, False], 32, 2)
model = load_model_from_checkpoint("/path/to/your/vavae.pth",
                                   model, device, 'vae')
model = DDP(model, device_ids = [local_rank])
save_dataset_reconstructions_distributed(dataset, model, save_path = '../data/output', device = device,
                                         local_rank = local_rank)
cleanup()

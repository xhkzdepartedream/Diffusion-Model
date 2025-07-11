import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
from diffusion_and_unet import *  # 假设这是你的数据处理、模型定义等模块
from torch.nn.parallel import DistributedDataParallel as DDP
from torchsummary import summary


from init_dataset import *

device, local_rank = init_distributed()


def main():
    dir = "/path/to/your/latents.lmdb/"
    train_dataset = LatentDataset(dir)
    print("[INFO] Initialization completed.")
    unet_launcher = unet_trainer(train_dataset, 800, 4, 128, [1, 1, 2, 2, 1],
                                 [False, False, True, True, False],
                                 3, 128, 1e-4, 1000)
    unet_launcher.eps_model = DDP(unet_launcher.eps_model, device_ids = [local_rank], output_device = local_rank)
    torch.cuda.empty_cache()
    unet_launcher.train()


if __name__ == '__main__':
    main()
    with open("done.txt", "w") as f:
        f.write("Training completed on {}\n".format(os.uname().nodename))
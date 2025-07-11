import torch
import torch.nn as nn
import timm
import torch.nn.functional as F


class VFloss_module(nn.Module):
    def __init__(self, latent_ch: int):
        super().__init__()
        model = timm.models.create_model("vit_large_patch14_dinov2.lvd142m", pretrained = False,
                                         dynamic_img_size = True,
                                         pretrained_cfg_overlay = dict(
                                             file = '../../../checkpoints/pytorch_model.bin'))
        model.eval()
        self.model = model

    def forward(self, x):
        x = F.interpolate(x, size = (224, 224), mode = 'bilinear', align_corners = False)  # 我有一计！
        with torch.no_grad():
            tokens = self.model.forward_features(x)  # [B, 256, 1024]
            patch_tokens = tokens[:, 1:, :]  # 去掉 CLS token
            return patch_tokens

# if __name__ == "__main__":
#     temp = VFloss_module(64)
#     x = torch.randn([1, 3, 224, 224])
#     x = temp(x)
#     print(x.shape)

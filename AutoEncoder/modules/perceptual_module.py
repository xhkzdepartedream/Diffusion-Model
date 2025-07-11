import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models.feature_extraction import create_feature_extractor


class PerceptualModule(nn.Module):
    def __init__(self, mode = 'vgg', layers = None, lpips_net = 'vgg'):
        super().__init__()
        self.mode = mode.lower()

        if layers is None:
            layers = ['features.0', 'features.3', 'features.8', 'features.15', 'features.18']

        if self.mode == 'vgg':
            vgg = vgg16(weights = VGG16_Weights.DEFAULT).eval()
            for p in vgg.parameters():
                p.requires_grad = False
            self.extractor = create_feature_extractor(vgg, return_nodes = {l: l for l in layers})

            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        elif self.mode == 'lpips':
            self.extractor = lpips.LPIPS(net = lpips_net).eval()
        else:
            raise ValueError(f"Unsupported mode {mode}. Choose 'vgg' or 'lpips'.")

        for p in self.extractor.parameters():
            p.requires_grad = False

    def normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, x, x_hat):
        with torch.no_grad():
            x = torch.clamp(x, 0, 1)
            x_hat = torch.clamp(x_hat, 0, 1)

            if self.mode == 'vgg':
                x = self.normalize(x)
                x_hat = self.normalize(x_hat)
                f_x = self.extractor(x)
                f_x_hat = self.extractor(x_hat)
                loss = sum(F.mse_loss(f_x[k], f_x_hat[k]) for k in f_x)
                return loss
            elif self.mode == 'lpips':
                return self.extractor(x, x_hat).mean()


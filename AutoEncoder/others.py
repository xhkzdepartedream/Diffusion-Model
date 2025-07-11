class Scaler(nn.Module):
    def __init__(self, tau = 0.5):
        super().__init__()
        self.tau = tau
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, inputs, mode: bool = True):
        if mode:
            scale = self.tau + (1 - self.tau) * torch.sigmoid(self.scale)
        else:
            scale = (1 - self.tau) * torch.sigmoid(-self.scale)
        for _ in range(len(inputs.shape) - len(scale.shape)):
            scale = scale.unsqueeze(-1)

        return inputs * torch.sqrt(scale)


class CombinedScaler(nn.Module):
    def __init__(self, num_features, tau = 0.5, affine_bn = False, eps = 1e-8):
        """
        :param num_features: 输入特征的数量（即通道数）
        :param tau: scaler 中的参数
        :param affine_bn: 是否让 BatchNorm 学习缩放和平移参数
        :param eps: BatchNorm 的小常量，避免除零错误
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine = affine_bn, eps = eps)
        self.scaler = Scaler(tau = tau)

    def forward(self, x, mode = 'positive'):
        x = self.bn(x)
        x = self.scaler(x, mode = mode)
        return x

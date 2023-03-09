import torch
import torch.nn as nn


class FRN(nn.Module):
    # 对于不含有1x1特征的模型, eps=1e-6，#对于含有1x1特征的模型, eps是可学习的参数
    def __init__(self, num_features, eps=1e-6, learnable_eps=True):
        super().__init__()
        shape = (1, num_features, 1, 1)
        self.eps = nn.Parameter(torch.ones(*shape) * eps, requires_grad=True)
        # print(self.eps.shape)
        if not learnable_eps:
            self.eps.requires_grad_(False)
        self.gamma = nn.Parameter(torch.Tensor(*shape), requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor(*shape), requires_grad=True)
        self.tlu = nn.Parameter(torch.Tensor(*shape), requires_grad=True)
        self.reset_parameters()

    def forward(self, x):
        avg_dims = tuple(range(2, x.dim()))  # range(2,4)=2,3，0123->NCHW
        # 计算宽高维度的均值，再求平方
        # nu2 = torch.pow(x, 2).mean(dim=avg_dims, keepdim=True)
        nu2 = torch.pow(x, 2).mean(dim=(2, 3), keepdim=True)
        # x = x * torch.rsqrt(nu2 + torch.abs(self.eps))
        x = x / torch.sqrt(nu2 + torch.abs(self.eps))
        return torch.max(self.gamma * x + self.beta, self.tlu)

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.ones_(self.beta)
        nn.init.ones_(self.tlu)


if __name__ == '__main__':
    x = torch.rand(1, 3, 12, 12)
    frn = FRN(3)
    # print(frn(x))
    print(frn(x).shape)

import torch
import torch.nn.functional as F
#定义上采样层，邻近插值
class UpsampleLayer(torch.nn.Module):
    def __init__(self):
        super(UpsampleLayer, self).__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='nearest')

#定义卷积层
class ConvolutionalLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(ConvolutionalLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.sub_module(x)

#定义下采样层
class DownsamplingLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsamplingLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, 3, 2, 1)
        )

    def forward(self, x):
        return self.sub_module(x)
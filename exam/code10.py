import torch
from torch import nn


class Focus(nn.Module):

    def __init__(self, c1, c2, k=1):
        super(Focus, self).__init__()
        self.conv = nn.Conv2d(c1 * 4, c2, k, 1)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))

if __name__ == '__main__':
    data=torch.randn([1,64,224,224])
    focus=Focus(64,64)
    out=focus(data)
    print(out.shape)
import torch
import torch.nn as nn

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, stride=1),  # 10*10*3
            # nn.BatchNorm2d(10),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2,padding=0),  # 5*5*10
            nn.Conv2d(10, 16, kernel_size=3, stride=1),  # 3*3*16
            # nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),  # 1*1*32
            # nn.BatchNorm2d(32),
            nn.PReLU()
        )

        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1, stride=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        self.conv4_3 = nn.Conv2d(32, 10, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.pre_layer(x)
        cls = torch.softmax(self.conv4_1(x),1)
        bbox = self.conv4_2(x)
        landmark = self.conv4_3(x)
        return cls, bbox,landmark

if __name__ == '__main__':
    data=torch.randn([10,3,12,12])
    net=Net()
    out=net(data)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
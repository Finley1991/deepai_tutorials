import torch.nn as nn
import torch
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,3,3,2,1),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )#3,14,14=588
        self.conv2 = nn.Sequential(
            nn.Conv2d(3,6,3,2,1),
            nn.BatchNorm2d(6),
            nn.ReLU()
        )#6,7,7=294
        self.fc=nn.Sequential(
            nn.Linear(6*7*7,64)
        )
    def forward(self,x):
        y=self.conv1(x)
        y=self.conv2(y)
        y=y.reshape(y.size(0),-1)
        y=self.fc(y)
        return y

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=nn.Sequential(
            nn.Linear(64,294),
            nn.BatchNorm1d(294),
            nn.ReLU()
        )
        self.conv_transpose1=nn.Sequential(
            nn.ConvTranspose2d(6,3,3,2,1,output_padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        self.conv_transpose2=nn.Sequential(
            nn.ConvTranspose2d(3,1,3,2,1,output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        y=self.fc(x)
        y=y.reshape([y.size(0),6,7,7])
        y=self.conv_transpose1(y)
        y=self.conv_transpose2(y)
        return y

class Main_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def forward(self, x):
        encoder_out=self.ecnoder(x)
        output=self.decoder(encoder_out)
        return output

if __name__ == '__main__':
    x = torch.randn([10,1,28,28])
    net=Main_Net()
    out = net(x)
    print(out.shape)
    # print(net.parameters())
    # print(list(net.parameters()))
    for param in net.parameters():
        print(param.shape)
        exit()
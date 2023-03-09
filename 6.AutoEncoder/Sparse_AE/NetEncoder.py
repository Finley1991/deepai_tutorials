import torch
import torch.nn as nn
class Encoder_Net(nn.Module):
    def __init__(self):
        super(Encoder_Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 128,3,2,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )#128*14*14
        self.layer2 = nn.Sequential(
            nn.Conv2d(128,512,3, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )#512*7*7
        self.layer3 = nn.Sequential(
            nn.Linear(512*7*7,128),
            nn.Sigmoid()
        )#128

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = torch.reshape(out,[out.size(0),-1])
        out = self.layer3(out)
        return out



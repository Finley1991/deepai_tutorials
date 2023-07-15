from torch import nn
import torch
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.aenet=nn.Sequential(
            nn.Linear(784,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.aenet(x)

if __name__ == '__main__':
    data=torch.randn(10,784)
    ae=AutoEncoder()
    out=ae(data)
    print(out.shape)




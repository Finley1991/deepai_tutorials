from torch import nn

class D_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.dnet=nn.Sequential(
            nn.Linear(784,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.dnet(x)

class G_Net(nn.Module):
    def __init__(self):
        super(G_Net, self).__init__()
        self.gnet=nn.Sequential(
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,784)
        )
    def forward(self,x):
        return self.gnet(x)

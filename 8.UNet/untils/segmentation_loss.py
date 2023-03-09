import torch
from torch import nn
class SoftDice_Loss(nn.Module):
    def __init__(self,sigmoid=False):
        super(SoftDice_Loss, self).__init__()
        self.sigmoid=sigmoid

    def forward(self,pred,target):
        if self.sigmoid==True:
            pred=torch.sigmoid(pred)
        p = pred.reshape(pred.size(0), -1)
        g = target.reshape(target.shape[0], -1)

        d = (2 * (p * g).sum() + 1) / ((p**2).sum() + (g**2).sum() + 1)
        loss=1-d
        return loss

class Focal_loss(nn.Module):
    def __init__(self,alpha,gamma,sigmoid=False):
        super(Focal_loss, self).__init__()
        self.alpha=alpha
        self.gamma=gamma
        self.sigmoid=sigmoid
    def forward(self,pred,target):
        if self.sigmoid==True:
            pred=torch.sigmoid(pred)

        p = pred.reshape(pred.size(0), -1)
        g = target.reshape(target.shape[0], -1)

        positive_sample=self.alpha*((1-p)**self.gamma)*g*torch.log(p)
        negative_sample=(1-self.alpha)*(p**self.gamma)*(1-g)*torch.log(1-p)
        focal_loss=-(positive_sample+negative_sample).mean()
        return focal_loss

if __name__ == '__main__':
    output = torch.rand(10, 3, 224, 224)
    label = torch.ones(10, 3, 224 * 224)
    print(SoftDice_Loss().forward(output, label))
    print((Focal_loss(0.5,1).forward(output,label)))

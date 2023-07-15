import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcNet(nn.Module):
    def __init__(self,feature_dim=2,cls_dim=10):
        super().__init__()
        #生成一个隔离带向量，训练这个向量和原来的特征向量尽量分开，达到增加角度的目的
        self.W=nn.Parameter(torch.randn(feature_dim,cls_dim).cuda(),requires_grad=True)
    def forward(self, feature,m=0.5,s=64):
        #对特征维度进行标准化
        x = F.normalize(feature,dim=1)#shape=【100，2】
        w = F.normalize(self.W, dim=0)#shape=【2，10】

        # s = torch.sqrt(torch.sum(torch.pow(x, 2))) * torch.sqrt(torch.sum(torch.pow(w, 2)))
        cosa = torch.matmul(x, w)/s

        a=torch.acos(cosa)#反三角函数得出的是弧度，而非角度，1弧度=1*180/3.14=57角度
        # 这里对e的指数cos(a+m)再乘回来，让指数函数的输出更大，
        # 从而使得arcsoftmax输出更小，即log_arcsoftmax输出更小，则-log_arcsoftmax更大。
        arcsoftmax = torch.exp(
            s * torch.cos(a + m)) / (torch.sum(torch.exp(s * cosa), dim=1, keepdim=True) - torch.exp(
            s * cosa) + torch.exp(s * torch.cos(a + m)))

        return arcsoftmax

if __name__ == '__main__':

    arc=ArcNet(feature_dim=2,cls_dim=10)
    feature=torch.randn(100,2).cuda()
    out=arc(feature)
    print(feature.shape)
    print(out.shape)


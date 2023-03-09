import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcNet(nn.Module):
    def __init__(self,feature_dim=2,cls_dim=10):
        super().__init__()
        #生成一个隔离带向量，训练这个向量和原来的特征向量尽量分开，达到增加角度的目的
        self.W=nn.Parameter(torch.randn(feature_dim,cls_dim).cuda(),requires_grad=True)
    def forward(self, feature,m=0.5,s=64):
        ''
        '''由于x是模型中经过BN之后的特征，已经做过均值化了，w是随机变量，所以已经具有取均值的特性，
        所以不用再去均值了，只需要做一下范数归一化压缩特征即可，即除以二范数的最大值：normalize'''
        #对特征维度进行二范数最大值标准化
        x = F.normalize(feature,dim=1)#shape=【100，2】
        w = F.normalize(self.W, dim=0)#shape=【2，10】
        # print(x.shape)
        # print(w.shape)
        # s=30
        # s = torch.sqrt(torch.sum(torch.pow(x, 2))) * torch.sqrt(torch.sum(torch.pow(w, 2)))  # ||x||*||w||
        # print(torch.sqrt(torch.sum(torch.pow(x, 2))))
        # print(torch.sqrt(torch.sum(torch.pow(w, 2))))
        # print(s)
        # 缩小torch.matmul(x, w)的值，防止出现-1和1，相当于防止出现0度和180度，变相的将cosa变小，防止acosa梯度爆炸
        cosa = torch.matmul(x, w)/s
        # print(torch.max(torch.matmul(x,w)))
        # print(torch.min(torch.matmul(x,w)))
        # print(cosa)#[-1,1]

        "标准化后的x(-1,1)再求平方(1,1)，相当于求它的单位向量(1)，所以求x的平方和就是批次100*1=100"
        "同理标准化后的w有10个维度，就等于10*1=10"
        "所以s就等于sqrt(100)*sqrt(10)≈31.6"
        # print(torch.sum(torch.pow(x,2)),torch.sum(torch.pow(w,2)))
        a=torch.acos(cosa)#反三角函数得出的是弧度，而非角度，1弧度=1*180/3.14=57角度
        # 这里把cosa的缩放指数s再乘回来，还原真实的cosa，这会让指数函数的输出更大，
        # 从而使得arcsoftmax输出更小，即log_arcsoftmax输出更小，则-log_arcsoftmax更大。
        # m=0.5
        arcsoftmax = torch.exp(
            s * torch.cos(a + m)) / (torch.sum(torch.exp(s * cosa), dim=1, keepdim=True) - torch.exp(
            s * cosa) + torch.exp(s * torch.cos(a + m)))
        # print(arcsoftmax)
        # 这里arcsomax的概率和不为1，小于1。这会导致交叉熵损失看起来很大，且最优点损失也很大
        # print(torch.sum(arcsoftmax, dim=1))
        # exit()

        return arcsoftmax

if __name__ == '__main__':

    arc=ArcNet(feature_dim=2,cls_dim=10)
    feature=torch.randn(100,2).cuda()
    out=arc(feature)
    print(feature.shape)
    print('out',out.shape)


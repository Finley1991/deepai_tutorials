from Gpt2_Poem.module import GPT2
from Gpt2_Poem.dataset import MyDataset
import Gpt2_Poem.config as cfg
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader

def weight_init(m):
    #判断输入的参数m和Linear层参数的类型是否一致
    if isinstance(m, nn.Linear):
        #初始化参数：参数通过网络层时，让输入和输出的方差相同，防止方差为0，或者方差过大
        #sigmoid激活函数，方差为0，失去非线性能力，方差太大，造成梯度弥散
        #relu激活函数，方差为0，导数可能为0(梯度消失)，方差太大，负半轴神经元为0
        #leaky relu激活函数，方差为0和方差太大，影响不大
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class Trainer:
    def __init__(self,isMask):
        self.batch_size = 4
        self.epoch = 50
        self.gpt2 = GPT2(isMask)

        # self.net = nn.DataParallel(self.gpt2, device_ids=[0, 2, 3])
        self.net = self.gpt2
        self.net.to(torch.device(cfg.device))

        # 网络
        self.weight_file = r"./weight/weight_Mask.pth"
        if os.path.exists(self.weight_file) and os.path.getsize(self.weight_file) != 0:
            self.net.load_state_dict(torch.load(self.weight_file))
            print("加载保存的参数成功")
        else:
            self.net.apply(weight_init)
            print("加载随机参数成功")
        self.loss_fn=nn.CrossEntropyLoss()#会对标签自动做one-hot
        self.opt = torch.optim.Adam(self.net.parameters(), lr=0.0001)

    def train(self):
        myDataset = MyDataset(r"./data/books_tokenized")
        for epoch in range(self.epoch):
            for i, (x, y) in enumerate(DataLoader(myDataset, batch_size=self.batch_size, shuffle=True, num_workers=4)):
                # [N,52]:【0，51】【1，52】
                # print(x.shape,y.shape)#[4,51],[4,51]
                x, y = x.to(torch.device(cfg.device)), y.to(torch.device(cfg.device))

                # 创建一个位置，存放词，词的位置编码，
                p = torch.arange(0, x.shape[1])[None, :].repeat(x.shape[0], 1).to(torch.device(cfg.device))
                # print(p.shape)#[4,51]:0123...50
                # print(p)

                #把字向量，和位置编码传入到网络中，得到输出结果
                # 把输出的形状变成和标签一致：
                _y = self.net(x, p).reshape(-1, cfg.vocab_num)
                # print(y.shape)  # NV:[4,51]
                y = y.reshape(-1)
                # print(self.net(x, p).shape)#NSV:[4, 51, 305]
                # print(self.net(x, p).reshape(-1, cfg.vocab_num).shape)#nv:[204, 305]
                # print(y.shape)#NV:[204,]
                loss = self.loss_fn(_y, y)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                print(epoch, i, "-", int(len(myDataset)/self.batch_size), loss.cpu().detach().item())
                if i % 100 == 0:
                    # torch.save(self.net.state_dict(), self.weight_file)
                    print("保存参数成功")


if __name__ == '__main__':
    trainer = Trainer(isMask=True)
    trainer.train()

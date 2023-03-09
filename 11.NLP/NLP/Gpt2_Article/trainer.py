from NLP.Gpt2_Article.module import GPT2
from NLP.Gpt2_Article.dataset import MyDataset
import NLP.Gpt2_Article.config as cfg
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class Trainer:
    def __init__(self):
        self.batch_size = 1
        self.epoch = 1000
        self.gpt2 = GPT2()

        # self.net = nn.DataParallel(self.gpt2, device_ids=[0, 2, 3])
        self.net = self.gpt2
        self.net.to(torch.device(cfg.device))

        # 网络
        self.weight_file = r"./weight/weight3.pth"
        if os.path.exists(self.weight_file) and os.path.getsize(self.weight_file) != 0:
            self.net.load_state_dict(torch.load(self.weight_file))
            print("加载保存的参数成功")
        else:
            self.net.apply(weight_init)
            print("加载随机参数成功")

        self.opt = torch.optim.Adam(self.net.parameters(), lr=0.0001)

    def train(self):
        myDataset = MyDataset(r"./data/books_tokenized")
        for epoch in range(self.epoch):
            for i, (x, y) in enumerate(DataLoader(myDataset, batch_size=self.batch_size, shuffle=True, num_workers=4)):
                # 【0，25】【1，26】
                x, y = x.to(torch.device(cfg.device)), y.to(torch.device(cfg.device))

                # 造一个位置，存放词，词的位置编码，
                p = torch.arange(0, x.shape[1])[None, :].repeat(x.shape[0], 1).to(torch.device(cfg.device))
                #把字向量，和位置编码传入到网络中，得到预测输出
                # 把输出的形状变成和标签一致：【1，24】，307
                _y = self.net(x, p).reshape(-1, cfg.vocab_num)
                y = y.reshape(-1)
                loss = F.cross_entropy(_y, y)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                print(epoch, i, "-", int(len(myDataset)/self.batch_size), loss.cpu().detach().item())
                if i % 100 == 0:
                    torch.save(self.net.state_dict(), self.weight_file)
                    print("保存参数成功")


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()

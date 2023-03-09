import torch.utils.data
import torchaudio
from torch.nn import functional as F
import os
import numpy as np

def normalize(tensor):
    tensor_minusmean = tensor - tensor.mean()
    return tensor_minusmean / tensor_minusmean.abs().max()

#从音频信号创建Mel-frequency倒谱系数
#包括数据采样率，梅尔频率倒谱上的采样率，傅里叶变换类型等等
tf = torchaudio.transforms.MFCC(sample_rate=8000,n_mfcc=128)#保留的梅尔频率倒谱系数的数量
#LFCC：线性频率倒谱系数

class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, (1, 3), (1, 2), (0, 1)),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 4, (1, 3), (1, 2), (0, 1)),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 4, (1, 3), (1, 2), (0, 1)),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 8, 3, 2, 1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, 3, 2, 1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 1, (8, 1)),
        )

    def forward(self, x):
        h = self.seq(x)
        return h.reshape(-1, 8)


if __name__ == '__main__':
    dataset = torchaudio.datasets.YESNO(r"./", download=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    print(len(data_loader))
    print(len(list(data_loader)[0]),list(data_loader)[0])
    net = Net()
    if os.path.exists("./wave.pth"):
        net.load_state_dict(torch.load("./wave.pth"))
        print("Loaded Params!")
    else:
        print("No Params!")
    opt = torch.optim.Adam(net.parameters())
    loss_fn = torch.nn.MSELoss()

    for epoch in range(1000):
        datas = []
        tags = []
        for data, _, tag in data_loader:
            # print(data.shape)
            # print(np.shape(tag))
            tag_ = torch.stack(tag, dim=1).float()
            specgram = normalize(tf(data))
            # print(specgram.shape)
            # print(tag_.shape)

            #自适应池化，将数据对齐，然后可以批量处理
            datas.append(F.adaptive_avg_pool2d(specgram, (32, 256)))
            tags.append(tag_)

        #合并形成批次
        specgrams = torch.cat(datas, dim=0)
        tags_ = torch.cat(tags, dim=0)
        # print(specgrams.shape)
        # print(tags_.shape)
        y = net(specgrams)
        loss = loss_fn(y, tags_)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 100 == 0:
            print(epoch, loss.item())
            print(y[0])
            a=(y[0].data>0.5).float()
            print(a)
            print(tags_[0])
            # torch.save(net.state_dict(), "./wave.pth")
            # print("saved params!")

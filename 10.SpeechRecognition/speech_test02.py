import torch.utils.data
import torchaudio
from torch.nn import functional as F
import os

def normalize(tensor):
    tensor_minusmean = tensor - tensor.mean()
    return tensor_minusmean / tensor_minusmean.abs().max()

#使用梅尔频率倒谱系数采样
#包括数据采样率，频率的采样率，傅里叶变换类型等等
tf = torchaudio.transforms.MFCC(sample_rate=8000,n_mfcc=128)

print(tf)
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
            torch.nn.BatchNorm2d(1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(1, 1, (1, 8)),
        )

    def forward(self, x):
        # print(x.shape)
        h = self.seq(x)
        # print(h.shape)
        return h.reshape(-1)


if __name__ == '__main__':
    dataset = torchaudio.datasets.SPEECHCOMMANDS(r"./", download=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    print(len(data_loader))
    # print(len(list(data_loader)[0]))
    net = Net().cuda()
    if os.path.exists("./speech2.pth"):
        net.load_state_dict(torch.load("./speech2.pth"))
        print("Loaded Params!")
    else:
        print("No Params!")
    opt = torch.optim.Adam(net.parameters())
    loss_fn = torch.nn.MSELoss()

    for epoch in range(100):
        datas = []
        tags = []
        for i,(data, _,_,_, tag) in enumerate(data_loader):
            tag_ = torch.stack((tag,), dim=1).float().cuda()
            specgram = normalize(tf(data)).cuda()
            datas.append(F.adaptive_avg_pool2d(specgram, (32, 256)))
            tags.append(tag_)

            specgrams = torch.cat(datas, dim=0)
            tags_ = torch.cat(tags, dim=0).reshape(-1)
            y = net(specgrams)
            # print(y.shape)
            # print(tags_.shape)
            loss = loss_fn(y, tags_)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % 100 == 0:
                print(epoch, loss.item())
                print("epoch:",epoch,"i:",i)
                print(y[:10])
                a=y[:10].data.round().abs()
                print(a)
                print(tags_[:10])
                # torch.save(net.state_dict(), "./speech2.pth")
                # print("saved params!")


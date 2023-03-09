import torchvision.models as models
from torch import nn
import torch
from torch.nn import functional as F
from dataset import *
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score

# torch.manual_seed(0)
class Arcsoftmax(nn.Module):
    def __init__(self, feature_num, cls_num):
        super().__init__()
        self.w = nn.Parameter(torch.randn((feature_num, cls_num)),requires_grad=True)
    def forward(self, x, s=64, m=0.5):
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.w, dim=0)

        cosa = torch.matmul(x_norm, w_norm) / s
        a = torch.acos(cosa)

        arcsoftmax = torch.exp(
            s * torch.cos(a + m)) / (torch.sum(torch.exp(s * cosa), dim=1, keepdim=True) - torch.exp(
            s * cosa) + torch.exp(s * torch.cos(a + m)))

        return arcsoftmax


class FaceNet(nn.Module):

    def __init__(self):
        super(FaceNet, self).__init__()
        self.feature_net = models.shufflenet_v2_x1_0(pretrained=True)
        # print(self.feature_net)
        self.feature_net.fc = nn.Linear(1024, 512)
        self.arc_softmax = Arcsoftmax(512, 160)

    def forward(self, x):

        feature = self.feature_net(x)
        return feature, self.arc_softmax(feature)

    def encode(self, x):
        return self.feature_net(x)


def compare(face1, face2):
    face1_norm = F.normalize(face1)
    face2_norm = F.normalize(face2)
    print(face1_norm.shape)
    print(face2_norm.shape)
    cosa = torch.matmul(face1_norm, face2_norm.T)
    # cosa = torch.dot(face1_norm.reshape(-1), face2_norm.reshape(-1))
    return cosa

if __name__ == '__main__':

    # 训练过程
    save_path = "params/face_params.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = FaceNet().to(device)

    if os.path.exists(save_path):
        net.load_state_dict(torch.load(save_path))
    else:
        print("NO Param")

    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(net.parameters())

    dataset = MyDataset(r"data")
    dataloader = DataLoader(dataset=dataset, batch_size=10, shuffle=True,num_workers=4)

    #训练
    train_bool=0
    if train_bool:
        for epoch in range(3):
            losses = []
            clses = []
            yses = []
            for xs, ys in dataloader:
                feature, cls = net(xs.to(device))
                loss = loss_fn(torch.log(cls), ys.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                clses.extend(np.argmax(cls.cpu().detach().numpy(), axis=1))
                yses.extend(ys)
            print(np.array(yses))
            print(np.array(clses))
            # acc=accuracy_score(yses,clses)
            acc = np.sum(np.array(yses) == np.array(clses)) / len(yses)
            print("Epoch:", epoch, "acc:", acc, "Loss:", np.array(losses).mean())
            if epoch % 2 == 0:
                torch.save(net.state_dict(), save_path)
                print(str(epoch) + "参数保存成功")

    # 使用
    net = FaceNet().to(device)
    net.load_state_dict(torch.load(save_path))
    net.eval()

    person1 = tf(Image.open("data/0/000_0.bmp")).to(device)
    person1_feature = net.encode(torch.unsqueeze(person1,0))
    # person1_feature = net.encode(person1[None, ...])

    person2 = tf(Image.open("data/1/001_1.bmp")).to(device)
    person2_feature = net.encode(person2[None, ...])

    siam = compare(person1_feature, person2_feature)
    print(siam.item())

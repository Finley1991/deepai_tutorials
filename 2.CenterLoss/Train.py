import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
from Net_Model import Net
from loss import center_loss
import os
import numpy as np

if __name__ == '__main__':

    save_path = "params/net_center.pth"
    train_data = torchvision.datasets.MNIST(root="./MNIST", download=True, train=True,
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5,],std=[0.5,])]))
    train_loader = data.DataLoader(dataset=train_data, shuffle=True, batch_size=100,num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)

    if os.path.exists(save_path):
        net.load_state_dict(torch.load(save_path))
    else:
        print("NO Param")
    '''
    nn.NLLLoss输入是一个对数概率向量和一个目标标签，
    它与nn.CrossEntropyLoss的关系可以描述为：softmax(x)+log(x)+nn.NLLLoss====>nn.CrossEntropyLoss
    '''
    'CrossEntropyLoss()=torch.log(torch.softmax(None))+nn.NLLLoss()'
    'CrossEntropyLoss()=log_softmax() + NLLLoss() '
    'nn.CrossEntropyLoss()是nn.logSoftmax()和nn.NLLLoss()的整合'
    # lossfn_cls = nn.CrossEntropyLoss()
    lossfn_cls = nn.NLLLoss()
    # optimzer = torch.optim.Adam(net.parameters())
    optimzer = torch.optim.SGD(net.parameters(),lr=1e-3, momentum=0.9)
    # optimzer = torch.optim.SGD(net.parameters(),lr=1e-3)
    # optimzer = torch.optim.SGD(net.parameters(),lr=1e-3, momentum=0.9, weight_decay=0.0005)

    epoch = 0
    while True:
        feat_loader = []
        label_loader = []
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            feature,output = net.forward(x)
            # print(feature.shape)#[N,2]
            # print(output.shape)#[N,10]
            # print(y.shape)#[N]
            # center = nn.Parameter(torch.randn(output.shape[1], feature.shape[1]))
            # print(center.shape)#[10,2]

            loss_cls = lossfn_cls(output, y)
            # print(output.dtype)
            # print(y.dtype)

            loss_center = center_loss(feature,y,2)

            loss = loss_cls+loss_center
            optimzer.zero_grad()
            loss.backward()
            optimzer.step()

            # feature.shape=[100,2]
            #y.shape=[100]
            feat_loader.append(feature)
            label_loader.append(y)

            if i % 600 == 0:
                print("epoch:",epoch,"i:",i,"total:",loss.item(),"softmax_loss:",loss_cls.item(),"center_loss:",loss_center.item())

        feat = torch.cat(feat_loader, 0)
        labels = torch.cat(label_loader, 0)
        '---------------'
        # print(feat.shape)#feat.shape=[60000,2]
        # print(labels.shape)#feat.shape=[60000]
        '-------------------'
        net.visualize(feat.data.cpu().numpy(), labels.data.cpu().numpy(), epoch)
        epoch+=1
        # torch.save(net.state_dict(), save_path)
        if epoch==150:
            break

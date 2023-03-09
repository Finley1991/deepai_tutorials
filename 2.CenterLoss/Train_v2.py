import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
from Net_Model import Net
from loss_class import Centerloss
import os
import numpy as np

if __name__ == '__main__':

    save_path = "params/net_center_v2.pth"
    train_data = torchvision.datasets.MNIST(root="./MNIST", download=True, train=True,
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.1307,],std=[0.3081,])]))
    train_loader = data.DataLoader(dataset=train_data, shuffle=True, batch_size=100,num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)

    if os.path.exists(save_path):
        net.load_state_dict(torch.load(save_path))
    else:
        print("NO Param")

    'CrossEntropyLoss()=torch.log(torch.softmax(None))+nn.NLLLoss()'
    'CrossEntropyLoss()=log_softmax() + NLLLoss() '
    'nn.CrossEntropyLoss()是nn.logSoftmax()和nn.NLLLoss()的整合'
    # lossfn_cls = nn.CrossEntropyLoss()
    lossfn_class = nn.NLLLoss()
    "Centerloss()是一个网络，也需要检查cuda"
    lossfn_center = Centerloss(feature_num=2,class_num=10,lambdas=2).to(device)
    optimzer_softmax = torch.optim.SGD(net.parameters(),lr=1e-3, momentum=0.9)
    "定义专门优化centerloss里的nn.Parameter权重的优化器"
    optimzer_cenetr = torch.optim.SGD(lossfn_center.parameters(),lr=0.5)
    #学习率衰减
    scheduler = lr_scheduler.StepLR(optimzer_softmax, 20, gamma=0.9)

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
            # center = nn.Parameter(torch.randn(output.shape[1], feature.shape[1]))
            # print(center.shape)#[10,2]

            loss_class = lossfn_class(output, y)

            loss_center = lossfn_center(feature,y)

            loss = loss_class+loss_center
            optimzer_softmax.zero_grad()
            optimzer_cenetr.zero_grad()
            loss.backward()
            optimzer_softmax.step()
            optimzer_cenetr.step()


            # feature.shape=[100,2]
            #y.shape=[100]
            feat_loader.append(feature)
            label_loader.append(y)

            if i % 600 == 0:
                print("epoch:",epoch,"i:",i,"total:",loss.item(),"softmax_loss:",loss_class.item(),"center_loss:",loss_center.item())

        feat = torch.cat(feat_loader, 0)
        labels = torch.cat(label_loader, 0)
        '---------------'
        # print(np.shape(feat_loader))#feat_loader.shape=[600,]=[[100,2],[100,2],...]600
        # print(feat.shape)#feat.shape=[60000,2]
        # print(np.shape(label_loader))#feat_loader.shape=[600,]=[[100],[100],...]600
        # print(labels.shape)#feat.shape=[60000]
        '-------------------'
        net.visualize(feat.data.cpu().numpy(), labels.data.cpu().numpy(), epoch)
        epoch+=1
        # torch.save(net.state_dict(), save_path)
        "更新学习率"
        lr = scheduler.get_last_lr()
        scheduler.step()
        print(epoch, scheduler.get_last_lr()[0])
        if epoch==50:
            break

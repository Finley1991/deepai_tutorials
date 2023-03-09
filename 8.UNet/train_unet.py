import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import model_UNet
from sample_data import Sampling_data
from torchvision.utils import save_image
from untils.dice import dice
from untils.segmentation_loss import SoftDice_Loss,Focal_loss

if __name__ == '__main__':

    img_path = r'D:\pycharmprojects\dataset\VOCdevkit\VOC2012'
    params_path = r'params/module.pth'
    img_save_path = r'./train_img'
    epoch = 1
    dataloader = DataLoader(Sampling_data(img_path,416), 2, shuffle=True,drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model_UNet.MainNet().to(device)
    if os.path.exists(params_path):
        net.load_state_dict(torch.load(params_path))
    else:
        print("No parameters!")

    optimizer = torch.optim.Adam(net.parameters())
    # optimizer = torch.optim.SGD(net.parameters(),lr=1e-3,momentum=0.9)
    # mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    dice_loss=SoftDice_Loss()
    focal_loss=Focal_loss(0.5,2)

    if not os.path.exists(img_save_path):
        os.mkdir(img_save_path)

    if not os.path.exists("./params"):
        os.mkdir("./params")

    while True:
        for i, (xs, ys) in enumerate(dataloader):
            xs = xs.to(device)
            ys = ys.to(device)
            xs_ = net(xs)

            loss = focal_loss(xs_, ys)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print('epoch:{},count:{},loss:{:.3f}'.format(epoch, i, loss))

                x = xs[0]
                x_ = xs_[0]
                y = ys[0]

                # print(y.shape)
                "将三张图像堆叠起来，便于保存"#三维变四维，堆叠后多一个维度
                img = torch.stack([x,x_,y],0)
                save_image(img.cpu(), os.path.join(img_save_path,'{}.png'.format(epoch)))


        torch.save(net.state_dict(), params_path)
        print('Model parameters saved successfully !')
        epoch += 1
        if epoch == 300:
            break
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import model_UNet
from sample_data import Sampling_data
from torchvision.utils import save_image


if __name__ == '__main__':

    img_path = r'E:\VOCtrainval_11-May-2012\VOCdevkit\VOC2012'
    params_path = r'params/module.pth'
    img_save_path = r'./test_img'

    dataloader = DataLoader(Sampling_data(img_path,416), 1, shuffle=True,drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model_UNet.MainNet().to(device)
    if os.path.exists(params_path):
        net.load_state_dict(torch.load(params_path))
    else:
        print("No parameters!")

    if not os.path.exists(img_save_path):
        os.mkdir(img_save_path)
    net.eval()
    for i, (xs, ys) in enumerate(dataloader):
        x = xs.to(device)
        y = ys.to(device)
        x_ = net(x)

        # print(y.shape)
        "将三张图像拼起来，便于保存"#四维还是四维，在第0轴拼接
        img = torch.cat([x,x_,y],0)
        save_image(img.cpu(), os.path.join(img_save_path,'{}.png'.format(i)))
        print(i)
       
        if i == 10:
            break


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
from torchvision import datasets,transforms
from Denoising_AE.NetEncoder import Encoder_Net
from Denoising_AE.NetDecoder import Decoder_Net
from Denoising_AE.add_noise import gasuss_noise_func
import matplotlib.pyplot as plt

num_epoch = 10
if __name__ == '__main__':

    if not os.path.exists("./params"):
        os.mkdir("./params")
    if not os.path.exists("./img"):
        os.mkdir("./img")
    trans = transforms.Compose([
        transforms.ToTensor()
    ])
    mnist_data = datasets.MNIST("../data",train=True,
                                transform=trans,download=True)
    train_loader = DataLoader(mnist_data, 100,shuffle=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    en_net = Encoder_Net().to(device)
    de_net = Decoder_Net().to(device)

    en_net.train()
    de_net.train()

    # en_net.load_state_dict(
    #     torch.load("./params/en_net.pth"))
    # de_net.load_state_dict(
    #     torch.load("./params/de_net.pth"))

    loss_fn = nn.MSELoss()
    en_optimizer = torch.optim.Adam(en_net.parameters())
    de_optimizer = torch.optim.Adam(de_net.parameters())
    # en_optimizer = torch.optim.SGD(en_net.parameters(),lr=1e-3)
    # de_optimizer = torch.optim.SGD(de_net.parameters(),lr=1e-3)

    for epoch in range(num_epoch):
        for i, (img,label) in enumerate(train_loader):
            raw_img = img.to(device)
            "转成numpy格式的形状"
            img = img.permute(0,2,3,1)*255
            # print(img.shape)
            "获取高斯噪声"
            noise_img = gasuss_noise_func(img)
            "轴变换，转Tensor"
            # print(img.shape)
            noise_img = torch.Tensor(noise_img.transpose(0,3,1,2))/255
            # print(noise_img.shape,torch.max(noise_img))
            # exit()
            noise_img = noise_img.to(device)
            feature = en_net(noise_img)
            out_img = de_net(feature)
            # print(out_img.shape)
            loss = loss_fn(raw_img,out_img)
            en_optimizer.zero_grad()
            de_optimizer.zero_grad()

            loss.backward()
            en_optimizer.step()
            de_optimizer.step()

            if i%100 == 0:
                print('Epoch [{}/{}], loss: {:.3f}'
                      .format(epoch, num_epoch, loss))

            # images = out_img.cpu().data
            # show_images = images.permute([0,2,3,1])
            # plt.imshow(show_images[0].reshape(28,28))
            # plt.pause(1)

        fake_images = out_img.cpu().data
        save_image(fake_images, './img/{}-fake_images.png'
                   .format(epoch + 1),nrow=10)
        raw_imgs = raw_img.cpu().data
        save_image(raw_imgs, './img/{}-raw_images.png'
                   .format(epoch + 1), nrow=10)
        noise_images = noise_img.cpu().data
        save_image(noise_images, './img/{}-noise_images.png'
                   .format(epoch + 1), nrow=10)
        torch.save(en_net.state_dict(), "./params/en_net.pth")
        torch.save(de_net.state_dict(), "./params/de_net.pth")



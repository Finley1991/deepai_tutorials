import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
from torchvision import datasets,transforms
from AE.autoencoder import Main_Net

num_epoch = 10
if __name__ == '__main__':

    if not os.path.exists("params"):
        os.mkdir("params")
    if not os.path.exists("./img"):
        os.mkdir("./img")
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5,],[0.5,])
    ])
    mnist_data = datasets.MNIST("../data",train=True,
                                transform=trans,download=True)
    train_loader = DataLoader(mnist_data, 100,shuffle=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    net = Main_Net().to(device)

    net.train()
    if os.path.exists("./params/net.pth"):
        net.load_state_dict(
            torch.load("./params/net.pth"))
    else:
        print("No Params!")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters())

    for epoch in range(num_epoch):
        for i, (img,_) in enumerate(train_loader):
            img = img.to(device)
            out_img = net(img)
            # print(out_img.shape)
            loss = loss_fn(img,out_img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%100 == 0:
                print('Epoch [{}/{}], loss: {:.3f}'
                      .format(epoch, num_epoch, loss))

        fake_images = out_img.cpu().data
        save_image(fake_images, './img/{}-fake_images.png'
                   .format(epoch + 1),nrow=10)
        real_images = img.cpu().data
        save_image(real_images, './img/{}-real_images.png'
                   .format(epoch + 1), nrow=10)
        torch.save(net.state_dict(), "./params/net.pth")


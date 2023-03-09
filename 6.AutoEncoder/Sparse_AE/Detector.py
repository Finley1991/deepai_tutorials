import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
from torchvision import datasets,transforms
from CNN_test.AutoEncode.Sparse_AE.NetEncoder import Encoder_Net
from CNN_test.AutoEncode.Sparse_AE.NetDecoder import Decoder_Net

num_epoch = 10
if __name__ == '__main__':

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

    en_net.eval()
    de_net.eval()

    en_net.load_state_dict(
        torch.load("./params/en_net.pth"))
    de_net.load_state_dict(
        torch.load("./params/de_net.pth"))

    loss_fn = nn.MSELoss()
    for epoch in range(num_epoch):
        for i, (img,label) in enumerate(train_loader):
            img = img.to(device)
            feature = en_net(img)
            out_img = de_net(feature)
            loss = loss_fn(img,out_img)
            print('Epoch [{}/{}], loss: {:.3f}'
                      .format(epoch, num_epoch, loss))
            images = out_img.cpu().data

            show_images = images.permute([0,2,3,1])
            # show_images = torch.transpose(images,1,3)
            # plt.imshow(show_images[0].reshape(28,28))
            # plt.pause(1)
            fake_images = out_img.cpu().data
            save_image(fake_images, './img/{}-fake_images.jpg'
                       .format(i + 1),nrow=10)
            real_images = img.cpu().data
            save_image(real_images, './img/{}-real_images.jpg'
                       .format(i + 1), nrow=10)



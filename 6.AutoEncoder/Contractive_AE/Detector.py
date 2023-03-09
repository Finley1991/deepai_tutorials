import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import PIL.Image as pimg
import numpy as np
from torchvision import datasets,transforms
from CNN_test.AutoEncode.Contractive_AE.NetEncoder import Encoder_Net
from CNN_test.AutoEncode.Contractive_AE.NetDecoder import Decoder_Net

num_epoch = 10
if __name__ == '__main__':
    images_path = r"./noise_img"
    if not os.path.exists("./img"):
        os.mkdir("./img")

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
    for i,img in enumerate(os.listdir(images_path)):
        img = pimg.open(os.path.join(images_path,img))
        img = img.convert("L")
        img = np.array(img)
        img = img.reshape([-1,28,28,1])
        img = torch.Tensor(img.transpose(0, 3, 1, 2)) / 255
        img = img.to(device)
        feature = en_net(img)
        out_img = de_net(feature)

        # images = out_img.cpu().data
        # show_images = images.permute([0,2,3,1])
        # plt.imshow(show_images[0].reshape(28,28))
        # plt.pause(1)
        # print(out_img.shape)
        fake_images = out_img.cpu().data
        save_image(fake_images, './img/{}-detect_fake_images.jpg'
                   .format(i + 1),nrow=1)
        noise_images = img.cpu().data
        save_image(noise_images, './img/{}-detect_noise_images.jpg'
                   .format(i + 1), nrow=1)


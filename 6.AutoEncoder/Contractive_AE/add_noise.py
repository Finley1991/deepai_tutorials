from torch.utils.data.dataloader import DataLoader
from torchvision.utils import save_image
from torchvision import datasets,transforms
import numpy as np
import random
import os
import PIL.Image as pimg
import matplotlib.pyplot as plt
def save_rawimg():
    trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,],std=[0.5,])
        ])

    mnist_data = datasets.MNIST("../data",train=True,
                                    transform=trans,download=True)
    load_data = DataLoader(mnist_data, 1,shuffle=True)

    for i, (img,_) in enumerate(load_data):
        # "保存时自动转换图像像素到0-255，RGB通道"
        save_image(img,"./raw_img/{0}.jpg".format(i),nrow=1)
        if i > 10:
            break

def gasuss_noise_img(images, mean=0, var=127):
    for filename in os.listdir(images):
        image = pimg.open(os.path.join(images,filename))
        image = np.array(image, dtype=np.float32)
        # print(np.max(image))
        # "获取三个通道的噪声数据：均值，方差，数据形状维度"
        # "相当于新生成了一张大小形状一样的图片"

        "增加高斯噪声：正负值：偏彩色"
        noise = np.random.normal(mean, var, image.shape)
        "增加正值:偏白"
        # noise1 = np.random.randint(mean, var, image.shape)
        "增加正值：偏黑"
        # noise2 = np.random.randint(mean, var, image.shape)-var*2
        out = image + noise
        # out = image + noise1
        # out = image + noise2
        # out = image + noise1 + noise2
        # "加了噪声的数据范围超过了（0，255），所以截取（0，255）之间的数据即可"
        # "把小于0的归为0，大于255的归于255"
        out = np.clip(out, 0, 255)
        # print(out.shape,out.dtype)
        out = pimg.fromarray(np.uint8(out))
        # out.show()
        out.save("{0}/{1}".format("./noise_img",filename))

def gasuss_noise_func(images, mean=0, var=127):
    images_list = []
    for i,image in enumerate(images):
        # print(image)
        image = np.array(image, dtype=np.float32)
        # print(np.max(image),image.dtype)
        noise = np.random.normal(mean, var, image.shape)
        out_arr = image + noise
        out_arr = np.clip(out_arr, 0, 255)
        # print(out.shape,out.dtype)
        # out_img = pimg.fromarray(np.uint8(out_arr.reshape([28,28])))
        # # out_img.show()
        # out_img.save("{0}/{1}.jpg".format("./noise_img",i))
        images_list.append(out_arr)
    images_arr = np.array(images_list)
    return images_arr

if __name__ == '__main__':
    save_rawimg()
    gasuss_noise_img("./raw_img")


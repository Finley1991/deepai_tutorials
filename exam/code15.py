import numpy as np
from PIL import Image

def crop(img_file, mask_file):
    img_array = np.array(Image.open(img_file))
    mask = np.array(Image.open(mask_file))

    #从mask中随便找一个通道，cat到原图的RGB通道后面，转成RGBA通道模式
    res = np.concatenate((img_array, mask[:, :, [0]]), -1)
    img = Image.fromarray(res.astype('uint8'), mode='RGBA')
    return img

if __name__ == "__main__":
    img_file = "1.jpg"
    mask_root = "2.png"
    res = crop(img_file,mask_root)
    print(res.mode)
    res.show()
    res.save("./{}.png".format("result"))


from xml.dom.minidom import parse
import os
import numpy as np
import PIL.Image as pimg
from PIL import ImageDraw
import matplotlib.pyplot as plt

"""
此模块作用：
    1、对图片重新命名，并且等比例缩放，粘贴为正方形大小图片
    2、对xml文件进行解析，并对标签框box按图像缩放比例缩放，获得缩放后的txt标签文件
"""


def class_name():
    # return ["人","老虎"]
    return ["人", "狮子", "老虎", "豹子"]


"按照要缩放的边长对图片等比例缩放，并转成正方形居中"


def scale_img(img, scale_side):
    # "获得图片宽高"
    w1, h1 = img.size
    # print(w1,h1)
    # "根据最大边长缩放,图像只会被缩小，不会变大"
    # "当被缩放的图片宽和高都小于缩放尺寸的时候，图像不变"
    img.thumbnail((scale_side, scale_side))
    # "获得缩放后的宽高"
    w2, h2 = img.size
    # print(w2,h2)
    # "获得缩放后的比例"
    s1 = w1 / w2
    s2 = h1 / h2
    s = (s1 + s2) / 2
    # "新建一张scale_side*scale_side的空白黑色背景图片"
    bg_img = pimg.new("RGB", (scale_side, scale_side), (0, 0, 0))
    # "根据缩放后的宽高粘贴图像到背景图上"
    bg_img.paste(img, (0, 0))

    return w2, h2, s, bg_img


"按照图片的缩放比对box进行缩放"


def scale_box(w, h, s, scale_side, x1, y1, x2, y2):
    # "宽和高都小于416的时候"
    if w < scale_side and h < scale_side:
        x1_ = x1
        y1_ = y1
        x2_ = x2
        y2_ = y2

    else:
        x1_ = x1 / s
        y1_ = y1 / s
        x2_ = x2 / s
        y2_ = y2 / s

    return x1_, y1_, x2_, y2_


def process_data(raw_img, raw_label, save_img, save_label, scale_side):
    count = 1
    "打开txt文件，等待写入文件信息"
    f = open(save_label, "w")
    for filename in os.listdir(raw_label):
        # "获得每个xml文件的对象和地址，解析每一个xml文件"
        dom = parse("{0}/{1}".format(raw_label, filename))
        # "获得每个xml文件的元素根节点"
        root = dom.documentElement
        # "根据根节点与图片名的元素标签获得图片名"
        img_name = root.getElementsByTagName("path")[0].childNodes[0].data
        # "使用文件的根节点获得所有元素节点，再根据元素节点中的名称获得该元素节点下对应的值"
        # "获得图片尺寸的元素列表，从列表格式中拿出值"
        img_size = root.getElementsByTagName("size")[0]
        # "获得标签框对象的元素列表，从根节点下获得object节点中的子节点，也就是所有的item"
        items = root.getElementsByTagName("item")
        # "从size元素节点中通过子元素名称width获得img_w,从width中取出值"
        img_w = img_size.getElementsByTagName("width")[0].childNodes[0].data
        img_h = img_size.getElementsByTagName("height")[0].childNodes[0].data
        img_c = img_size.getElementsByTagName("depth")[0].childNodes[0].data
        # print(img_w,img_h,img_c)
        name_list = class_name()

        # "根据xml保存的图片路径获得图片名"
        # print(img_name)
        img_name = img_name.split("\\")[-1]
        # print(img_name)
        # "打开图片"
        img = pimg.open(os.path.join(raw_img, img_name))
        # img.show()
        "输入图片和要缩放的尺寸，获得图片缩放后的宽和高,缩放比，缩放后的粘贴的正方形图片，"
        scale_w, scale_h, scale, bg_img = scale_img(img, scale_side)
        # "重新保存图片名称"
        bg_img.save("{}\{}.jpg".format(save_img, count))

        bboxes = []
        # "遍历items列表里的所有box"
        for box in items:
            # "获得类名"
            cls_name = box.getElementsByTagName("name")[0].childNodes[0].data
            # "获得坐标值，str转成int"
            x1 = int(box.getElementsByTagName("xmin")[0].childNodes[0].data)
            y1 = int(box.getElementsByTagName("ymin")[0].childNodes[0].data)
            x2 = int(box.getElementsByTagName("xmax")[0].childNodes[0].data)
            y2 = int(box.getElementsByTagName("ymax")[0].childNodes[0].data)
            cls = name_list.index(cls_name)

            "传入缩放后的图片的宽、高、缩放比、缩放边长，以及当前box的坐标，获得缩放后的box坐标"
            x1_, y1_, x2_, y2_ = scale_box(scale_w, scale_h, scale, scale_side, x1, y1, x2, y2)
            # "获得缩放后的bbox的宽和高"
            box_w = x2_ - x1_
            box_h = y2_ - y1_
            # "获得缩放后的bbox的中心点坐标"
            center_x = x1_ + box_w // 2
            center_y = y1_ + box_h // 2
            bbox = [cls, center_x, center_y, box_w, box_h]
            # print(filename)
            # print(bbox)
            imgdraw = ImageDraw.ImageDraw(bg_img)
            imgdraw.rectangle((x1_, y1_, x2_, y2_), outline="red", width=3)
            bboxes.extend(bbox)
        # print(bboxes)

        # "将一张图片的数据存入一行txt文件，保存数据的时候去掉列表的括号"
        f.write("{0}.jpg, {1} \n".format(str(count), str(bboxes).strip("[,]")))
        plt.imshow(bg_img)
        plt.pause(1)
        print(count)
        count += 1


if __name__ == '__main__':
    rawimage_path = r"../raw_data/images"
    rawlabel_path = r"../raw_data/labels"
    saveimage_path = r"../processed_data/image"
    savelabel_path = r"../processed_data/label/label.txt"

    process_data(rawimage_path, rawlabel_path, saveimage_path, savelabel_path, 416)

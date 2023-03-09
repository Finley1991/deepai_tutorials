import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import cfg
import os
from PIL import Image
import math

LABEL_FILE_PATH = "data/person_label.txt"
IMG_BASE_DIR = "data/"

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # coco数据集经验值
        std=[0.229, 0.224, 0.225])
])


def one_hot(sum_cls, i):
    b = np.zeros(sum_cls)
    b[i] = 1.
    return b


class MyDataset(Dataset):

    def __init__(self):
        with open(LABEL_FILE_PATH) as f:
            self.dataset = f.readlines()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        labels = {}
        line = self.dataset[index]
        strs = line.split()
        # print(strs)
        _img_data = Image.open(os.path.join(IMG_BASE_DIR, strs[0]))
        img_data = transforms(_img_data)
        _boxes = np.array(list(float(x) for x in strs[1:]))
        # _boxes = np.array(list(map(float, strs[1:])))
        # print(_boxes)
        boxes = np.split(_boxes, len(_boxes) // 5)
        # print(boxes)

        for feature_size, anchors in cfg.ANCHORS_GROUP.items():
            labels[feature_size] = np.zeros(shape=(feature_size, feature_size, 3, 5 + cfg.CLASS_NUM))
            for box in boxes:
                cls, cx, cy, w, h = box
                cx_offset, cx_index = math.modf(cx * feature_size / cfg.IMG_WIDTH)  # cx*13/416=cx*1/32=cx/32
                cy_offset, cy_index = math.modf(cy * feature_size / cfg.IMG_HEIGHT)
                for i, anchor in enumerate(anchors):
                    anchor_area = cfg.ANCHORS_GROUP_AREA[feature_size][i]
                    p_w, p_h = np.log(w / anchor[0]), np.log(h / anchor[1])
                    p_area = w * h
                    # 标签框与锚框同心，所以面积比即是IOU,相当于变相的放大IOU值，也是放大置信度标签
                    iou = min(p_area, anchor_area) / max(p_area, anchor_area)
                    # N H W C
                    labels[feature_size][int(cy_index), int(cx_index), i] = np.array(
                        [iou, cx_offset, cy_offset, p_w, p_h, *one_hot(cfg.CLASS_NUM, int(cls))])  # 10,i
        return labels[13], labels[26], labels[52], img_data


if __name__ == '__main__':
    x = one_hot(10, 2)
    print(x)
    print(*x)
    data = MyDataset()
    dataloader = DataLoader(data, 2, shuffle=True)
    for i, x in enumerate(dataloader):
        print(x[0].shape)
        print(x[1].shape)
        print(x[2].shape)
        print(x[3].shape)

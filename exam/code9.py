from torch.utils.data import Dataset,DataLoader
import torchvision
import numpy as np
import os
from PIL import Image
import math

class cfg:
    IMG_HEIGHT = 416
    IMG_WIDTH = 416
    CLASS_NUM = 10

    "anchor box是对coco数据集聚类获得"
    ANCHORS_GROUP_KMEANS = {
        52: [[10, 13], [16, 30], [33, 23]],
        26: [[30, 61], [62, 45], [59, 119]],
        13: [[116, 90], [156, 198], [373, 326]]}

    ANCHORS_GROUP = {
        13: [[360, 360], [360, 180], [180, 360]],
        26: [[180, 180], [180, 90], [90, 180]],
        52: [[90, 90], [90, 45], [45, 90]]}

    ANCHORS_GROUP_AREA = {
        13: [x * y for x, y in ANCHORS_GROUP[13]],
        26: [x * y for x, y in ANCHORS_GROUP[26]],
        52: [x * y for x, y in ANCHORS_GROUP[52]],
    }


transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225])
])

def one_hot(cls_num, i):
    b = np.zeros(cls_num)
    b[i] = 1.
    return b

class MyDataset(Dataset):

    def __init__(self,label_path,image_dir):
        self.label_apth=label_path
        self.image_dir=image_dir
        with open(self.label_apth) as f:
            self.dataset = f.readlines()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        labels = {}
        line = self.dataset[index]
        strs = line.split()
        _img_data = Image.open(os.path.join(self.image_dir, strs[0]))
        img_data = transforms(_img_data)
        _boxes = np.array(float(x) for x in strs[1:])
        _boxes = np.array(list(map(float, strs[1:])))
        boxes = np.split(_boxes, len(_boxes) // 5)

        for feature_size, anchors in cfg.ANCHORS_GROUP.items():
            labels[feature_size] = np.zeros(shape=(feature_size, feature_size, 3, 5 + cfg.CLASS_NUM))
            for box in boxes:
                cls, cx, cy, w, h = box
                cx_offset, cx_index = math.modf(cx * feature_size / cfg.IMG_WIDTH)
                cy_offset, cy_index = math.modf(cy * feature_size / cfg.IMG_WIDTH)
                for i, anchor in enumerate(anchors):
                    anchor_area = cfg.ANCHORS_GROUP_AREA[feature_size][i]
                    p_w, p_h = w / anchor[0], h / anchor[1]
                    # tw=np.log(w / anchor[0])
                    p_area = w * h
                    iou = min(p_area, anchor_area) / max(p_area, anchor_area)
                    labels[feature_size][int(cy_index), int(cx_index), i] = np.array(
                        [iou, cx_offset, cy_offset, np.log(p_w), np.log(p_h), *one_hot(cfg.CLASS_NUM, int(cls))])#10,i
        return labels[13], labels[26], labels[52], img_data
if __name__ == '__main__':
    x=one_hot(10,2)
    print(x)
    LABEL_FILE_PATH = "data/person_label.txt"
    IMG_BASE_DIR = "data/"

    data = MyDataset(LABEL_FILE_PATH,IMG_BASE_DIR)
    dataloader = DataLoader(data,2,shuffle=True)
    for target_13, target_26, target_52, img_data in dataloader:
        print(target_13.shape)
        print(target_26.shape)
        print(target_52.shape)
        print(img_data.shape)
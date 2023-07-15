"""
实现IOU代码
"""

import numpy as np


def iou(archer_box, boxes, isMin=False):
    archer_box_area = (archer_box[2] - archer_box[0]) * (archer_box[3] - archer_box[1])
    boxes_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    xx1 = np.maximum(archer_box[0], boxes[:, 0])
    yy1 = np.maximum(archer_box[1], boxes[:, 1])
    xx2 = np.minimum(archer_box[2], boxes[:, 2])
    yy2 = np.minimum(archer_box[3], boxes[:, 3])

    inter_area = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)

    if isMin:
        iou_score = np.true_divide(inter_area,np.minimum(archer_box_area, boxes_areas))
    else:
        iou_score = np.true_divide(inter_area, archer_box_area + boxes_areas - inter_area)
    return iou_score

if __name__ == '__main__':
    a = np.array([1,1,11,11])
    bs = np.array([[1,1,10,10],[11,11,20,20]])
    print(iou(a,bs))
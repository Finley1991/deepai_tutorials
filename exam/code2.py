import numpy as np


def iou(box, boxes, isMin = False):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)

    inter = w * h
    if isMin:
        ovr = np.true_divide(inter, np.minimum(box_area, area))
    else:
        ovr = np.true_divide(inter, (box_area + area - inter))

    return ovr


def nms(boxes, thresh=0.3, isMin = False):

    if boxes.shape[0] == 0:
        return np.array([])

    _boxes = boxes[(-boxes[:, 4]).argsort()]
    r_boxes = []

    while _boxes.shape[0] > 1:
        a_box = _boxes[0]
        b_boxes = _boxes[1:]

        r_boxes.append(a_box)

        # print(iou(a_box, b_boxes))

        index = np.where(iou(a_box, b_boxes,isMin) < thresh)
        _boxes = b_boxes[index]

    if _boxes.shape[0] > 0:
        r_boxes.append(_boxes[0])

    return np.stack(r_boxes)

if __name__ == '__main__':

    bs = np.array([[1, 1, 10, 10, 0.98], [1, 1, 9, 9, 0.8], [9, 8, 13, 20, 0.7], [6, 11, 18, 17, 0.85]])
    print((-bs[:,4]).argsort())
    print(nms(bs))
import numpy as np

def convert_to_square(bbox):
    square_bbox = bbox.copy()
    if bbox.shape[0] == 0:
        return np.array([])
    h = bbox[:, 3] - bbox[:, 1]
    w = bbox[:, 2] - bbox[:, 0]
    max_side = np.maximum(h, w)
    #中心点-最大边长一半
    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
    square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side
    square_bbox[:, 3] = square_bbox[:, 1] + max_side

    return square_bbox

if __name__ == '__main__':
    box=np.array([[16,36,25,94],[54,62,79,12]])
    print(convert_to_square(box))
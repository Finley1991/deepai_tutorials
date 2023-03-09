import xml.etree.ElementTree as ET
import numpy as np
import math
import glob


def iou(box, clusters):

    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")
    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    iou_ = intersection / (box_area + cluster_area - intersection)
    return iou_


def avg_iou(boxes, clusters):
    #用每一个标签框与聚类输出框做IOU，取IOU的最大值求平均作为最终聚类框的平均精度
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])

def translate_boxes(boxes):
    #将(x1,y1,x2,y2)转换成(w,h)
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)

def kmeans(boxes, k, dist=np.median):
    # print(boxes.shape)#(N,2)
    rows = boxes.shape[0]#标签框的数量:N
    distances = np.empty((rows, k)) # (N, 9)#创建空矩阵做后期IOU距离的填充
    # print(distances.shape)
    last_clusters = np.zeros((rows,))#(N,)#创建存放最终的距离的位置索引
    # print(last_clusters.shape)
    # np.random.seed(0)
    #从所有标签框中随机选取K个框作为簇群
    clusters = boxes[np.random.choice(rows, k, replace=False)]#(9,2)
    # print(clusters.shape)
    while True:
        for row in range(rows):
            #填充距离:(N,9)，更新距离矩阵
            distances[row] = 1 - iou(boxes[row], clusters)#1-当前框和随机选中K个框的iou，值越小说明两个框距离越小
        nearest_clusters = np.argmin(distances, axis=1)#获取所有标签框与随机框的距离最小的索引 (N,)
        # print(nearest_clusters.shape)
        #如果最终更新的距离索引等于最小值的索引则停止更新距离
        if (last_clusters == nearest_clusters).all():
            break
        for cluster in range(k):
            # print(cluster)
            # print(nearest_clusters)
            #如果标签框中K个随机框的索引和最小距离索引一样，则替换为最终的距离值
            #取出最小距离索引和当前K个框的索引一样的标签框，计算中值作为新的簇群
            clusters[cluster] = np.median(boxes[nearest_clusters == cluster], axis=0)
        #将最近的距离索引替换为最终的距离索引
        last_clusters = nearest_clusters

    return clusters


def load_dataset(path):
    dataset = []
    for xml_file in glob.glob("{}/*xml".format(path)):
        tree = ET.parse(xml_file)
        height = int(tree.findtext("./size/height"))
        width = int(tree.findtext("./size/width"))

        for obj in tree.iter("object"):
            xmin = float(obj.findtext("bndbox/xmin")) / width
            ymin = float(obj.findtext("bndbox/ymin")) / height
            xmax = float(obj.findtext("bndbox/xmax")) / width
            ymax = float(obj.findtext("bndbox/ymax")) / height
            xmin = np.float64(xmin)
            ymin = np.float64(ymin)
            xmax = np.float64(xmax)
            ymax = np.float64(ymax)
            if xmax == xmin or ymax == ymin:
                print(xml_file)
            dataset.append([xmax - xmin, ymax - ymin])#w,h
    return np.array(dataset)#所有标签框的宽、高:[N,2]


if __name__ == '__main__':
    ANNOTATIONS_PATH = r"D:\pycharmprojects\dataset\VOCdevkit\VOC2012\Annotations"  # xml文件所在文件夹
    CLUSTERS = 9  # 聚类数量，anchor数量
    INPUTDIM = 416  # 输入网络大小
    data = load_dataset(ANNOTATIONS_PATH)
    out = kmeans(data, k=CLUSTERS)
    print('Boxes:')
    # boxes = out[out[:, 0].argsort()]* INPUTDIM
    #按box面积大小排序
    boxes = out[(out[:, 0]*out[:, 1]).argsort()]* INPUTDIM
    print(boxes)
    print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
    final_anchors = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    print("Before Sort Ratios:\n {}".format(final_anchors))
    # 宽高比，可以看出VOC数据集竖着的框比较多
    print("After Sort Ratios:\n {}".format(sorted(final_anchors)))

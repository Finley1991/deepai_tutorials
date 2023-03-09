import numpy as np
from sklearn.metrics import precision_score,recall_score

def voc_ap(rec, prec, use_07_metric=False):
    """在给定的精度和召回率下计算VOC、AP。如果use_07_metric为true，则使用VOC 07 11点方法(默认:False)。.
    11点的AP计算就定义为在这11个recall下precision的平均值
    """
    if use_07_metric:  # 使用07年方法,11 个点
        ap = 0.
        # 2010年以前按recall等间隔取11个不同点处的精度值做平均(0., 0.1, 0.2, …, 0.9, 1.0)
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])  # 插值
            ap = ap + p / 11.
    else:  # 新方式，计算所有点
        # 新的 AP 计算方法，在rec和pre的前后添加标记值，形成闭合的值域
        mrec = np.concatenate(([0.], rec, [1.]))#召回率前后附加标记值，将P-R曲线的值填充完整:[0,0.0666, 0.1333, 0.1333, 0.4, 0.4666,1]
        mpre = np.concatenate(([0.], prec, [0.]))# 精度值前后也附加标记值，填完整的P-R曲线输入:[0,1., 0.6666, 0.6666, 0.4285, 0.3043,0]
        for i in range(mpre.size - 1, 0, -1):#从大到小循环精度
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i]) # 计算PR曲线下的面积，寻找点：相同召回率下的最大精度值
        # 更新召回率
        i = np.where(mrec[1:] != mrec[:-1])[0]#如果前后两个召回率不一样，则取出索引
        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])#取出不同召回率段下的精度面积求和：sum（recall*precautions）
        return ap

if __name__ == '__main__':
    # rec = np.array([0.0666, 0.1333, 0.1333, 0.4, 0.4666])
    # prec = np.array([1., 0.6666, 0.6666, 0.4285, 0.3043])
    rec =  np.array([1/6,2/6,2/6,2/6,2/6,3/6,4/6,4/6,4/6,4/6,  5/6, 5/6, 5/6, 5/6, 5/6, 6/6, 6/6, 6/6, 6/6, 6/6])
    prec = np.array([1/1,2/2,2/3,2/4,2/5,3/6,4/7,4/8,4/9,4/10,5/11,5/12,5/13,5/14,5/15,6/16,6/17,6/18,6/19,6/20])
    ap = voc_ap(rec,prec)
    print(ap)


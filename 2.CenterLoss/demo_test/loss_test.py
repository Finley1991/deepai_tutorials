import torch
import torch.nn as nn

def center_loss():
    #创建特征数据：5组特征数据
    data = torch.tensor([[3, 6], [5, 8], [7, 6], [6, 4], [4, 3]], dtype=torch.float32)
    #创建数据对应的类别标签：5个对应标签，共2个类别
    label = torch.tensor([0, 0, 1, 0, 1], dtype=torch.float32)
    #创建2个类别的中心点坐标
    center = torch.tensor([[1, 1], [9, 9]], dtype=torch.float32)


    # 把2个中心数据转换成5个数据，让center的大小和data及label对应，data-center
    #[  0,    0,     1,     0,     1  ]
    #[[1,1],[1,1],[9, 9],[1, 1],[9, 9]]
    # 索引查找：根据索引找到元素。找到索引label对应的数组元素center
    # 从label的第一个索引开始，获取每个label元素对应center的索引，来扩充center，方便和data做loss
    center_exp = center.index_select(dim=0, index=label.long())
    # print(center_exp)

    # 统计每个不重复的label元素出现的次数：【3，2】，方便做类别归一化
    # label，不重复的label个数（max+1）：2，最小label(min)：0，最大label（max）：1
    count = torch.histc(label, bins=int(max(label).item() + 1), min=int(min(label).item()), max=int(max(label).item()))
    # print(count)

    # 索引查找：统计每个label的重复次数：【3，2】&【0，0，1，0，1】=【3，3，2，3，2】
    count_exp = count.index_select(dim=0, index=label.long())
    # print(count_exp)

    # center loss
    #求完每个样本和每类中心点的欧氏距离后，得到的距离需要除以每个类别的样本数，
    # 相当于对每类样本的距离都做了归一化。最后再对归一化后的所有距离求均值获得损失。
    #每类样本距离都做归一化的好处是，能够让每类样本距离每类中心的距离大小都差不多，
    # 这样能将所有距离放在一个较小方差的范围内，计算平均损失的时候对于每类样本都比较公平，
    # 否则，不同类别的样本数量如果不均衡，会导致不同类别的距离值差异太大，更新距离的时候会出现不同类别的更新失衡，
    #由于是计算总距离的均值，对于样本数量较多的类别更新过慢，也不利于距离中心较远的样本的收敛。
    loss1 = torch.pow(data - center_exp, 2)
    loss2 = torch.sum(torch.pow(data - center_exp, 2), dim=1)
    loss3 = torch.div(torch.sum(torch.pow(data - center_exp, 2), dim=1), count_exp)
    loss = torch.mean(torch.div(torch.sum(torch.pow(data - center_exp, 2), dim=1), count_exp))
    print(data)
    print(center_exp)
    print(loss1)
    print(loss2)
    print(loss3)
    print(loss)
center_loss()

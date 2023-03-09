import torch
import matplotlib.pyplot as plt

data = torch.tensor([[3, 6], [5, 8], [7, 6], [6, 4], [4, 3]], dtype=torch.float32)
label = torch.tensor([0, 0, 1, 0, 1], dtype=torch.float32)
center = torch.tensor([[1, 1], [9, 9]], dtype=torch.float32)


#画出每类的每个点到每类中心点的连接直线
plt.plot([3,1],[6,1],c="r")#1
plt.plot([5,1],[8,1],c="r")#1
plt.plot([7,9],[6,9],c="b")#9
plt.plot([6,1],[4,1],c="r")#1
plt.plot([4,9],[3,9],c="b")#9
#
# #画出5个数据点，和两个类别中心点
plt.scatter(data[:,0],data[:,1],c="r")
plt.scatter(1,1,c="g",marker="*")
plt.scatter(9,9,c="y",marker="*")
plt.show()
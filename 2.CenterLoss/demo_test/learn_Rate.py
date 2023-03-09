import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.models import AlexNet
import matplotlib.pyplot as plt

model = AlexNet(num_classes=2)
optimizer = optim.SGD(params=model.parameters(), lr=0.05)
# lr_scheduler.StepLR()
# Assuming optimizer uses lr = 0.05 for all groups
# lr = 0.05     if epoch < 30
# lr = 0.005    if 30 <= epoch < 60
# lr = 0.0005   if 60 <= epoch < 90

"""
torch.optim.lr_scheduler.StepLR
学习率按学习轮次衰减：每隔N个轮次，对学习率乘以衰减值
"""
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)
plt.figure()
x = list(range(100))
y = []
for epoch in range(100):
    lr = scheduler.get_last_lr()
    scheduler.step(epoch=None)
    print(epoch, scheduler.get_last_lr()[0])
    y.append(scheduler.get_last_lr()[0])
plt.plot(x, y)
plt.show()
"""
0<epoch<30, lr = 0.05
30<=epoch<60, lr = 0.005
60<=epoch<90, lr = 0.0005
"""

"""
torch.optim.lr_scheduler.MultiStepLR
指定空间的学习率衰减
与StepLR相比，MultiStepLR可以设置指定的区间
"""

# 可以指定区间#
# lr_scheduler.MultiStepLR()
#  Assuming optimizer uses lr = 0.05 for all groups
# lr = 0.05     if epoch < 30
# lr = 0.005    if 30 <= epoch < 80
#  lr = 0.0005   if epoch >= 80
print()
plt.figure()
y.clear()
scheduler = lr_scheduler.MultiStepLR(optimizer, [30, 80], 0.1)
for epoch in range(100):
    scheduler.step()
    print(epoch, 'lr={:.6f}'.format(scheduler.get_last_lr()[0]))
    y.append(scheduler.get_last_lr()[0])
plt.plot(x, y)
plt.show()

"""
torch.optim.lr_scheduler.ExponentialLR
学习率指数衰减：gamma为底数，轮次为指数
"""
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
print()
plt.figure()
y.clear()
for epoch in range(100):
    print(epoch, 'lr={:.6f}'.format(scheduler.get_last_lr()[0]))
    scheduler.step()
    y.append(scheduler.get_last_lr()[0])
plt.plot(x, y)
plt.show()

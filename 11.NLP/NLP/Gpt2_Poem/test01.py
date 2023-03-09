import torch
a=torch.ones(4,4)
b=torch.tril(a)
print(b)
#对应轴上的数量等于原维度除以份数
x1,x2=a.chunk(2,dim=1)
print(x1.shape)
print(x2.shape)
import torch
from torch.nn import functional

a=torch.tensor([1.,2.,3.])
# print(a,a.dtype)
#二范数：求完平方和之后，再求开根号
print(torch.sqrt(torch.sum(torch.pow(a,2))))
print(torch.norm(a))

#生成数据：生成指定连续空间的均值和标准差的tensor
print(torch.normal(mean=torch.arange(1.,11.),std=torch.arange(0,1,0.1)))

#二范数最大值标准化：当前的数值除以所有数据二范数中最大值
print(functional.normalize(a,dim=-1))
print(a/torch.max(torch.sqrt(torch.sum(torch.pow(a,2)))))
print(a/torch.max(torch.norm(a)))
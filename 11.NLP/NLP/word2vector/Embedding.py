import torch
from  torch import nn
a = nn.Embedding(4,3)
b = nn.Parameter(torch.randn(4,3),requires_grad=True)
# print(a)
print(a.weight)
# print(b)
#Embedding对象的索引必须是tensor
#根据索引获取某个词的词向量
idx = torch.tensor([0])
#用当前词库a中的词成一句话，就是按照当前词库中词的索引组成一句话（向量组、张量）
idx1 = torch.tensor([0,1,3])
#组成两句话，索引长度必须一样，可以写诗
idx2 = torch.tensor([[1,0,1,3,2],[2,1,2,0,1]])
print(a(idx))
print(a(idx1))
print(a(idx2))

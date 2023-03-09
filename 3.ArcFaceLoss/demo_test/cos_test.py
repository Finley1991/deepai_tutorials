import torch
from torch.nn import functional
import math

#两个向量的余弦相似度
a=torch.tensor([1.,2.])
b=torch.tensor([4.,5.])

# cos_alpha=a@b/(torch.sqrt(torch.sum(torch.pow(a,2)))*torch.sqrt(torch.sum(torch.pow(b,2))))
cos_alpha=torch.matmul(a,b)/torch.mean(torch.norm(a,dim=-1)*torch.norm(b,dim=-1))
#相似度值
print(cos_alpha)
#弧度
print(torch.acos(cos_alpha))
#角度
print(math.degrees(torch.acos(cos_alpha)))

c=functional.normalize(a,dim=-1)
d=functional.normalize(b,dim=-1)

#数据做了normalize之后不会影响输出结果，但是会对两个向量做一定的缩放，防止相似度值大于1
cos_alpha=torch.matmul(c,d)/torch.mean(torch.norm(c,dim=-1)*torch.norm(d,dim=-1))
print(cos_alpha)

a_b=torch.cat((a,b))
min_value=torch.min(a_b)
max_value=torch.max(a_b)
mean_value=(min_value+max_value)/2

e=a-mean_value
f=b-mean_value
e=functional.normalize(e,dim=-1)
f=functional.normalize(f,dim=-1)

print(e,f)
cos_alpha=torch.matmul(e,f)/torch.mean(torch.norm(e,dim=-1)*torch.norm(f,dim=-1))
print(cos_alpha)

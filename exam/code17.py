import torch
from torch import nn
a = nn.Embedding(4,5)
print(a.weight)
idx = torch.tensor([[0,1,2,3],[3,2,1,0]])
print(a(idx))


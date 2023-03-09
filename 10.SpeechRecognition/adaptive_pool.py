import torch
from torch.nn import functional as F

x = torch.rand(1, 54080)
y = F.adaptive_avg_pool1d(x[None, ...], (8000))
print(y.shape)

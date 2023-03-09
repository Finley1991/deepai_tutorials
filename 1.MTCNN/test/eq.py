import torch
import numpy as np

a=torch.tensor([1.,2.,3.])
print(torch.lt(a,3))
x=np.array([1,2,3,4,5,6])
print(x[:len(x)//2].mean())
print(x[len(x)//2:].mean())
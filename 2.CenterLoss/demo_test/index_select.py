import torch
#创建一个一维数组，只有0轴
# input_tensor = torch.tensor([1,2,3,4,5])
# #获取tensor的索引【0，2，4】对应的值//【1，3，5】，共有一个轴，只能取0轴
# print(input_tensor.index_select(0,torch.tensor([0,2,4])))
# #创建一个二维数组，有0轴和1轴
input_tensor = torch.tensor([[1,2,3,4,5],[6,7,8,9,10]])
# #获取第0轴的第1个元素，为【【6,7,8,9,10】】
print(input_tensor.index_select(0,torch.tensor([0,0,1,0,1])))
# #获取第1轴的第1个元素，为【【2】,【7】】
# print(input_tensor.index_select(1,torch.tensor([1])))

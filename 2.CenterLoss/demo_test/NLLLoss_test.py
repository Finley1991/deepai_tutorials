import torch
# torch.manual_seed(0) # 为CPU设置随机种子
feature=torch.randn([100,10])
label=torch.randint(0,10,(100,))
CrossEntropyLoss=torch.nn.CrossEntropyLoss()
loss1=CrossEntropyLoss(feature,label)

softmax=torch.softmax(feature,1)
log=torch.log(softmax)
NLLLoss=torch.nn.NLLLoss()#NLLLoss的作用是对log的输出去掉负号，再求均值
loss2=NLLLoss(log,label)
log_softmax=torch.log_softmax(feature,1)
loss3=NLLLoss(log_softmax,label)
mean_log=torch.mean(-log)

print(loss1)
print(loss2)
print(loss3)
print(mean_log)
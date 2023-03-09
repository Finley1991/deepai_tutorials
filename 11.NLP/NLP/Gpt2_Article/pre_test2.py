from Gpt2_Article.module import *
from Gpt2_Article.dataset import *
import traceback
import os
import config
vocab_path = "./data/vocab.txt"
weight_file = r"./weight/weight2.pth"


# 调用网络
net = GPT2().to(torch.device(config.device))
if os.path.exists(weight_file):
    print("有网络参数文件")
    if os.path.getsize(weight_file) != 0:
        net.load_state_dict(torch.load(weight_file))
        print("加载保存的参数成功")
    else:
        print("网络文件里没有网络参数")
net.eval()

#以只读方式发开字典，并将所有字拿到后装在一个列表里
with open(vocab_path, "r+", encoding="utf-8") as f:
    tokens = f.read().split()
# print(tokens)
while True:
    try:
        # 给定一个开始词
        print("\n请输入开始词:")
        inputs = input()
        x_index = []
        for i, token in enumerate(inputs):
            #得到每个输入词在字典中的索引，并装在列表中
            x_index.append(tokens.index(token))

        os = []
        # 将输入词在字典中的索引组合转到对应设备上
        x = torch.tensor([x_index]).to(torch.device(config.device))
        # 对索引组合进行位置编码，从0到len(索引组合)
        p = torch.tensor([[a for a in range(len(x_index))]]).to(torch.device(config.device))

        #新生成（加上输入字长度)总共500个字的文章
        for i in range(len(x_index)-1, 500):
            #将输入词和位置都传入模型
            y = net(x, p)
            # print(y.shape)
            # [nsv]=[1,s,v],取最后一个s,shape=[1,1,v],v是固定的658
            y = y[:,-1:,:]
            # print(y.shape)
            # 预测的所有字中，选概率最大的8个字的向量内积值和在字典中的位置，取最后一个轴:v
            v, y = torch.topk(y, 8, dim=int(-1))
            v, y = v.reshape(-1, 8), y.reshape(-1, 8)
            # print(v)
            # print(y)
            # 将预测结果中概率最大的8个字的向量内积值转成概率值，再通过概率随机选一个
            v = torch.multinomial(torch.softmax(v, dim=-1), 1)
            # 通过索引拿到取的值(在字典里的索引值)
            y = torch.gather(y, -1, v)
            # 将生成的字拼接到前面去,周而复始，x一直在增加。
            x = torch.cat([x, y], dim=1)
            # 加一个位置，i从0 开始，加的位置从1开始加，range(2)=[0,1]
            p = torch.tensor([range(i + 2)]).to(torch.device(config.device))

        # 通过x把token取出来，x是索引
        for i, index in enumerate(x.detach().cpu()[0]):
            #当获取的字不是[Ent]的时候不换行输出
            if tokens[index] != "[Ent]":
                print(tokens[index], end="")
            #否则如果获取的字是[Ent]的时候就换行，使用print自带的换行
            else:
                print()
    except:
        print("no such word!")
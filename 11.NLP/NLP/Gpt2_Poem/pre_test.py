from Gpt2_Poem.module import *
from Gpt2_Poem.dataset import *
import traceback
import os

vocab_path = "./data/vocab.txt"
weight_file = r"./weight/weight.pth"


# 网络
net = GPT2(isMask=False).to(torch.device(cfg.device))
if os.path.exists(weight_file):
    print("有网络参数文件")
    if os.path.getsize(weight_file) != 0:
        net.load_state_dict(torch.load(weight_file))
        print("加载保存的参数成功")
    else:
        print("网络文件里没有网络参数")
        exit()
net.eval()

#以只读方式发开字典，并将所有字拿到后装在一个列表里
with open(vocab_path, "r+", encoding="utf-8") as f:
    tokens = f.read().split()
while True:
    try:
        # 给定一个开始词
        print("\n请输入开始词:")
        x = input()
        if x=="end":
            break
        x_index = []
        for i, token in enumerate(x):
            # 得到每个输入词在字典中的索引，并装在列表中
            x_index.append(tokens.index(token))

        # 将输入词在字典中的索引组合转到对应设备上
        x = torch.tensor([x_index]).to(torch.device(cfg.device))
        # 对索引组合进行位置编码，从0到len(索引组合)
        p = torch.tensor([[a for a in range(len(x_index))]]).to(torch.device(cfg.device))

        #新生成（加上输入字长度）总共51个字的诗词
        for i in range(len(x_index)-1, 51):
            # 将输入词和位置都传入模型
            y = net(x, p)
            # print(y.shape)
            # [nsv]=[1,s,v],取最后一个s,shape=[1,1,v],v是固定的305
            y = y[:, -1:]
            # print(y.shape)
            # 预测的所有字中，选概率最大的10个字的向量内积值和在字典中的位置，取最后一个轴:v
            v, y = torch.topk(y, 10, dim=int(-1))
            v, y = v.reshape(-1, 10), y.reshape(-1, 10)
            # print(v)
            # print(y)
            # 将预测结果中概率最大的10个字的向量值转成概率值，再通过概率随机选一个，结果为索引值
            v = torch.multinomial(torch.softmax(v, dim=-1), 1)
            # print(v)
            # 通过索引拿到取的值(在字典里的索引值)
            y = torch.gather(y, -1, v)
            # 将生成的字拼接到后面去,周而复始，x一直在增加。
            x = torch.cat([x, y], dim=1)
            # print(len(x.detach().cpu()[0]))
            # 更新位置变量p,加一个位置，i从0 开始，加的位置从1开始加，range(2)=[0,1],[0,1,2],...
            p = torch.tensor([range(i + 2)]).to(torch.device(cfg.device))
            # print(p)
        # 通过x把token取出来，x是索引
        for i, index in enumerate(x.detach().cpu()[0]):
            #当获取的字不是[Ent]的时候不换行输出
            if tokens[index] != "[Ent]":
                print(tokens[index], end="")
            #否则如果获取的字是[Ent]的时候就换行
            else:
                # 使用print自带的换行
                print()

    except:
        print("no such word!")
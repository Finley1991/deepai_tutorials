import Gpt2_Poem.config as cfg
import torch
import os
from torch.utils.data import Dataset

class MyDataset(Dataset):

    def __init__(self, dir):
        self.dataset = []

        for filename in os.listdir(dir):
            with open(os.path.join(dir, filename), "r+") as f:
                #读出语料库的索引表token里的位置编码，放在列表里面：610
                ws = [int(x) for x in f.readline().split()]
                # print(len(ws))#:610
                ws_len = len(ws)
                start = 0
                # 将语料库索引表分段表示，可按每首诗长度来分成数据集，或者按照更大的步长来分
                #当拿到剩余的长度大于52个时候：
                # cfg.pos_num=26：20(字)+2(逗号)+2(句号)+2(换行)
                "可以一次学习多首诗，26的倍数"
                while ws_len - start >= cfg.pos_num:
                    #ws[0:26],ws[13:39],......
                    #ws[0:52],ws[13:65],......
                    self.dataset.append(ws[start:start + cfg.pos_num])
                    # print(ws[start:start + cfg.pos_num])
                    # 步长表示每条样本之间在语料库中的相隔的字数，
                    # 步长越小学习的结果越多样化，样本也越多；步长越大，样本集越少
                    #步长为26 ，刚好为一首诗，也就是一首一首的学，学出的结果更加接近原来的诗词，容易过拟合
                    # start += cfg.pos_num # +26
                    #半首半首重复的学
                    start += 13
                    #步长为1，样本集最多，对于诗词学习不太适合
                    # start += 1
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = torch.tensor(self.dataset[index])
        # 【0，51】，【1，52】
        #前半段为输入，后半段为标签，所以学同一类型的文章较好，不然会不伦不类
        return data[0:-1], data[1:]


if __name__ == '__main__':
    myDataset = MyDataset("./data/books_tokenized")
    print(len(myDataset))#按诗的长度做数据集就是多少首诗

    #按半首诗学习
    print(myDataset[0])#第一条学习样本：[0:52]
    print(myDataset[1])#第二条学习样本：[13:65]

    print(myDataset[0][0].shape)#第一条样本的输入：[0:51]
    print(myDataset[0][1].shape)#第一条样本的标签：[1:52]
